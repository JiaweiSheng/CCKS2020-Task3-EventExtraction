import torch
import time
import warnings
from pathlib import Path
from pybert.train.losses import BCEWithLogLoss
from pybert.train.trainer import Trainer
from torch.utils.data import DataLoader
from pybert.io.utils import collate_fn
from pybert.io.bert_processor import BertProcessor
from pybert.common.tools import init_logger, logger
from pybert.common.tools import seed_everything
from pybert.model.bert_for_multi_label import BertForMultiLable
from pybert.preprocessing.preprocessor import ChinesePreProcessor
from pybert.callback.modelcheckpoint import ModelCheckpoint
from pybert.callback.trainingmonitor import TrainingMonitor
from pybert.train.metrics import *
from pybert.callback.optimizater.adamw import AdamW
from pybert.callback.lr_schedulers import get_linear_schedule_with_warmup
from torch.utils.data import RandomSampler, SequentialSampler
import pandas as pd
import numpy as np
import os
from args_sigmoid import parse_args

warnings.filterwarnings("ignore")


def run_train(args, label_list):
    # 数据预处理和ids，BERT必备
    processor = BertProcessor(vocab_path=args.bert_vocab_path, do_lower_case=args.do_lower_case)
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}

    train_data = processor.get_train(args.train_data_path)
    train_examples = processor.create_examples(lines=train_data,
                                               example_type='train',
                                               cached_examples_file=args.cache_dir + '/' + f"cached_train_examples_{args.arch}")
    train_features = processor.create_features(examples=train_examples,
                                               max_seq_len=args.train_max_seq_len,
                                               cached_features_file=args.cache_dir + '/' + "cached_train_features_{}_{}".format(
                                                   args.train_max_seq_len, args.arch
                                               ))
    train_dataset = processor.create_dataset(train_features, is_sorted=args.sorted)

    if args.sorted:
        train_sampler = SequentialSampler(train_dataset)
    else:
        train_sampler = RandomSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)

    valid_data = processor.get_dev(args.valid_data_path)
    valid_examples = processor.create_examples(lines=valid_data,
                                               example_type='valid',
                                               cached_examples_file=args.cache_dir + '/' + f"cached_valid_examples_{args.arch}")

    valid_features = processor.create_features(examples=valid_examples,
                                               max_seq_len=args.eval_max_seq_len,
                                               cached_features_file=args.cache_dir + '/' + "cached_valid_features_{}_{}".format(
                                                   args.eval_max_seq_len, args.arch
                                               ))
    valid_dataset = processor.create_dataset(valid_features)
    valid_sampler = SequentialSampler(valid_dataset)
    valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.eval_batch_size,
                                  collate_fn=collate_fn)

    # 加载BERT，并定义分类模型
    logger.info("initializing model")
    if args.resume_path:
        args.resume_path = Path(args.resume_path)
        model = BertForMultiLable.from_pretrained(args.resume_path, num_labels=len(label_list))
    else:
        model = BertForMultiLable.from_pretrained(args.bert_model_dir, num_labels=len(label_list))
        BertForMultiLable.unfreeze(model, 16, 23)

    t_total = int(len(train_dataloader) / args.gradient_accumulation_steps * args.epochs)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    # 监视器，训练并记录
    logger.info("initializing callbacks")
    # train_monitor = TrainingMonitor(file_dir=args.figure_dir, arch=args.arch)
    model_checkpoint = ModelCheckpoint(checkpoint_dir=args.checkpoint_dir, mode=args.mode,
                                       monitor=args.monitor, arch=args.arch,
                                       save_best_only=args.save_best)

    # **************************** training model ***********************
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    trainer = Trainer(args=args, model=model, logger=logger, criterion=BCEWithLogLoss(), optimizer=optimizer,
                      scheduler=scheduler, early_stopping=None, training_monitor=None,
                      model_checkpoint=model_checkpoint,
                      batch_metrics=[
                          AccuracyThresh(thresh=args.thresh, normalizate=True)],
                      epoch_metrics=[
                          AccuracyThresh(thresh=args.thresh, normalizate=True),
                          F1Score(thresh=args.thresh, normalizate=True, task_type='binary', average='macro',
                                  search_thresh=False),
                          F1Score(thresh=args.thresh, normalizate=True, task_type='binary', average='micro',
                                  search_thresh=False),
                          MultiLabelReportAucF1(id2label=id2label, thresh=args.thresh),
                          F1Score(thresh=None, normalizate=True, task_type='binary', average='macro',
                                  search_thresh=True),
                      ])
    trainer.train(train_data=train_dataloader, valid_data=valid_dataloader)


def run_valid(args, label_list):
    # 数据预处理和ids，BERT必备
    processor = BertProcessor(vocab_path=args.bert_vocab_path, do_lower_case=args.do_lower_case)
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}

    train_data = processor.get_train(args.train_data_path)
    train_examples = processor.create_examples(lines=train_data,
                                               example_type='train',
                                               cached_examples_file=args.cache_dir + '/' + f"cached_train_examples_{args.arch}")
    train_features = processor.create_features(examples=train_examples,
                                               max_seq_len=args.train_max_seq_len,
                                               cached_features_file=args.cache_dir + '/' + "cached_train_features_{}_{}".format(
                                                   args.train_max_seq_len, args.arch
                                               ))
    train_dataset = processor.create_dataset(train_features, is_sorted=args.sorted)

    if args.sorted:
        train_sampler = SequentialSampler(train_dataset)
    else:
        train_sampler = RandomSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)

    valid_data = processor.get_dev(args.valid_data_path)
    valid_examples = processor.create_examples(lines=valid_data,
                                               example_type='valid',
                                               cached_examples_file=args.cache_dir + '/' + f"cached_valid_examples_{args.arch}")

    valid_features = processor.create_features(examples=valid_examples,
                                               max_seq_len=args.eval_max_seq_len,
                                               cached_features_file=args.cache_dir + '/' + "cached_valid_features_{}_{}".format(
                                                   args.eval_max_seq_len, args.arch
                                               ))
    valid_dataset = processor.create_dataset(valid_features)
    valid_sampler = SequentialSampler(valid_dataset)
    valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.eval_batch_size,
                                  collate_fn=collate_fn)
    # BERT 处理
    model = BertForMultiLable.from_pretrained(args.checkpoint_dir, num_labels=len(label_list))

    trainer = Trainer(args=args, model=model, logger=logger, criterion=BCEWithLogLoss(), optimizer=None,
                      scheduler=None, early_stopping=None, training_monitor=None,
                      model_checkpoint=None,
                      batch_metrics=[
                          AccuracyThresh(thresh=args.thresh, normalizate=True)],
                      epoch_metrics=[
                          AccuracyThresh(thresh=args.thresh, normalizate=True),
                          F1Score(thresh=args.thresh, normalizate=True, task_type='binary', average='macro',
                                  search_thresh=False),
                          F1Score(thresh=args.thresh, normalizate=True, task_type='binary', average='micro',
                                  search_thresh=False),
                          MultiLabelReportAucF1(id2label=id2label, thresh=args.thresh),
                          F1Score(thresh=None, normalizate=True, task_type='binary', average='macro',
                                  search_thresh=True),
                      ])
    performance = trainer.valid_epoch(valid_dataloader)

    with open(args.checkpoint_dir+'/valid_performance', 'w', encoding='utf-8') as f:
        import json
        json.dump(performance, f, ensure_ascii=False, indent='\t')


def run_test(args, label_list):
    from split_data_by_ids import TaskData
    from pybert.test.predictor import Predictor
    # 加载无标签的数据
    taskdata = TaskData()
    target_lst = [-1] * len(label_list)
    ids, targets, sentences = taskdata.read_data(raw_data_path=args.test_raw_data_path,
                                                 preprocessor=ChinesePreProcessor(),
                                                 is_train=False, target_lst=target_lst)
    test_data = []
    for i in range(len(ids)):
        test_data.append((sentences[i], targets[i]))

    processor = BertProcessor(vocab_path=args.bert_vocab_path, do_lower_case=args.do_lower_case)
    id2label = {i: label for i, label in enumerate(label_list)}

    test_examples = processor.create_examples(lines=test_data,
                                              example_type='test',
                                              cached_examples_file=args.cache_dir + '/' + f"cached_test_examples_{args.arch}")
    test_features = processor.create_features(examples=test_examples,
                                              max_seq_len=args.eval_max_seq_len,
                                              cached_features_file=args.cache_dir + '/' + "cached_test_features_{}_{}".format(
                                                  args.eval_max_seq_len, args.arch
                                              ))
    test_dataset = processor.create_dataset(test_features)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.train_batch_size,
                                 collate_fn=collate_fn)
    model = BertForMultiLable.from_pretrained(args.checkpoint_dir, num_labels=len(label_list))

    logger.info('model predicting....')
    predictor = Predictor(model=model,
                          logger=logger,
                          n_gpu=args.n_gpu,
                          normalize=True)
    result = predictor.predict(data=test_dataloader)
    ids = np.array(ids)
    df1 = pd.DataFrame(ids, index=None)
    df2 = pd.DataFrame(result, index=None)
    all_df = pd.concat([df1, df2], axis=1)

    all_df.columns = ['id'] + label_list
    all_df.to_csv(args.results_output+'_proba', index=False)
    for label in label_list:
        all_df[label] = all_df[label].apply(lambda x: 1 if x > args.thresh else 0)
    all_df.to_csv(args.results_output, index=False)


def main():
    args = parse_args()
    seed_everything(args.seed)

    init_logger(log_file=args.log_dir + f'{args.arch}-{time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())}.log')
    args.checkpoint_dir = args.checkpoint_dir + '/' + args.arch
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    # torch.save(args, args.checkpoint_dir + '/training_args.bin')

    logger.info("Training/evaluation parameters %s", args)

    assert args.type in ['base', 'trans']
    if args.type == 'base':
        label_list = ['质押', '股份股权转让', '投资', '起诉', '减持']
    elif args.type == 'trans':
        label_list = ['收购', '判决', '签署合同', '担保', '中标']
    else:
        return

    if args.do_train:
        run_train(args, label_list)
        run_valid(args, label_list)

    if args.do_test:
        run_test(args, label_list)

    if args.do_valid:
        run_valid(args, label_list)


if __name__ == '__main__':
    main()
