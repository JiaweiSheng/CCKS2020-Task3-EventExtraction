# -*- encoding:utf -*-
"""
  This script provides an _example to wrap UER-py for NER.
"""
import random
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from uer.model_builder import build_model
from uer.utils.config import load_hyperparam
from uer.utils.optimizers import *
from uer.utils.constants import *
from uer.utils.vocab import Vocab
from uer.utils.seed import set_seed
from torch.utils.data import DataLoader

from uer.model_saver import save_model
from uer.model_loader import load_model
from v9.args import parse_args
from v9.utils import *
from v9.model import BertTagger
from v9.predict_for_eval import *
import os


def seed_everything(seed=7):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(args, model, dev_data_loader, device, ty_args_id, evaluate_type=None):
    if hasattr(model, "module"):
        model = model.module
    model.eval()
    dev_len = 0
    t_ps, t_rs, t_fs, a_ps, a_rs, a_fs = 0, 0, 0, 0, 0, 0
    for i, (idx, dt, token, seg, mask, t_index, r_pos, t_m, t_t, a_t, word_segs, word_poss, ners, deps) in tqdm(
            enumerate(dev_data_loader)):
        dt = dt[0]
        if evaluate_type:
            if dt not in evaluate_type:
                continue
            else:
                pass
        else:
            pass
        token = torch.LongTensor(token).to(device)
        seg = torch.LongTensor(seg).to(device)
        mask = torch.LongTensor(mask).to(device)
        r_pos = torch.LongTensor(r_pos).to(device)
        t_m = torch.LongTensor(t_m).to(device)
        word_segs = torch.LongTensor(word_segs).to(device)
        word_poss = torch.LongTensor(word_poss).to(device)
        ners = torch.LongTensor(ners).to(device)
        deps = torch.LongTensor(deps).to(device)
        t_p, t_r, t_f, a_p, a_r, a_f = evaluate_one(model, args, dt, token, seg, mask, r_pos, t_m, t_t, a_t, ty_args_id,
                                                    word_segs, word_poss, ners, deps)
        t_ps += t_p
        t_rs += t_r
        t_fs += t_f
        a_ps += a_p
        a_rs += a_r
        a_fs += a_f
        dev_len += 1
    t_ps /= dev_len
    t_rs /= dev_len
    t_fs /= dev_len
    a_ps /= dev_len
    a_rs /= dev_len
    a_fs /= dev_len
    return t_ps, t_rs, t_fs, a_ps, a_rs, a_fs


def main():
    args = parse_args()
    seed_everything(args.seed)

    vocab = Vocab()
    vocab.load(args.vocab_path)
    args.vocab = vocab

    args_s_vocab, args_e_vocab = get_args2id()
    args.args_num = len(args_s_vocab.keys())
    args_id, id_args, ty_args, ty_args_id = get_dict()

    args.target = "bert"
    model = build_model(args)

    # Build sequence labeling model.
    model = BertTagger(args, model, pos_emb_size=args.rp_size)
    model.load_state_dict(torch.load(args.pretrained_downstream_path))

    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # Training phase.
    print("Start training.")
    train_set = Data(task='train', fn=args.train_path, vocab=vocab, seq_len=args.seq_length, args_vocab_s=args_s_vocab,
                     args_vocab_e=args_e_vocab)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn_train)

    dev_set = Data(task='dev', fn=args.dev_path, vocab=vocab, seq_len=args.seq_length, args_vocab_s=args_s_vocab,
                   args_vocab_e=args_e_vocab)
    dev_loader = DataLoader(dev_set, batch_size=1, shuffle=False,
                            collate_fn=collate_fn_dev)

    instances_num = len(train_set)
    train_steps = int(instances_num * args.epochs_num / args.batch_size) + 1

    print("Batch size: ", args.batch_size)
    print("The number of training instances:", instances_num)
    print("The number of evaluating instances:", len(dev_set))

    bert_embedding_params = list(map(id, model.embedding.parameters()))
    bert_encoder_params = list(map(id, model.encoder.parameters()))

    other_params = filter(lambda p: id(p) not in bert_embedding_params + bert_encoder_params,
                          model.parameters())
    optimizer_grouped_parameters = [{'params': model.embedding.parameters()},
                                    {'params': model.encoder.parameters()},
                                    {'params': other_params, 'lr': 1e-4}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=train_steps * args.warmup, t_total=train_steps)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    # going to train
    total_loss = 0.0
    best_f1 = 0.0
    best_epoch = 0
    for epoch in range(1, args.epochs_num + 1):
        print('Training...')
        model.train()
        for i, (idx, dt, token, seg, mask, t_index, r_pos, t_m, t_s, t_e, a_s, a_e, a_m, word_segs, word_poss,
                ners, deps) in enumerate(train_loader):
            model.zero_grad()
            token = torch.LongTensor(token).to(device)
            seg = torch.LongTensor(seg).to(device)
            mask = torch.LongTensor(mask).to(device)
            r_pos = torch.LongTensor(r_pos).to(device)
            t_m = torch.LongTensor(t_m).to(device)
            t_s = torch.FloatTensor(t_s).to(device)
            t_e = torch.FloatTensor(t_e).to(device)
            a_s = torch.FloatTensor(a_s).to(device)
            a_e = torch.FloatTensor(a_e).to(device)
            a_m = torch.LongTensor(a_m).to(device)
            word_segs = torch.LongTensor(word_segs).to(device)
            word_poss = torch.LongTensor(word_poss).to(device)
            ners = torch.LongTensor(ners).to(device)
            deps = torch.LongTensor(deps).to(device)
            loss = model(token, seg, mask, t_s, t_e, r_pos, t_m, a_s, a_e, a_m, word_segs, word_poss, ners, deps)

            if torch.cuda.device_count() > 1:
                loss = torch.mean(loss)
            # loss = torch.abs(loss-0.001)+0.001
            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0:
                print("Epoch id: {}, Training steps: {}, Avg loss: {:.6f}".format(epoch, i + 1,
                                                                                  total_loss / args.report_steps))
                total_loss = 0.

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            optimizer.step()
            scheduler.step()

        print('Evaluating...')

        t_ps, t_rs, t_fs, a_ps, a_rs, a_fs = evaluate(args, model, dev_loader, device, ty_args_id)
        f1_mean_all = (t_fs + a_fs) / 2
        print('Evaluate on all types:')
        print("Epoch id: {}, Trigger P: {:.3f}, Trigger R: {:.3f}, Trigger F: {:.3f}".format(epoch, t_ps, t_rs, t_fs))
        print("Epoch id: {}, Args P: {:.3f}, Args R: {:.3f}, Args F: {:.3f}".format(epoch, a_ps, a_rs, a_fs))
        print("Epoch id: {}, F1 Mean All: {:.3f}".format(epoch, f1_mean_all))

        evaluate_type = ['收购', '判决', '签署合同', '担保', '中标']
        t_ps, t_rs, t_fs, a_ps, a_rs, a_fs = evaluate(args, model, dev_loader, device, ty_args_id,
                                                      evaluate_type=evaluate_type)
        f1_mean = (t_fs + a_fs) / 2
        print('Evaluate on transfer types:')
        print("Epoch id: {}, Trigger P: {:.3f}, Trigger R: {:.3f}, Trigger F: {:.3f}".format(epoch, t_ps, t_rs, t_fs))
        print("Epoch id: {}, Args P: {:.3f}, Args R: {:.3f}, Args F: {:.3f}".format(epoch, a_ps, a_rs, a_fs))
        print("Epoch id: {}, F1 Mean: {:.3f}".format(epoch, f1_mean))
        if f1_mean > best_f1:
            best_f1 = f1_mean
            best_epoch = epoch
            save_model(model, args.output_model_path)
        print("The Best F1 Is: {:.3f}, When Epoch Is: {}".format(best_f1, best_epoch))
        # save_model(model, args.output_model_path)

    # Evaluation phase.
    if args.test_path is not None:
        print("Test set evaluation.")
        dev_set = Data(task='dev', fn=args.test_path, vocab=vocab, seq_len=args.seq_length, args_vocab_s=args_s_vocab,
                       args_vocab_e=args_e_vocab)
        dev_loader = DataLoader(dev_set, batch_size=1, shuffle=False,
                                collate_fn=collate_fn_dev)
        model = load_model(model, args.output_model_path)
        model = model.to(device)

        t_ps, t_rs, t_fs, a_ps, a_rs, a_fs = evaluate(args, model, dev_loader, device, ty_args_id)
        f1_mean_all = (t_fs + a_fs) / 2
        print('Evaluate on all types:')
        print("Trigger P: {:.3f}, Trigger R: {:.3f}, Trigger F: {:.3f}".format(t_ps, t_rs, t_fs))
        print("Args P: {:.3f}, Args R: {:.3f}, Args F: {:.3f}".format(a_ps, a_rs, a_fs))
        print("F1 Mean All: {:.3f}".format(f1_mean_all))

        evaluate_type = ['收购', '判决', '签署合同', '担保', '中标']
        t_ps, t_rs, t_fs, a_ps, a_rs, a_fs = evaluate(args, model, dev_loader, device, ty_args_id,
                                                      evaluate_type=evaluate_type)
        f1_mean = (t_fs + a_fs) / 2
        print('Evaluate on transfer types:')
        print("Trigger P: {:.3f}, Trigger R: {:.3f}, Trigger F: {:.3f}".format(t_ps, t_rs, t_fs))
        print("Args P: {:.3f}, Args R: {:.3f}, Args F: {:.3f}".format(a_ps, a_rs, a_fs))
        print("F1 Mean: {:.3f}".format(f1_mean))


if __name__ == "__main__":
    main()
