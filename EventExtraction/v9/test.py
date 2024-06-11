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
import json
from uer.model_saver import save_model
from uer.model_loader import load_model
from v9.args import parse_args
from v9.utils import *
from v9.model import BertTagger
# from v9.score import *
from v9.predict_for_test import *
import os


def seed_everything(seed=7):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(args, model, dev_data_loader, device, id_args, ty_args_id):
    if torch.cuda.device_count() > 1:
        model = model.module
    model.eval()
    results_dict = {}  # 全部的预测结果
    for i, (idx, dt, dc, token, seg, mask, word_segs, word_poss, ners, deps) in tqdm(enumerate(dev_data_loader)):
        idx = idx[0]
        dt = dt[0]
        dc = dc[0]
        if idx not in results_dict:
            results_dict[idx] = []
        token = torch.LongTensor(token).to(device)
        seg = torch.LongTensor(seg).to(device)
        mask = torch.LongTensor(mask).to(device)
        word_segs = torch.LongTensor(word_segs).to(device)
        word_poss = torch.LongTensor(word_poss).to(device)
        ners = torch.LongTensor(ners).to(device)
        deps = torch.LongTensor(deps).to(device)
        extract_all_items_for_one(model, device, idx, dt, dc, token, seg, mask, args.seq_length, args.args_num,
                                  args.threshold_1, args.threshold_2, args.threshold_3, args.threshold_4,
                                  args.threshold_trigger, args.threshold_args,
                                  results_dict, id_args, ty_args_id, word_segs, word_poss, ners, deps)
    return results_dict


def post_results(results_dict, fn):
    records = []
    for idx in results_dict:
        events = results_dict[idx]
        record = {'id': idx, 'events': events}
        records.append(record)
    with open(fn, 'w', encoding='utf-8') as f:
        for record in records:
            record_json = json.dumps(record, ensure_ascii=False)
            f.write(record_json + '\n')


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
    model = BertTagger(args, model, pos_emb_size=args.rp_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model, args.output_model_path)
    model = model.to(device)

    if args.batch_size != 1:
        print('Reset batch_size=1')
        args.batch_size = 1
    print("Test set evaluation.")
    dev_set = Data(task='test', fn=args.test_path, vocab=vocab, seq_len=args.seq_length, args_vocab_s=args_s_vocab,
                   args_vocab_e=args_e_vocab)
    print("The number of testing instances:", len(dev_set))
    dev_loader = DataLoader(dev_set, batch_size=1, shuffle=False,
                            collate_fn=collate_fn_test)
    results_dict = evaluate(args, model, dev_loader, device, id_args, ty_args_id)

    post_results(results_dict, fn=args.results_path)


if __name__ == "__main__":
    main()
