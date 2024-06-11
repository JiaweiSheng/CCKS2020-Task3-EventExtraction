import random
import argparse

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from uer.model_builder import build_model
from uer.utils.config import load_hyperparam
from uer.utils.optimizers import *
from uer.utils.constants import *
from uer.utils.vocab import Vocab
from uer.utils.seed import set_seed
from uer.model_saver import save_model
from uer.model_loader import load_model


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--pretrained_model_path", default='models/mixed_large_24_model.bin', type=str,
                        help="Path of the pretrained model.")
    parser.add_argument("--output_model_path", default="./models/task_model.bin", type=str,
                        help="Path of the output model.")
    parser.add_argument("--vocab_path", type=str, default='models/google_zh_vocab.txt',
                        help="Path of the vocabulary file.")
    parser.add_argument("--train_path", type=str, required=True,
                        help="Path of the trainset.")
    parser.add_argument("--dev_path", type=str, required=True,
                        help="Path of the devset.")
    parser.add_argument("--test_path", type=str, required=True,
                        help="Path of the testset.")
    parser.add_argument("--config_path", default="./models/bert_large_config.json", type=str,
                        help="Path of the config file.")

    # Model options.
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch_size.")
    parser.add_argument("--seq_length", default=400, type=int,
                        help="Sequence length.")
    parser.add_argument("--embedding", choices=["bert", "word"], default="bert",
                        help="Emebdding type.")
    parser.add_argument("--encoder", choices=["bert", "lstm", "gru", \
                                              "cnn", "gatedcnn", "attn", \
                                              "rcnn", "crnn", "gpt", "bilstm"], \
                        default="bert", help="Encoder type.")
    parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent model.")

    # Subword options.
    parser.add_argument("--subword_type", choices=["none", "char"], default="none",
                        help="Subword feature type.")
    parser.add_argument("--sub_vocab_path", type=str, default="models/sub_vocab.txt",
                        help="Path of the subword vocabulary file.")
    parser.add_argument("--subencoder", choices=["avg", "lstm", "gru", "cnn"], default="avg",
                        help="Subencoder type.")
    parser.add_argument("--sub_layers_num", type=int, default=2, help="The number of subencoder layers.")

    # Optimizer options.
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warm up value.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    # Training options.
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout.")
    parser.add_argument("--epochs_num", type=int, default=20,
                        help="Number of epochs.")
    parser.add_argument("--report_steps", type=int, default=20,
                        help="Specific steps to print prompt.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed.")
    # help for inference
    parser.add_argument("--step", type=str, choices=["dev", "test"])
    parser.add_argument("--results_path", type=str, default='results/results_testing.txt')

    # hyper-parameters
    parser.add_argument("--rp_size", type=int, default=64)
    parser.add_argument("--threshold_1", type=float, default=0.5)
    parser.add_argument("--threshold_2", type=float, default=0.5)
    parser.add_argument("--threshold_3", type=float, default=0.5)
    parser.add_argument("--threshold_4", type=float, default=0.5)
    parser.add_argument("--threshold_trigger", type=float, default=0.0)
    parser.add_argument("--threshold_args", type=float, default=0.0)

    parser.add_argument("--args_weight", type=float, default=0.8)
    parser.add_argument("--trigger_pow", type=int, default=2)
    parser.add_argument("--args_pow", type=int, default=2)
    parser.add_argument("--decoder_dropout", type=float, default=0.3)

    parser.add_argument("--pretrained_downstream_path", type=str, default='models/v9.2.4_spt1_15000_fake_epoch1.bin')

    args = parser.parse_args()

    # Load the hyperparameters of the config file.
    args = load_hyperparam(args)
    set_seed(args.seed)
    return args
