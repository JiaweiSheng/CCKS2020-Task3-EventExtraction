import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    # BERT 配置
    parser.add_argument("--bert_vocab_path", default='pretrained_model/chinese_roberta_wwm_large_ext_pytorch/vocab.txt',
                        type=str)
    parser.add_argument("--bert_config_file",
                        default='pretrained_model/chinese_roberta_wwm_large_ext_pytorch/bert_config.json', type=str)
    parser.add_argument("--bert_model_dir", default='pretrained_model/chinese_roberta_wwm_large_ext_pytorch', type=str)
    # 中间监视文件
    parser.add_argument("--log_dir", default='monitor_dir/log', type=str)
    parser.add_argument("--writer_dir", default='monitor_dir/TSboard', type=str)
    parser.add_argument("--figure_dir", default='monitor_dir/figure', type=str)
    parser.add_argument("--checkpoint_dir", default='models_save', type=str)
    parser.add_argument("--cache_dir", default='cache', type=str)
    # 数据存储位置
    parser.add_argument("--results_output", default='results/test_base_cls.csv', type=str)
    parser.add_argument("--train_data_path", default='datasets/spt1/base_train.train.pkl', type=str)
    parser.add_argument("--valid_data_path", default='datasets/spt1/base_train.valid.pkl', type=str)
    parser.add_argument("--test_raw_data_path", default='datasets/pred_base/online_sample.csv', type=str)
    # 执行设置
    parser.add_argument("--arch", default='bert', type=str)
    parser.add_argument("--do_data", action='store_true')
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_valid", action='store_true')
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument("--save_best", action='store_true')
    parser.add_argument("--do_lower_case", action='store_true')
    # parser.add_argument('--data_name', default='ccks', type=str)
    parser.add_argument("--mode", default='min', type=str)
    parser.add_argument("--monitor", default='valid_loss', type=str)
    # 训练设置
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--resume_path", default='', type=str)
    parser.add_argument("--predict_checkpoints", type=int, default=0)
    parser.add_argument("--valid_size", default=0.2, type=float)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--sorted", default=0, type=int, help='1 : True  0:False ')
    parser.add_argument("--n_gpu", type=str, default='0,1', help='"0,1,.." or "0" or "" ')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument('--eval_batch_size', default=8, type=int)
    parser.add_argument("--train_max_seq_len", default=400, type=int)
    parser.add_argument("--eval_max_seq_len", default=400, type=int)
    parser.add_argument('--loss_scale', type=float, default=0)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--grad_clip", default=1.0, type=float)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fp16_opt_level', type=str, default='O1')
    # other
    parser.add_argument('--type', type=str, default='trans')
    parser.add_argument('--thresh', type=float, default=0.5)  # only for test

    args = parser.parse_args()
    # print(vars(args))
    for k, v in vars(args).items():
        print(k, ':', v)
    return args


if __name__ == '__main__':
    parse_args()
