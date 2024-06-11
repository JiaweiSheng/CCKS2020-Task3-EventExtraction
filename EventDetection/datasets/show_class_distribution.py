import json
import pandas as pd


def read_spt_ids(fn):
    with open(fn, 'r', encoding='utf-8') as f:
        spts = json.load(f)
    return spts['train'], spts['dev']


# def fun_base(fn_sample, fn_spts):
#     train_ids, dev_ids = read_spt_ids(fn_spts)
#     df = pd.read_csv(fn_sample)
#     cls_names = ['质押', '股份股权转让', '投资', '起诉', '减持']
#     cls_num = [0, 0, 0, 0, 0]
#     for item in df.itertuples():
#         if item[1] in train_ids:
#             for i in range(5):
#                 cls_num[i] += item[i + 3]
#     # print(cls_num)
#     print(fn_spts, 'train', [round(item / sum(cls_num), 4) for item in cls_num])
#
#     cls_num = [0, 0, 0, 0, 0]
#     for item in df.itertuples():
#         if item[1] in dev_ids:
#             for i in range(5):
#                 cls_num[i] += item[i + 3]
#     # print(cls_num)
#     print(fn_spts, 'dev', [round(item / sum(cls_num), 4) for item in cls_num])
#     print()


def fun_trans(fn_sample, fn_spts):
    train_ids, dev_ids = read_spt_ids(fn_spts)
    df = pd.read_csv(fn_sample)
    cls_names = ['收购', '判决', '签署合同', '担保', '中标']
    cls_num = [0, 0, 0, 0, 0]
    for item in df.itertuples():
        if item[1] in train_ids:
            for i in range(5):
                cls_num[i] += item[i + 3]
    # print(cls_num)
    print(fn_spts, 'train', [round(item / sum(cls_num), 4) for item in cls_num])

    cls_num = [0, 0, 0, 0, 0]
    for item in df.itertuples():
        if item[1] in dev_ids:
            for i in range(5):
                cls_num[i] += item[i + 3]
    # print(cls_num)
    print(fn_spts, 'dev', [round(item / sum(cls_num), 4) for item in cls_num])
    print()


if __name__ == '__main__':
    # fun_base('./pred_base/train_sample.csv', './spt1/data_ids.json')
    # fun_base('./pred_base/train_sample.csv', './spt2/data_ids.json')
    # fun_base('./pred_base/train_sample.csv', './spt3/data_ids.json')
    # fun_base('./pred_base/train_sample.csv', './spt4/data_ids.json')
    # fun_base('./pred_base/train_sample.csv', './spt5/data_ids.json')
    #
    # fun_trans('./pred_trans/train_sample.csv', './spt1/data_ids.json')
    # fun_trans('./pred_trans/train_sample.csv', './spt2/data_ids.json')
    # fun_trans('./pred_trans/train_sample.csv', './spt3/data_ids.json')
    # fun_trans('./pred_trans/train_sample.csv', './spt4/data_ids.json')
    # fun_trans('./pred_trans/train_sample.csv', './spt5/data_ids.json')
    #
    # fun_base('./pred_base/train_sample.csv', './spt6/data_ids.json')
    # fun_base('./pred_base/train_sample.csv', './spt7/data_ids.json')
    # fun_base('./pred_base/train_sample.csv', './spt8/data_ids.json')
    # fun_base('./pred_base/train_sample.csv', './spt9/data_ids.json')
    # fun_base('./pred_base/train_sample.csv', './spt10/data_ids.json')
    #
    # fun_trans('./pred_trans/train_sample.csv', './spt6/data_ids.json')
    # fun_trans('./pred_trans/train_sample.csv', './spt7/data_ids.json')
    # fun_trans('./pred_trans/train_sample.csv', './spt8/data_ids.json')
    # fun_trans('./pred_trans/train_sample.csv', './spt9/data_ids.json')
    # fun_trans('./pred_trans/train_sample.csv', './spt10/data_ids.json')

    fun_trans('./pred_trans/train_sample.csv', './spt1/data_ids.json')
    fun_trans('./pred_trans/train_sample.csv', './spt2/data_ids.json')
    fun_trans('./pred_trans/train_sample.csv', './spt3/data_ids.json')
    fun_trans('./pred_trans/train_sample.csv', './spt4/data_ids.json')
    fun_trans('./pred_trans/train_sample.csv', './spt5/data_ids.json')
