import json
import pandas as pd


def check_num_type(f_in):
    lst = []
    in_file = open(f_in, 'r', encoding='utf-8')
    for line in in_file:
        line = line.strip()
        line = json.loads(line)
        # print(line)
        ids = line['id']
        content = line['content']
        for k in line['events']:
            evn_type = k['type']
            lst.append(evn_type)
    lst = set(lst)
    print(lst)


def change_data(f_in, org_lst):
    in_file = open(f_in, 'r', encoding='utf-8')
    final_lst = []
    for line in in_file:
        line = line.strip()
        line = json.loads(line)
        ids = line['id']
        content = line['content']
        lst = []
        for k in line['events']:
            evn_type = k['type']
            lst.append(evn_type)
        # print(ids, content, lst)
        label_lst = []
        label_lst.append(ids)
        label_lst.append(content)
        for i in org_lst:
            # print(i, lst)
            if i in lst:
                label_lst.append(1)
            else:
                label_lst.append(0)
        # print(label_lst)
        final_lst.append(label_lst)
    return final_lst


def get_cls_train_data(f_in, f_out, org_lst):
    final_lst = change_data(f_in, org_lst)
    df = pd.DataFrame()
    df = df.append(final_lst, ignore_index=True)
    col_lst = ['id', 'content'] + org_lst
    df.columns = col_lst
    df.to_csv(f_out, index=0, encoding='utf-8')
    print('分类模型训练集已转换完成！')


def get_cls_test_data(f_in, f_out):
    test_df = open(f_in, 'r', encoding='utf-8')
    lst = []
    for line in test_df.readlines():
        line = line.strip()
        line = json.loads(line)
        lst.append(line)
        # print(line)

    df = pd.DataFrame(lst)
    df = df[['id', 'content']]
    df.to_csv(f_out, index=0, encoding='utf-8', )
    print('分类模型测试集已转换完成！')


if __name__ == '__main__':
    # # 处理大样本类别
    # org_lst = ['质押', '股份股权转让', '投资', '起诉', '减持']
    # f_train_in = './datasets/data/train_base.json'
    # f_train_out = './datasets/pred_base/train_sample.csv'
    # get_cls_train_data(f_train_in, f_train_out, org_lst)
    #
    # f_test_in = './datasets/data/dev_base.json'
    # f_test_out = './datasets/pred_base/online_sample.csv'
    # get_cls_test_data(f_test_in, f_test_out)
    #
    # # 处理小样本类别
    # org_lst = ['收购', '判决']
    # f_train_in = './datasets/data/trans_train.json'
    # f_train_out = './datasets/pred_trans/train_sample.csv'
    # get_cls_train_data(f_train_in, f_train_out, org_lst)
    #
    # f_test_in = './datasets/data/trans_dev.json'
    # f_test_out = './datasets/pred_trans/online_sample.csv'
    # get_cls_test_data(f_test_in, f_test_out)

    # # 处理小样本类别
    org_lst = ['收购', '判决', '签署合同', '担保', '中标']
    f_train_in = './datasets/data/trans_train.json'
    f_train_out = './datasets/pred_trans/train_sample.csv'
    get_cls_train_data(f_train_in, f_train_out, org_lst)

    f_test_in = './datasets/data/trans_test.json'
    f_test_out = './datasets/pred_trans/online_sample.csv'
    get_cls_test_data(f_test_in, f_test_out)
