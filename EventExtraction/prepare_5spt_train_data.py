from datasets import transform_5spt_data_to_train_format

if __name__ == '__main__':
    for i in range(5):
        transform_5spt_data_to_train_format.do('datasets/data/train_base.json', 'datasets/data/trans_train.json',
                                               'datasets/spt' + str(i + 6))
