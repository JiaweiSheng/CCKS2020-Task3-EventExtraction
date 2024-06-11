import random
import pandas as pd
from tqdm import tqdm
import pickle
from pybert.preprocessing.preprocessor import ChinesePreProcessor
import json


def save_pickle(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


class TaskData(object):
    def __init__(self):
        pass

    def train_val_split(self, ids, X, y, save=True,
                        save_data_path=None,
                        train_ids=None, valid_ids=None):
        train_data = []
        valid_data = []
        for i, idx in enumerate(ids):
            if idx in train_ids:
                train_data.append((X[i], y[i]))
            elif idx in valid_ids:
                valid_data.append((X[i], y[i]))
        print(len(train_data), len(valid_data))
        if save:
            train_path = save_data_path + ".train.pkl"
            valid_path = save_data_path + ".valid.pkl"
            save_pickle(data=train_data, file_path=train_path)
            save_pickle(data=valid_data, file_path=valid_path)
        return train_data, valid_data

    def read_data(self, raw_data_path, target_lst, preprocessor=None, is_train=True):
        '''
        :param raw_data_path:
        :param preprocessor:
        :param is_train:
        :param target_lst:
        :return:
        '''
        ids, targets, sentences = [], [], []
        data = pd.read_csv(raw_data_path)
        for row in data.values:
            if is_train:
                target = row[2:]
            else:
                target = target_lst
            sentence = str(row[1])
            number = str(row[0])
            if preprocessor:
                sentence = preprocessor(sentence)
            if sentence:
                ids.append(number)
                targets.append(target)
                sentences.append(sentence)
        return ids, targets, sentences

    def load_ids_split(self, data_path):
        with open(data_path, 'r') as f:
            spts = json.load(f)
        train_ids = set(spts['train'])
        valid_ids = set(spts['dev'])
        return train_ids, valid_ids


def main_train(raw_data_path, target_lst, spts_data_ids, save_data_path):
    taskdata = TaskData()
    ids, targets, sentences = taskdata.read_data(raw_data_path=raw_data_path,
                                                 preprocessor=ChinesePreProcessor(),
                                                 is_train=True, target_lst=target_lst)
    train_ids, valid_ids = taskdata.load_ids_split(data_path=spts_data_ids)
    taskdata.train_val_split(ids, sentences, targets, save=True,
                             save_data_path=save_data_path,
                             train_ids=train_ids, valid_ids=valid_ids)


if __name__ == '__main__':

    # org_lst = ['收购', '判决', '签署合同', '担保', '中标']
    raw_data_path = 'datasets/pred_trans/train_sample.csv'
    target_lst = [-1, -1, -1, -1, -1]
    for i in range(10):
        spts_data_ids = 'datasets/spt' + str(i + 1) + '/data_ids.json'
        save_data_path = 'datasets/spt' + str(i + 1) + '/trans_train'
        main_train(raw_data_path, target_lst, spts_data_ids, save_data_path)
