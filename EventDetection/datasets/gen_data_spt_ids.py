import json
import random
import os

'''
对发布的两个训练集合并，然后按id划分成本地训练集和本地验证集
'''


# data
def load_data(file):
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    data_ids = []
    contents = []
    events = []
    for line in lines:
        record = json.loads(line)
        data_id = record['id']
        content = record['content']
        event = record['events']
        data_ids.append(data_id)
        contents.append(content)
        events.append(event)
    return data_ids, contents, events


def save_list(fn, data):
    with open(fn, 'w', encoding='utf-8') as f:
        for line in data:
            f.write(line + '\n')


# split ids
def load_data_ids(file):
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    data_ids = []
    for line in lines:
        record = json.loads(line)
        data_id = record['id']
        data_ids.append(data_id)
    return data_ids


def split_data(train_radio, seed, fn_begin):
    fn_1 = './data/trans_train.json'
    data_ids_1, contents_1, events_1 = load_data(fn_1)

    data_ids = data_ids_1
    contents = contents_1
    events = events_1

    random.seed(seed)
    random.shuffle(data_ids)
    n_data = len(data_ids)
    n_train = int(n_data * train_radio)
    n_dev = n_data - n_train
    print(n_data, n_train, n_dev)
    data_ids = data_ids[::-1]

    devs = []
    for begin_dev in range(0, n_data, n_dev):
        devs.append(data_ids[begin_dev:begin_dev + n_dev])

    trains = []
    for dev in devs:
        train = []
        for id in data_ids:
            if id not in dev:
                train.append(id)
        trains.append(train)

    for i in range(len(devs)):
        dev = devs[i]
        train = trains[i]
        # print(dev)
        print(len(dev), len(train))
        if not os.path.exists('spt' + str(fn_begin + i)):
            os.mkdir('spt' + str(fn_begin + i))
            d = {'train': train, 'dev': dev}
            json.dump(d, open('spt' + str(fn_begin + i) + '/data_ids.json', 'w'))


if __name__ == '__main__':
    split_data(0.9, 2040, 1)
    # split_data(0.8, 2040, 6)
    # split_data(0.9, 1123, 11)

