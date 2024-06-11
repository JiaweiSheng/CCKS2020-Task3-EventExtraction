import random


def read_data(fn):
    with open(fn, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    return lines


def save_data(fn, data):
    with open(fn, 'w', encoding='utf-8') as f:
        for line in data:
            f.write(line + '\n')


def do_b(fn_fake, fn_out=None):
    data2 = read_data(fn_fake)
    for i in range(10):
        data1 = read_data('spt' + str(i + 1) + '/train_format.json')
        save_data('spt' + str(i + 1) + '/train_format_b.json', data1 + data2)


def do_a(fn_truth, fn_fake, fn_out=None):
    data1 = read_data(fn_truth)
    data2 = read_data(fn_fake)

    random.seed(2040)
    random.shuffle(data2)

    n_data2 = len(data2)
    n_spt_data2 = int(n_data2 / 10)
    for i in range(10):
        data_spt = data2[i * n_spt_data2:(i + 1) * n_spt_data2]
        save_data('spt' + str(i + 1) + '/train_format_a.json', data1 + data_spt)


if __name__ == '__main__':
    # do_a('spt1/train_format.json', 'fake_label_data/fake_train_format_a.json')
    do_b('spt1/train_format.json', 'fake_label_data/fake_train_format_b.json')
