import json

'''
补充没有预测结果的数据，用来形成提交格式
'''


# data
def load_data(file):
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    data_ids = []
    contents = []
    # events = []
    for line in lines:
        record = json.loads(line)
        data_id = record['id']
        content = record['content']
        # event = record['events']
        data_ids.append(data_id)
        contents.append(content)
        # events.append(event)
    return data_ids, contents


def load_result(file):
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    result_json = []
    ids = []
    for i, line in enumerate(lines):
        # print(i, line)
        j = json.loads(line)
        ids.append(j['id'])
        result_json.append(j)
    return result_json, ids


def save_new(file, lines):
    with open(file, 'w', encoding='utf-8') as f:
        for line in lines:
            line = json.dumps(line, ensure_ascii=False)
            f.write(line + '\n')


def fill_empty_for_submit(result_json: list, ids_with_pred):
    '''
    :param result_json: list of dict
    :param ids_with_pred:
    :return:
    '''
    fn_1 = './data/dev_base.json'
    fn_2 = './data/trans_dev.json'
    data_ids_1, contents_1 = load_data(fn_1)
    data_ids_2, contents_2 = load_data(fn_2)
    data_ids = data_ids_1 + data_ids_2
    ids = set(ids_with_pred)
    count = 0
    for i, id in enumerate(data_ids):
        if id not in ids:
            result = {'id': id, 'events': []}
            result_json.append(result)
            count += 1
        if i % 100 == 0:
            print(i)
    print('No predicted type num is:', count)
    return result_json


def main():
    fn_2 = 'datasets/data/trans_test.json'
    data_ids_2, contents_2 = load_data(fn_2)
    data_ids = data_ids_2
    result_json, ids = load_result('final_results/results_testing_ensemble.txt')
    print(len(ids))
    ids = set(ids)
    print(len(ids))
    print(len(data_ids))
    count = 0
    for i, id in enumerate(data_ids):
        if id not in ids:
            result = {'id': id, 'events': []}
            result_json.append(result)
            count += 1
        if i % 1000 == 0:
            print(i)
    print(count)
    save_new('final_results/result.json', result_json)


if __name__ == '__main__':
    main()
