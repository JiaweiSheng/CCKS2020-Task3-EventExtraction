import json


def read_cls_result(fn):
    with open(fn, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    class_name = lines[0].strip().split(',')[1:]
    lines = lines[1:]
    data_ids = []
    data_labels = []
    for line in lines:
        line = line.strip()
        spts = line.split(',')
        data_id = spts[0]
        muti_label = [int(item) for item in spts[1:]]
        data_ids.append(data_id)
        data_labels.append(muti_label)
    return data_ids, data_labels, class_name


def read_data_text(fn):
    with open(fn, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    data_ids = []
    contents = []
    for line in lines:
        data = json.loads(line)
        data_id = data['id']
        content = data['content']
        data_ids.append(data_id)
        contents.append(content)
    return data_ids, contents


def main(thresh=5):
    data_ids_5, data_labels_5, class_name_5 = read_cls_result('datasets/event_classification_result/trans_5.cls')
    data_ids1, contents1 = read_data_text('datasets/data/trans_test.json')
    data_ids = data_ids1
    contents = contents1
    assert len(data_ids) == len(contents)
    id_2_content = {data_ids[i]: contents[i] for i in range(len(data_ids))}

    data_ids = []
    data_labels = []

    class_name_5_new = []
    for item in class_name_5:
        if item == '签署合同':
            class_name_5_new.append('合同')
        else:
            class_name_5_new.append(item)
    class_name_5 = class_name_5_new

    print(class_name_5)
    # 5
    for i in range(len(data_ids_5)):
        data_id = data_ids_5[i]
        multi_label = data_labels_5[i]
        for j in range(len(class_name_5)):
            if multi_label[j] >= thresh:
                data_ids.append(data_id)
                data_labels.append(class_name_5[j])

    with open('datasets/testing_data/test_format_new.json', 'w', encoding='utf-8') as f:
        for i, data_id in enumerate(data_ids):
            data = {'id': data_id,
                    'type': data_labels[i],
                    'content': id_2_content[data_id]}
            data_json = json.dumps(data, ensure_ascii=False)
            f.write(data_json + '\n')


if __name__ == '__main__':
    main(thresh=3)  # 根据ED模型数量决定投票阈值
