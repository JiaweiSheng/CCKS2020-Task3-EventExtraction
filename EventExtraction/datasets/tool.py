import json


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


def load_data_ids(fn):
    d = json.load(open(fn, 'r'))
    train_ids = d['train']
    dev_ids = d['dev']
    return train_ids, dev_ids


def split_by_ids(data_ids, contents, events, split_ids):
    records = []
    for i, item in enumerate(data_ids):
        if item in split_ids:
            record = {}
            record['id'] = item
            record['content'] = contents[i]
            record['events'] = events[i]
            record = json.dumps(record, ensure_ascii=False)
            records.append(record)
    return records


def save_lines(fn, lines):
    with open(fn, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')
