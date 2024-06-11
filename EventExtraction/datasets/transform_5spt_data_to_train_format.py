import json
from datasets import tool

TYPES = ['起诉', '投资', '减持', '股份股权转让', '质押', '收购', '判决', '签署合同', '担保', '中标']


def trans(records):
    data_len = len(records)
    data_new = []
    for i in range(data_len):
        record = json.loads(records[i])
        # print('#', record)
        data_id = record['id']
        schemas = record['events']
        content = record['content']

        for TYPE in TYPES:
            # 收集TYPE类型的事件schema，便于分割事件类型
            schemas_type = []
            for schema in schemas:
                event_type = schema['type']
                if event_type == TYPE:
                    schemas_type.append(schema)
            # 如果存在TYPE类型
            if len(schemas_type) != 0:
                triggers = []  # 保存事件触发词列表，去重；字符串span
                trigger_args = {}  # 保存事件触发词-论元字典，key去重
                for schema in schemas_type:
                    event_mention = schema['mentions']
                    for m in event_mention:
                        if m['role'] == 'trigger':
                            trigger = m['span']
                            trigger_args[str(trigger)] = {}
                            if trigger not in triggers:
                                triggers.append(trigger)
                # 将事件论元和事件触发词对应，其中相同触发词的事件论元会被合并
                for schema in schemas_type:
                    event_mention = schema['mentions']
                    for m in event_mention:
                        if m['role'] == 'trigger':
                            trigger = m['span']
                    args_dict = trigger_args[str(trigger)]
                    for m in event_mention:
                        if m['role'] == 'trigger':
                            continue
                        if m['role'] not in args_dict:
                            args_dict[m['role']] = []
                        args_dict[m['role']].append(m['span'])  # 相同触发词相同类型，顺序就不区分了
                # 按照trigger顺序，写出多个json数据
                triggers_str = [str(trigger) for trigger in triggers]  # 带顺序
                for trigger_str in trigger_args:
                    index = triggers_str.index(trigger_str)
                    data_dict = {}
                    data_dict['id'] = data_id
                    data_dict['type'] = TYPE
                    if TYPE == '股份股权转让':
                        data_dict['type'] = '转让'
                    if TYPE == '签署合同':
                        data_dict['type'] = '合同'
                    data_dict['content'] = content
                    data_dict['triggers'] = triggers
                    data_dict['index'] = index
                    data_dict['args'] = trigger_args[trigger_str]
                    # print(data_dict)
                    data_dict = json.dumps(data_dict, ensure_ascii=False)
                    data_new.append(data_dict)
    return data_new


def do(fn_src1, fn_src2, fn_tgt):
    data_ids1, contents1, events1 = tool.load_data(fn_src1)
    data_ids2, contents2, events2 = tool.load_data(fn_src2)
    data_ids = data_ids1 + data_ids2
    contents = contents1 + contents2
    events = events1 + events2

    train_ids, dev_ids = tool.load_data_ids(fn_tgt + '/data_ids.json')
    dev_records = tool.split_by_ids(data_ids, contents, events, dev_ids)
    dev_records = trans(dev_records)
    tool.save_lines(fn_tgt + '/dev_format.json', dev_records)

    train_records = tool.split_by_ids(data_ids, contents, events, train_ids)
    train_records = trans(train_records)
    tool.save_lines(fn_tgt + '/train_format.json', train_records)


if __name__ == '__main__':
    do('data/train_base.json', 'data/trans_train.json', 'spt1')
    do('data/train_base.json', 'data/trans_train.json', 'spt2')
    do('data/train_base.json', 'data/trans_train.json', 'spt3')
    do('data/train_base.json', 'data/trans_train.json', 'spt4')
    do('data/train_base.json', 'data/trans_train.json', 'spt5')

