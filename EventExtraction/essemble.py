import json

'''
模型结果集成和后处理
集成方法为: 对触发词和论元分别投票
后处理方法为: 增加论元长度约束；清除重复的mention；清楚mention为空的事件
'''


def read_data(fn):
    with open(fn, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    id2events = {}
    for line in lines:
        data_dict = json.loads(line)
        data_id = data_dict['id']
        events = data_dict['events']
        id2events[data_id] = events
    return id2events


def unique_each_mention_in_event(results):
    # discard
    # 清除重复的mentions
    results_new = []
    for result in results:
        result_new = {'type': result['type']}
        mentions = result['mentions']
        mentions_new = []
        for m in mentions:
            if m not in mentions_new:
                mentions_new.append(m)
            else:
                print(m)
                pass
        result_new['mentions'] = mentions_new
        # print(result_new)
        results_new.append(result_new)
    return results_new


def is_smaller_than_maxlen(role, spanlen):
    role_len = {'collateral': 12, 'proportion': 37, 'obj-org': 32, 'trigger': 6, 'number': 18, 'date': 27,
                'sub-org': 35, 'sub': 38, 'obj': 36, 'money': 21, 'target-company': 59, 'share-org': 19, 'title': 6,
                'sub-per': 15, 'obj-per': 15, 'share-per': 20, 'institution': 22, 'way': 6, 'amount': 17}
    assert role in role_len.keys()
    if spanlen > role_len[role]:
        return False
    else:
        return True


def filter_args_by_vote(role_span_count, thresh=0):
    # 若是'sub-per_#_[26, 30]_#_冉盛盛瑞': 5, 'sub-per_#_[27, 30]_#_盛盛瑞': 2，可以用投票法解决
    # 同一个词，不同位置，投票试试？
    # ro_sp = m['role'] + '_#_' + str(m['span']) + '_#_' + m['word']
    role_list = []
    span_list = []
    word_list = []
    count_list = []
    flag_list = []
    ro_sp_list = []
    for ro_sp in role_span_count:
        role, span, word = ro_sp.split('_#_')
        b, e = span[1:-1].split(',')
        span = [int(b), int(e)]
        role_list.append(role)
        span_list.append(span)
        word_list.append(word)
        count_list.append(role_span_count[ro_sp])
        ro_sp_list.append(ro_sp)
        flag_list.append(True)
    for i in range(len(role_list)):
        for j in range(i + 1, len(role_list)):
            if role_list[i] == role_list[j] and (
                    span_list[i][0] == span_list[j][0] or span_list[i][1] == span_list[j][1]):
                # and span_list[i] != span_list[j]:
                if count_list[i] > count_list[j]:
                    flag_list[j] = False
                else:
                    flag_list[i] = False
    role_span_count_new = {}
    for i in range(len(flag_list)):
        if flag_list[i]:
            role_span_count_new[ro_sp_list[i]] = role_span_count[ro_sp_list[i]]
    return role_span_count_new


def vote_for_results(results, trigger_thresh=1, args_thresh=1):
    # 寻找大于阈值的触发词
    type_trigger = {}
    for schama in results:
        ty = schama['type']
        for m in schama['mentions']:
            if m['role'] == 'trigger':
                ty_tri = ty + '_#_' + str(m['span']) + '_#_' + m['word']
                if ty_tri not in type_trigger:
                    type_trigger[ty_tri] = 0
                type_trigger[ty_tri] += 1
    # print(type_trigger)
    type_trigger_filtered = [key for key in type_trigger if type_trigger[key] >= trigger_thresh]
    # 保留选定触发词的schema
    type_trigger_filtered_schema = {key: [] for key in type_trigger_filtered}
    for schama in results:
        ty = schama['type']
        for m in schama['mentions']:
            if m['role'] == 'trigger':
                ty_tri = ty + '_#_' + str(m['span']) + '_#_' + m['word']
                if ty_tri in type_trigger_filtered_schema.keys():
                    type_trigger_filtered_schema[ty_tri].append(schama)
    # for key in type_trigger_filtered_schema:
    #     print(key, type_trigger_filtered_schema[key])
    # 保留大于阈值的论元
    events_new = []
    for key in type_trigger_filtered_schema:
        schamas = type_trigger_filtered_schema[key]
        role_span_count = {}
        for schama in schamas:
            for m in schama['mentions']:
                if m['role'] != 'trigger':
                    ro_sp = m['role'] + '_#_' + str(m['span']) + '_#_' + m['word']
                    if ro_sp not in role_span_count:
                        role_span_count[ro_sp] = 0
                    role_span_count[ro_sp] += 1
        # 保留大于阈值的论元
        # print(role_span_count)
        role_span_count = filter_args_by_vote(role_span_count)

        ms = []
        ty, span, word = key.split('_#_')
        b, e = span[1:-1].split(',')
        span = [int(b), int(e)]
        ms.append({'role': 'trigger', 'span': span, 'word': word})
        for ro_sp in role_span_count:
            if role_span_count[ro_sp] >= args_thresh:
                role, span, word = ro_sp.split('_#_')
                b, e = span[1:-1].split(',')
                span = [int(b), int(e)]
                # 长度约束
                if is_smaller_than_maxlen(role, spanlen=span[1] - span[0]):
                    mention = {'role': role, 'span': span, 'word': word}
                    ms.append(mention)
                else:
                    mention = {'role': role, 'span': span, 'word': word}
                    # print('role larger than max length constrain:', mention)
        schema_new = {'type': ty, 'mentions': ms}
        events_new.append(schema_new)
    return events_new


def unique_constrain(events):
    # 去除空的mentions的事件
    events_new = []
    for event in events:
        if event in events_new:
            pass
        else:
            if len(event['mentions']) == 0:
                pass
            else:
                events_new.append(event)
    return events_new


def main():
    result_1 = read_data('results/model_spt1_result_15000')
    result_2 = read_data('results/model_spt2_result_15000')
    result_3 = read_data('results/model_spt3_result_15000')
    result_4 = read_data('results/model_spt4_result_15000')
    result_5 = read_data('results/model_spt5_result_15000')
    result_1_b = read_data('results/model_spt1_result_fakeb_hard')
    result_1_a = read_data('results/model_spt1_result_fakea')

    result_3_b = read_data('results/model_spt3_result_fakeb_hard')
    result_4_b = read_data('results/model_spt4_result_fakeb_hard')
    result_5_b = read_data('results/model_spt5_result_fakeb_hard')
    result_6_b = read_data('results/model_spt6_result_fakeb_hard')
    result_7_b = read_data('results/model_spt7_result_fakeb_hard')
    result_8_b = read_data('results/model_spt8_result_fakeb_hard')
    result_9_b = read_data('results/model_spt9_result_fakeb_hard')
    result_10_b = read_data('results/model_spt10_result_fakeb_hard')

    id_list = list(result_1.keys())
    results_new = []
    for idx in id_list:
        results = result_1[idx] + result_2[idx] + result_3[idx] + result_4[idx] + result_5[idx] + \
                  result_1_a[idx] + \
                  result_1_b[idx] + \
                  result_3_b[idx] + result_4_b[idx] + result_5_b[idx] + result_6_b[idx] + result_7_b[idx] + \
                  result_8_b[idx] + result_9_b[idx] + result_10_b[idx]
        # results = unique_each_mention_in_event(results)
        events_new = vote_for_results(results, trigger_thresh=11, args_thresh=1)  # best: 11, 1
        events_new = unique_constrain(events_new)
        result_new = {'id': idx, 'events': events_new}
        results_new.append(result_new)

    with open('final_results/results_testing_ensemble.txt', 'w', encoding='utf-8')as f:
        for result in results_new:
            line = json.dumps(result, ensure_ascii=False)
            f.write(line + '\n')


if __name__ == '__main__':
    main()
