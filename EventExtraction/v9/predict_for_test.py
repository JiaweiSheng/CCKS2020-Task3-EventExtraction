import torch
import numpy as np
from v9.utils import *


def extract_all_items_for_one(model, device, idx, dt, dc, token, seg, mask, seq_len, args_num,
                              threshold_1, threshold_2, threshold_3,
                              threshold_4,
                              threshold_trigger, threshold_args,
                              results_dict: dict, id_args: dict, ty_args_id: dict,
                              word_segs, word_poss, ners, deps):
    assert idx in results_dict
    # under develop
    if dt == '转让':
        dt = '股份股权转让'
    if dt == '合同':
        dt = '签署合同'
    text_seq_len = seq_len - 4

    # 预测全部触发词
    p_s, p_e, text_emb, query_emb, hand_feature = model.predict_trigger_one(token, seg, mask, word_segs, word_poss, ners, deps)
    trigger_s = np.where(p_s > threshold_1)[0]
    trigger_e = np.where(p_e > threshold_2)[0]
    trigger_spans = []
    for i in trigger_s:
        es = trigger_e[trigger_e > i]
        if len(es) > 0:
            e = es[0]
            if p_s[i] * p_e[e] > threshold_trigger:
                trigger_spans.append((i, e))

    # 这个是要返回的预测结果；
    # idx为key，一个独立的事件schema为value
    # 注意，一个句子，多个事件类型，就会有多条数据（一条数据指[CLS]type[SEP]text[SEP])
    # 所以value的来源可能的一条数据的事件，也可能是多个数据的事件
    events = results_dict[idx]

    # 依次针对每个触发词，预测论元
    for k, span in enumerate(trigger_spans):
        # 构造论元识别模块的模型输入成分
        rp = get_relative_pos(span[0], span[1], text_seq_len)
        rp = [p + text_seq_len for p in rp]
        tm = get_trigger_mask(span[0], span[1], text_seq_len)
        rp = torch.LongTensor(rp).to(device)
        tm = torch.LongTensor(tm).to(device)
        rp = rp.unsqueeze(0)
        tm = tm.unsqueeze(0)
        # 预测论元
        p_s, p_e = model.predict_args_one(query_emb, text_emb, rp, tm, mask, hand_feature)
        p_s = np.transpose(p_s)
        p_e = np.transpose(p_e)
        # 准备数据的存储格式
        event = {'type': dt}
        mention = {'span': [int(span[0]), int(span[1]) + 1], 'role': 'trigger',
                   'word': dc[int(span[0]):int(span[1]) + 1]}
        mentions = [mention]  # 先把事件的触发词加上
        args_candidates = ty_args_id[dt]  # 事件类型对应论元类型的约束
        for i in args_candidates:
            # 依次遍历所有论元类型
            args_s = np.where(p_s[i] > threshold_3)[0]
            args_e = np.where(p_e[i] > threshold_4)[0]
            for j in args_s:
                es = args_e[args_e > j]
                if len(es) > 0:
                    e = es[0]
                    if p_s[i][j] * p_e[i][e] > threshold_args:
                        mention = {'span': [int(j), int(e) + 1], 'role': id_args[i], 'word': dc[int(j): int(e) + 1]}
                        mentions.append(mention)  # 再把事件的论元加上
        event['mentions'] = mentions
        events.append(event)
