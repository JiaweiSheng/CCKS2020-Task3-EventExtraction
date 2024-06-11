import torch
import numpy as np
from v9.utils import *


def extract_specific_item_for_one(model, dt, token, seg, mask, rp, tm, args_num,
                                  threshold_1, threshold_2,
                                  threshold_3, threshold_4,
                                  threshold_trigger, threshold_args,
                                  ty_args_id, word_segs, word_poss, ners, deps):
    '''
    分别预测触发词和论元，两个任务
    '''
    if dt == '转让':
        dt = '股份股权转让'
    if dt == '合同':
        dt = '签署合同'
    # 预测触发词；
    # 注意，这里要预测句子中的所有触发词，可能多个
    p_s, p_e, text_emb, query_emb, hand_feature = model.predict_trigger_one(token, seg, mask, word_segs, word_poss, ners, deps)
    # 预测大于一定阈值，视为发生
    trigger_s = np.where(p_s > threshold_1)[0]
    trigger_e = np.where(p_e > threshold_2)[0]
    trigger_spans = []
    for i in trigger_s:
        es = trigger_e[trigger_e > i]
        if len(es) > 0:
            e = es[0]
            if p_s[i] * p_e[e] > threshold_trigger:
                # 这里试图按两者概率之积过滤，但其实没用-_-!
                # 使threshold_args默认为0
                trigger_spans.append((i, e))

    # 预测论元；
    # 注意，这里的论元只预测，句子中的一个触发词的全部论元
    p_s, p_e = model.predict_args_one(query_emb, text_emb, rp, tm, mask, hand_feature)
    # p_s 形状为：[论元类型数，句子长度]
    # p_e 形状为：[论元类型数，句子长度]
    p_s = np.transpose(p_s)
    p_e = np.transpose(p_e)
    args_spans = {i: [] for i in range(args_num)}
    for i in ty_args_id[dt]:
        # 依次遍历每种类型的论元预测序列
        args_s = np.where(p_s[i] > threshold_3)[0]
        args_e = np.where(p_e[i] > threshold_4)[0]
        for j in args_s:
            es = args_e[args_e > j]
            if len(es) > 0:
                e = es[0]
                if p_s[i][j] * p_e[i][e] > threshold_args:
                    # 这里试图按两者概率之积过滤，但其实没用-_-!
                    # 使threshold_args默认为0
                    args_spans[i].append((j, e))
    return trigger_spans, args_spans


def evaluate_one(model, args, dt, token, seg, mask, rp, tm, trigger_truth, args_truth, ty_args_id, word_segs, word_poss, ners, deps):
    '''
    在验证集上评估模型的性能，过程如：
    1. 分别预测两个任务的结果
    2. 分别计算两个任务的性能
    '''

    trigger_spans, args_spans = extract_specific_item_for_one(model, dt, token, seg, mask,
                                                              rp, tm,
                                                              args.args_num,
                                                              args.threshold_1, args.threshold_2, args.threshold_3,
                                                              args.threshold_4, args.threshold_trigger,
                                                              args.threshold_args, ty_args_id, word_segs, word_poss, ners, deps)
    trigger_truth, args_truth = trigger_truth[0], args_truth[0]
    trigger_spans = set(trigger_spans)
    trigger_truth = set(trigger_truth)
    num_pred = len(trigger_spans)
    num_truth = len(trigger_truth)
    num_correct = len(trigger_spans & trigger_truth)
    t_p = num_correct / (num_pred + 1e-10)
    t_r = num_correct / (num_truth + 1e-10)
    t_f = 2 * t_p * t_r / (t_p + t_r + 1e-10)
    num_pred = 0
    num_truth = 0
    num_correct = 0
    if dt == '转让':
        dt = '股份股权转让'
    if dt == '合同':
        dt = '签署合同'
    # 值得一提的是，这里按类型的schema来筛选论元类型
    # 比如<事件类型1>没有论元<时间>，那么即使预测的有时间，这里也丢弃它
    args_candidates = ty_args_id[dt]  # 事件schema
    for i in args_candidates:
        args_pred_one = set(args_spans[i])
        args_truth_one = set(args_truth[i])
        num_truth += len(args_truth_one)
        num_pred += len(args_pred_one)
        num_correct += len(args_pred_one & args_truth_one)

    a_p = num_correct / (num_pred + 1e-10)
    a_r = num_correct / (num_truth + 1e-10)
    a_f = 2 * a_p * a_r / (a_p + a_r + 1e-10)
    return t_p, t_r, t_f, a_p, a_r, a_f
