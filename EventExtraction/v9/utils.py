from uer.utils.constants import *
from uer.utils.vocab import Vocab
from torch.utils.data import Dataset
import json
import numpy as np


def get_args2id():
    '''返回arg2id的字典'''
    args = ['collateral', 'sub-per', 'number', 'way', 'institution', 'money', 'date', 'obj-org', 'title',
            'target-company', 'sub', 'amount', 'sub-org', 'obj', 'share-org', 'obj-per', 'share-per',
            'proportion']

    args_s_id = {}
    args_e_id = {}
    for i in range(len(args)):
        s = args[i] + '_s'
        args_s_id[s] = i
        e = args[i] + '_e'
        args_e_id[e] = i
    return args_s_id, args_e_id


def get_dict():
    '''返回事件类型向args的映射关系'''
    args_list = ['collateral', 'sub-per', 'number', 'way', 'institution', 'money', 'date', 'obj-org', 'title',
                 'target-company', 'sub', 'amount', 'sub-org', 'obj', 'share-org', 'obj-per', 'share-per',
                 'proportion']

    ty_args = {
        '质押': ['date', 'proportion', 'number', 'collateral', 'obj-per', 'sub-per', 'sub-org', 'obj-org',
               'money'],
        '投资': ['date', 'sub', 'obj', 'money'],
        '股份股权转让': ['target-company', 'date', 'proportion', 'number', 'collateral', 'obj-per', 'sub-per',
                   'sub-org', 'obj-org', 'money'],
        '减持': ['sub', 'date', 'share-org', 'title', 'share-per', 'obj'],
        '起诉': ['date', 'obj-per', 'sub-per', 'sub-org', 'obj-org'],
        '收购': ['date', 'proportion', 'number', 'sub-per', 'sub-org', 'obj-org', 'money', 'way'],
        '判决': ['date', 'obj-per', 'sub-per', 'sub-org', 'obj-org', 'institution', 'money'],
        '中标': ['amount', 'date', 'sub', 'obj'],
        '签署合同': ['amount', 'date', 'obj-per', 'sub-per', 'sub-org', 'obj-org'],
        '担保': ['amount', 'date', 'sub-per', 'sub-org', 'obj-org', 'way']}

    id_args = {i: item for i, item in enumerate(args_list)}  # id向论元的对应关系
    args_id = {item: i for i, item in enumerate(args_list)}  # 论元向id的对应关系
    ty_args_id = {}  # type 向 args_id 的对应关系
    for ty in ty_args:
        args = ty_args[ty]
        tmp = [args_id[a] for a in args]
        ty_args_id[ty] = tmp
    return args_id, id_args, ty_args, ty_args_id


def _read_labeled_data(fn):
    '''读取带有标签的数据（训练集、验证集）'''
    with open(fn, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    data_ids = []
    data_content = []
    data_type = []
    data_triggers = []
    data_index = []
    data_args = []
    for line in lines:
        line_dict = json.loads(line.strip())
        data_ids.append(line_dict['id'])
        data_type.append(line_dict['type'])
        data_content.append(line_dict['content'])
        data_index.append(line_dict['index'])
        data_triggers.append(line_dict['triggers'])
        data_args.append(line_dict['args'])
    return data_ids, data_type, data_content, data_triggers, data_index, data_args


def _read_unlabeled_data(fn):
    '''读取测试集数据'''
    with open(fn, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    data_ids = []
    data_content = []
    data_type = []
    for line in lines:
        line_dict = json.loads(line.strip())
        data_ids.append(line_dict['id'])
        data_type.append(line_dict['type'])
        data_content.append(line_dict['content'])
    return data_ids, data_type, data_content


def get_relative_pos(start_idx, end_idx, length):
    '''
    相对位置编码，其中包含end_idx的元素
    return relative position, [start_idx, end_idx]
    '''
    pos = list(range(-start_idx, 0)) + [0] * (end_idx - start_idx + 1) + list(range(1, length - end_idx))
    return pos


def get_trigger_mask(start_idx, end_idx, length):
    '''
    用来生成trigger的mask，其中开始结束位置为1，这样就能够通过矩阵乘法实现trigger mean pooling
    [000010100000], [start_idx, end_idx]
    '''
    # mask = [1 if start_idx <= i <= end_idx else 0 for i in range(length)]
    mask = [0] * length
    mask[start_idx] = 1
    mask[end_idx] = 1
    return mask


class Data(Dataset):
    def __init__(self, task, fn, vocab, seq_len, args_vocab_s, args_vocab_e):
        assert task in ['train', 'dev', 'test']
        self.task = task
        self.vocab = vocab
        self.seq_len = seq_len
        self.args_vocab_s = args_vocab_s
        self.args_vocab_e = args_vocab_e
        self.args_num = len(args_vocab_s.keys())
        if self.task == 'test':
            # 读取数据并id化
            data_ids, data_type, data_content = _read_unlabeled_data(fn)
            self.data_ids = data_ids
            self.data_type = data_type
            self.data_content = data_content
            tokens_ids, segs_ids, masks_ids = self.data_to_id(data_type, data_content)
            self.token = tokens_ids  # 文字 id
            self.seg = segs_ids  # bert segment id，用来区分前半句和后半句
            self.mask = masks_ids  # bert mask id，用来区分padding
            # 读取人工特征
            self.word_segs = self.load_word_segs(data_ids)  # 分词序列
            self.word_poss = self.load_word_pos(data_ids)  # 词性序列
            self.ners = self.load_ners(data_ids)  # ner序列
            self.deps = self.load_deps(data_ids)  # 依存序列
        else:
            # 读取数据并id化
            data_ids, data_type, data_content, data_triggers, data_index, data_args = _read_labeled_data(fn)
            self.data_ids = data_ids
            self.data_type = data_type
            tokens_ids, segs_ids, masks_ids = self.data_to_id(data_type, data_content)
            self.token = tokens_ids
            self.seg = segs_ids
            self.mask = masks_ids
            # 感知trigger位置
            self.r_pos, self.t_m = self.get_rp_tm(data_triggers, data_index)
            self.t_index = data_index
            # 调研迁移人工特征
            self.word_segs = self.load_word_segs(data_ids)
            self.word_poss = self.load_word_pos(data_ids)
            self.ners = self.load_ners(data_ids)
            self.deps = self.load_deps(data_ids)

            if self.task == 'train':
                # 对训练集，读取真实标签（序列），用来计算损失
                t_s, t_e = self.trigger_seq_id(data_triggers)
                self.t_s = t_s
                self.t_e = t_e
                a_s, a_e, a_m = self.args_seq_id(data_args)
                self.a_s = a_s
                self.a_e = a_e
                self.a_m = a_m

            if self.task == 'dev':
                # 对验证集，读取真实标签（字典形式），在一定程度上用来评估事件抽取任务的性能（实体级评估）
                self.data_content = data_content
                self.data_args = data_args
                self.data_triggers = data_triggers
                triggers_truth_s, args_truth_s = self.results_for_eval()
                self.triggers_truth = triggers_truth_s
                self.args_truth = args_truth_s

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, index):
        if self.task == 'train':
            return self.data_ids[index], \
                   self.data_type[index], \
                   self.token[index], \
                   self.seg[index], \
                   self.mask[index], \
                   self.t_index[index], \
                   self.r_pos[index], \
                   self.t_m[index], \
                   self.t_s[index], \
                   self.t_e[index], \
                   self.a_s[index], \
                   self.a_e[index], \
                   self.a_m[index], \
                   self.word_segs[index], \
                   self.word_poss[index], \
                   self.ners[index], \
                   self.deps[index]
        elif self.task == 'dev':
            return self.data_ids[index], \
                   self.data_type[index], \
                   self.token[index], \
                   self.seg[index], \
                   self.mask[index], \
                   self.t_index[index], \
                   self.r_pos[index], \
                   self.t_m[index], \
                   self.triggers_truth[index], \
                   self.args_truth[index], \
                   self.word_segs[index], \
                   self.word_poss[index], \
                   self.ners[index], \
                   self.deps[index]
        elif self.task == 'test':
            return self.data_ids[index], \
                   self.data_type[index], \
                   self.data_content[index], \
                   self.token[index], \
                   self.seg[index], \
                   self.mask[index], \
                   self.word_segs[index], \
                   self.word_poss[index], \
                   self.ners[index], \
                   self.deps[index]
        else:
            raise Exception('task not define !')

    def data_to_id(self, data_types, data_contents):
        '''
        将文字转换为id
        值得一提的是，形如：
        tokens_ids: [CLS] type [SEP] text [SEP]
        segs_ids:   111111111111111122222222222
        masks_ids   000000000000000011111111110
        '''
        tokens_ids = []
        segs_ids = []
        masks_ids = []
        for i in range(len(self.data_ids)):
            data_type = data_types[i]
            data_content = data_contents[i]
            query_tokens = [CLS_ID] + [self.vocab.get(t) for t in data_type] + [SEP_ID]
            content_tokens = [self.vocab.get(t) for t in data_content] + [SEP_ID]
            tokens = query_tokens + content_tokens
            segs = [1] * len(query_tokens) + [2] * len(content_tokens)
            masks = [0] * len(query_tokens) + [1] * (len(content_tokens) - 1) + [0]
            if len(tokens) > self.seq_len:
                tokens = tokens[:self.seq_len]
                segs = segs[:self.seq_len]
                masks = masks[:self.seq_len]
            while len(tokens) < self.seq_len:
                tokens.append(0)
                segs.append(0)
                masks.append(0)
            tokens_ids.append(tokens)
            segs_ids.append(segs)
            masks_ids.append(masks)
        return tokens_ids, segs_ids, masks_ids

    def trigger_seq_id(self, data_triggers):
        '''
        此乃ground truth trigger标签，是训练的目标
        生成的trigger矩阵，形如：
        t_s，trigger 开始为1，不然为0
        t_e，trigger 结束为1，不然为0

        此外，因为tokens_ids: [CLS] type [SEP] text [SEP]，
        固定了 type 两个字，[CLS] type [SEP]为4个字
        那么 text [SEP]部分的为 seq_len - 4 个字
        '''
        trigger_s = []
        trigger_e = []
        for i in range(len(self.data_ids)):
            data_trigger = data_triggers[i]
            t_s = [0] * (self.seq_len - 4)
            t_e = [0] * (self.seq_len - 4)

            for t in data_trigger:
                t_s[t[0]] = 1
                t_e[t[1] - 1] = 1

            trigger_s.append(t_s)
            trigger_e.append(t_e)
        return trigger_s, trigger_e

    def args_seq_id(self, data_args_list):
        '''
        ground truth args标签

        '''
        args_s_lines = []
        args_e_lines = []
        arg_masks = []
        for i in range(len(self.data_ids)):
            args_s = np.zeros(shape=[self.args_num, self.seq_len - 4])
            args_e = np.zeros(shape=[self.args_num, self.seq_len - 4])
            data_args_dict = data_args_list[i]
            arg_mask = [0] * self.args_num
            for args_name in data_args_dict:
                s_r_i = self.args_vocab_s[args_name + '_s']
                e_r_i = self.args_vocab_e[args_name + '_e']
                arg_mask[s_r_i] = 1
                for span in data_args_dict[args_name]:
                    args_s[s_r_i][span[0]] = 1
                    args_e[e_r_i][span[1] - 1] = 1
            args_s_lines.append(args_s)
            args_e_lines.append(args_e)
            arg_masks.append(arg_mask)
        return args_s_lines, args_e_lines, arg_masks

    def results_for_eval(self):
        '''
        用来读取ground_truth，便于验证模型性能
        只参与dev，在train、test中不参与
        :return:
        '''
        triggers_truth_s = []
        args_truth_s = []
        for i in range(len(self.data_ids)):
            triggers = self.data_triggers[i]
            args = self.data_args[i]
            triggers_truth = [(span[0], span[1] - 1) for span in triggers]
            args_truth = {i: [] for i in range(self.args_num)}
            for args_name in args:
                s_r_i = self.args_vocab_s[args_name + '_s']
                for span in args[args_name]:
                    args_truth[s_r_i].append((span[0], span[1] - 1))
            triggers_truth_s.append(triggers_truth)
            args_truth_s.append(args_truth)
        return triggers_truth_s, args_truth_s

    def get_rp_tm(self, triggers, data_index):
        '''
        用来根据触发词的，生成触发词的相对位置编码和触发词mask
        r_pos: 相对位置编码
        t_m: 触发词mask，用来做pooling
        '''
        r_pos = []
        t_m = []
        for i in range(len(self.data_ids)):
            trigger = triggers[i]
            index = data_index[i]
            span = trigger[index]
            pos = get_relative_pos(span[0], span[1] - 1, self.seq_len - 4)
            pos = [p + self.seq_len - 4 for p in pos]
            mask = get_trigger_mask(span[0], span[1] - 1, self.seq_len - 4)
            r_pos.append(pos)
            t_m.append(mask)
        return r_pos, t_m

    def pad_seqs(self, seqs, max_len):
        '''
        对序列进行padding，用于分词、词性、ner、依存特征序列
        '''
        seqs_new = []
        for seq in seqs:
            seq_new = [0] * max_len
            for i in range(min(len(seq), max_len)):
                seq_new[i] = seq[i]
            seqs_new.append(seq_new)
        return seqs_new

    def load_word_segs(self, data_ids):
        '''
        读取分词特征序列
        :param data_ids:
        :return:
        '''
        with open('./datasets/hand_feature/ltp/data_segs_seq.json', 'r', encoding='utf-8') as f:
            lines = f.readlines()
        segs2id = {'<PAD>': 0, 'B': 1, 'I': 2, 'E': 3, 'S': 4}

        id_segs = {}
        for line in lines:
            data = json.loads(line)
            id_segs[data['id']] = [segs2id[item] for item in data['parse']]

        segs = []
        for data_id in data_ids:
            assert data_id in id_segs
            segs.append(id_segs[data_id])
        segs = self.pad_seqs(segs, self.seq_len - 4)
        return segs

    def load_word_pos(self, data_ids):
        '''
        独取词性特征序列
        :param data_ids:
        :return:
        '''
        with open('./datasets/hand_feature/ltp/data_poss_seq.json', 'r', encoding='utf-8') as f:
            lines = f.readlines()

        pos2id = {'<PAD>': 0, 'b': 1, 'n': 2, 'd': 3, 'p': 4, 'nt': 5, 'v': 6, 'u': 7, 'wp': 8, 'nd': 9, 'm': 10,
                  'a': 11, 'c': 12, 'nz': 13, 'q': 14, 'ws': 15, 'ns': 16, 'r': 17, 'j': 18, 'i': 19, 'nh': 20,
                  'nl': 21, 'z': 22, 'ni': 23, 'h': 24, 'k': 25, '%': 26, 'o': 27, 'e': 28}

        id_segs = {}
        for line in lines:
            data = json.loads(line)
            id_segs[data['id']] = [pos2id[item] for item in data['parse']]

        poss = []
        for data_id in data_ids:
            assert data_id in id_segs
            poss.append(id_segs[data_id])
        poss = self.pad_seqs(poss, self.seq_len - 4)
        return poss

    def load_ners(self, data_ids):
        '''
        读取ner特征序列
        :param data_ids:
        :return:
        '''
        with open('./datasets/hand_feature/ltp/data_ners.json', 'r', encoding='utf-8') as f:
            lines = f.readlines()

        pos2id = {'<PAD>': 0, 'O': 1, 'B-Ni': 2, 'I-Ni': 3, 'E-Ni': 4, 'S-Ns': 5, 'S-Nh': 6, 'B-Ns': 7, 'E-Ns': 8,
                  'S-Ni': 9, 'I-Ns': 10, 'B-Nh': 11, 'E-Nh': 12, 'I-Nh': 13}

        id_segs = {}
        for line in lines:
            data = json.loads(line)
            id_segs[data['id']] = [pos2id[item] for item in data['parse']]

        poss = []
        for data_id in data_ids:
            assert data_id in id_segs
            poss.append(id_segs[data_id])
        poss = self.pad_seqs(poss, self.seq_len - 4)
        return poss

    def load_deps(self, data_ids):
        '''
        读取依存特征序列
        :param data_ids:
        :return:
        '''
        with open('./datasets/hand_feature/ltp/data_deps_seq.json', 'r', encoding='utf-8') as f:
            lines = f.readlines()

        pos2id = {'<PAD>': 0, 'ATT': 1, 'SBV': 2, 'ADV': 3, 'POB': 4, 'HED': 5, 'RAD': 6, 'VOB': 7, 'WP': 8, 'COO': 9,
                  'FOB': 10, 'LAD': 11, 'CMP': 12, 'DBL': 13, 'IOB': 14}

        id_segs = {}
        for line in lines:
            data = json.loads(line)
            id_segs[data['id']] = [pos2id[item] for item in data['parse']]

        poss = []
        for data_id in data_ids:
            assert data_id in id_segs
            poss.append(id_segs[data_id])
        poss = self.pad_seqs(poss, self.seq_len - 4)
        return poss


def collate_fn_train(data):
    '''
    :param data: [(x, y), (x, y), (), (), ()]
    :return:
    idx, 数据的id标识
    dt, 数据的事件类型，文字形式
    token, 字序列
    seg, bert的seg序列，用来区分前半句和后半句
    mask, bert的mask序列，用来识别padding等，是则为0
    t_index, 废弃；用来返回要识别的论元，属于左数第index个触发词；然而这个并没有用到，已经通过r_pos和t_m搞定
    r_pos, 相对位置编码
    t_m, trigger_mask，是则为1，用来做trigger pooling
    t_s, ground_truth 触发词开始
    t_e, ground_truth 触发词结束
    a_s, ground_truth 论元开始
    a_e, ground_truth 论元结束
    a_m, 废弃；用来标志这个句子存在哪些论元，存在为1
    word_segs, 分词id序列
    word_poss, 词性id序列
    ners, ner id序列
    deps, 依存id序列
    '''
    idx, dt, token, seg, mask, t_index, r_pos, t_m, t_s, t_e, a_s, a_e, a_m, word_segs, word_poss, ners, deps = zip(
        *data)
    return idx, dt, token, seg, mask, t_index, r_pos, t_m, t_s, t_e, a_s, a_e, a_m, word_segs, word_poss, ners, deps


def collate_fn_dev(data):
    '''
    :param data: [(x, y), (x, y), (), (), ()]
    :return:
    '''
    idx, dt, token, seg, mask, t_index, r_pos, t_m, t_t, a_t, word_segs, word_poss, ners, deps = zip(*data)
    return idx, dt, token, seg, mask, t_index, r_pos, t_m, t_t, a_t, word_segs, word_poss, ners, deps


def collate_fn_test(data):
    '''
    :param data: [(x, y), (x, y), (), (), ()]
    :return:
    '''
    idx, dt, dc, token, seg, mask, word_segs, word_poss, ners, deps = zip(*data)
    return idx, dt, dc, token, seg, mask, word_segs, word_poss, ners, deps
