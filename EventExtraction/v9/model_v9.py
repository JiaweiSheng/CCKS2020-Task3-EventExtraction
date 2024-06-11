import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from uer.model_builder import build_model
from uer.utils.config import load_hyperparam
from uer.utils.optimizers import *
from uer.utils.constants import *
from uer.utils.vocab import Vocab
from uer.utils.seed import set_seed
from uer.model_saver import save_model
from uer.model_loader import load_model
from uer.utils.constants import *
from torch import nn
from v9.layers import *


class TriggerRec(nn.Module):
    def __init__(self, args, hidden_size, all_hand_feature_size=0):
        super(TriggerRec, self).__init__()
        self.CLN = ConditionalLayerNorm(hidden_size)
        # self.CSA = ConditionalSelfAttention(hidden_size, hidden_size, hidden_size)
        # self.SA = SelfAttention(hidden_size, hidden_size, hidden_size, args.decoder_dropout)
        self.SA = MultiHeadedAttention(hidden_size, heads_num=1, dropout=args.decoder_dropout)

        # change
        self.hidden = nn.Linear(hidden_size * 2 + all_hand_feature_size, hidden_size)
        self.head_cls = nn.Linear(hidden_size, 1)
        self.tail_cls = nn.Linear(hidden_size, 1)

        self.dropout = nn.Dropout(args.decoder_dropout)
        self.GlobalAttention = AttentionAddition(hidden_size)
        self.sent_cls = nn.Linear(hidden_size, 1)

    def forward(self, query_emb, text_emb, mask, hand_feature):
        '''

        :param query_emb: [b, 4, e]
        :param text_emb: [b, t-4, e]
        :param mask: 0 if masked
        :return:
        '''
        query_emb = torch.mean(query_emb, dim=1)
        h_cln = self.CLN(text_emb, query_emb)
        # h_csa = self.CSA(text_emb, mask, query_emb)
        h_sa = self.SA(h_cln, h_cln, h_cln, mask)
        inp = torch.cat([h_cln, h_sa, hand_feature], dim=-1)

        inp = self.dropout(inp)
        inp = torch.relu(self.hidden(inp))
        inp = self.dropout(inp)
        p_s = torch.sigmoid(self.head_cls(inp))  # [b, t, 1]
        p_e = torch.sigmoid(self.tail_cls(inp))  # [b, t, 1]

        sent_ctx = self.GlobalAttention(query_emb, text_emb, mask)
        p_g = torch.sigmoid(self.sent_cls(sent_ctx))  # [b, 1]

        p_g = p_g.unsqueeze(2).expand_as(p_s)  # [b, 1, 1]-> [b, t, e]
        p_s = p_s * p_g
        p_e = p_e * p_g
        return p_s, p_e, h_cln


class ArgsRec(nn.Module):
    def __init__(self, args, hidden_size, num_labels, seq_len, pos_emb_size, all_hand_feature_size):
        super(ArgsRec, self).__init__()
        self.relative_pos_embed = nn.Embedding(seq_len * 2, pos_emb_size)
        self.CLN = ConditionalLayerNorm(hidden_size)
        # self.CSA = ConditionalSelfAttention(hidden_size, hidden_size, hidden_size)
        # self.SA = SelfAttention(hidden_size, hidden_size, hidden_size, args.decoder_dropout)
        self.SA = MultiHeadedAttention(hidden_size, heads_num=1, dropout=args.decoder_dropout)
        self.GlobalAttention = AttentionAddition(hidden_size)

        # change
        self.hidden = nn.Linear(hidden_size * 2 + pos_emb_size + all_hand_feature_size, hidden_size)
        self.head_cls = nn.Linear(hidden_size, num_labels)
        self.tail_cls = nn.Linear(hidden_size, num_labels)

        self.seq_len = seq_len
        self.dropout = nn.Dropout(args.decoder_dropout)
        self.type_cls = nn.Linear(hidden_size, num_labels)
        self.sent_cls = nn.Linear(hidden_size, num_labels)

    def forward(self, query_emb, text_emb, relative_pos, trigger_mask, mask, hand_feature):
        '''
        :param query_emb: [b, 4, e]
        :param text_emb: [b, t-4, e]
        :param relative_pos: [b, t-4, e]
        :param trigger_mask: [b, t-4]
        :param mask:
        :return:
        '''
        trigger_emb = torch.bmm(trigger_mask.unsqueeze(1).float(), text_emb).squeeze(1)  # [b, e]
        trigger_emb = trigger_emb / 2

        h_cln = self.CLN(text_emb, trigger_emb)
        h_sa = self.SA(h_cln, h_cln, h_cln, mask)
        rp_emb = self.relative_pos_embed(relative_pos)

        inp = torch.cat([h_cln, h_sa, rp_emb, hand_feature], dim=-1)
        inp = self.dropout(inp)
        inp = torch.relu(self.hidden(inp))
        inp = self.dropout(inp)

        p_s = torch.sigmoid(self.head_cls(inp))
        p_e = torch.sigmoid(self.tail_cls(inp))

        query_emb = torch.mean(query_emb, dim=1)  # [b, e]
        p_ty = torch.sigmoid(self.type_cls(query_emb))  # [b, 1]

        sent_ctx = self.GlobalAttention(trigger_emb, text_emb, mask)
        p_g = torch.sigmoid(self.sent_cls(sent_ctx))  # [b, 1]

        p_ty = p_ty.unsqueeze(1).expand_as(p_s)
        p_g = p_g.unsqueeze(1).expand_as(p_s)  # [b,1,l] -> [b, t, l]

        p_s = p_s * p_ty * p_g
        p_e = p_e * p_ty * p_g
        return p_s, p_e


class BertTagger(nn.Module):
    def __init__(self, args, model, pos_emb_size):
        super(BertTagger, self).__init__()
        self.embedding = model.embedding
        self.encoder = model.encoder
        self.target = model.target
        self.args = args
        self.args_num = args.args_num
        self.text_seq_len = args.seq_length - 4

        self.hand_feature_size = 40
        self.segs_emb = nn.Embedding(5, self.hand_feature_size)
        self.poss_emb = nn.Embedding(29, self.hand_feature_size)
        self.ners_emb = nn.Embedding(14, self.hand_feature_size)
        self.deps_emb = nn.Embedding(15, self.hand_feature_size)

        self.trigger_rec = TriggerRec(args, args.hidden_size,
                                      self.hand_feature_size * 4)
        self.args_rec = ArgsRec(args, args.hidden_size, self.args_num, self.text_seq_len, pos_emb_size,
                                self.hand_feature_size * 4)
        self.bce = nn.BCELoss(reduction='none')

    def forward(self,
                tokens, segment, mask,
                trigger_s, trigger_e, relative_pos, trigger_mask,
                args_s, args_e,
                args_m, word_segs, word_poss, ners, deps):
        '''

        :param tokens: [b, t]
        :param segment: [b, t]
        :param mask: [b, t], 0 if masked
        :param trigger_s: [b, t-4]
        :param trigger_e: [b, t-4]
        :param relative_pos:
        :param trigger_mask: [0000011000000]
        :param args_s: [b, l, t-4]
        :param args_e: [b, l, t-4]
        :param args_m: [b, k]
        :return:
        '''
        tokens_emb = self.embedding(tokens, segment)
        output_emb = self.encoder(tokens_emb, segment)  # [b, s, e]

        # 这个部分是人工特征，可以考虑暂时删去
        segs_emb = self.segs_emb(word_segs)  # [b, t, 40]
        poss_emb = self.poss_emb(word_poss)
        ners_emb = self.ners_emb(ners)
        deps_emb = self.deps_emb(deps)
        hand_feature = torch.cat([segs_emb, poss_emb, ners_emb, deps_emb], dim=-1)

        query_emb = output_emb[:, 1:3, :]  # [b,2,e]
        text_emb = output_emb[:, 4:, :]
        text_mask = mask[:, 4:]

        p_s, p_e, text_emb = self.trigger_rec(query_emb, text_emb, text_mask, hand_feature)
        p_s = p_s.pow(self.args.trigger_pow)
        p_e = p_e.pow(self.args.trigger_pow)
        p_s = p_s.squeeze(-1)  # [b, t-4]
        p_e = p_e.squeeze(-1)
        trigger_loss_s = self.bce(p_s, trigger_s)  # [b, t-4]
        trigger_loss_e = self.bce(p_e, trigger_e)

        mask_t = text_mask.float()
        trigger_loss_s = 0.5 * torch.sum(trigger_loss_s.mul(mask_t)) / torch.sum(mask_t)
        trigger_loss_e = 0.5 * torch.sum(trigger_loss_e.mul(mask_t)) / torch.sum(mask_t)

        p_s, p_e = self.args_rec(query_emb, text_emb, relative_pos, trigger_mask, text_mask, hand_feature)
        p_s = p_s.pow(self.args.args_pow)
        p_e = p_e.pow(self.args.args_pow)
        args_loss_s = self.bce(p_s, args_s.transpose(1, 2))  # [b, t, l]
        args_loss_e = self.bce(p_e, args_e.transpose(1, 2))

        mask_a = text_mask.unsqueeze(-1).expand_as(args_loss_s).float()  # [b, t, l]
        args_loss_s = 0.5 * torch.sum(args_loss_s.mul(mask_a)) / torch.sum(mask_a)
        args_loss_e = 0.5 * torch.sum(args_loss_e.mul(mask_a)) / torch.sum(mask_a)

        loss = (1.0 - self.args.args_weight) * (trigger_loss_s + trigger_loss_e) + \
               self.args.args_weight * (args_loss_s + args_loss_e)
        return loss

    def predict_trigger_one(self,
                            tokens, segment, mask, word_segs, word_poss, ners, deps):
        assert tokens.size(0) == 1
        tokens_emb = self.embedding(tokens, segment)
        output_emb = self.encoder(tokens_emb, segment)  # [b, s, e]
        segs_emb = self.segs_emb(word_segs)  # [b, t, 40]
        poss_emb = self.poss_emb(word_poss)
        ners_emb = self.ners_emb(ners)
        deps_emb = self.deps_emb(deps)
        hand_feature = torch.cat([segs_emb, poss_emb, ners_emb, deps_emb], dim=-1)

        query_emb = output_emb[:, 1:3, :]  # [b,2,e]
        text_emb = output_emb[:, 4:, :]
        mask = mask[:, 4:]

        p_s, p_e, text_emb = self.trigger_rec(query_emb, text_emb, mask, hand_feature)
        p_s = p_s.squeeze(-1)  # [b, t-4]
        p_e = p_e.squeeze(-1)

        mask = mask.float()  # [1, t]
        p_s = p_s.mul(mask)
        p_e = p_e.mul(mask)

        p_s = p_s.view(self.text_seq_len).data.cpu().numpy()  # [b, t]
        p_e = p_e.view(self.text_seq_len).data.cpu().numpy()
        return p_s, p_e, text_emb, query_emb, hand_feature

    def predict_args_one(self, query_emb, text_emb, relative_pos, trigger_mask, mask, hand_feature):
        assert text_emb.size(0) == 1
        mask = mask[:, 4:]
        p_s, p_e = self.args_rec(query_emb, text_emb, relative_pos, trigger_mask, mask, hand_feature)
        mask = mask.unsqueeze(-1).expand_as(p_s).float()  # [b, t, l]
        p_s = p_s.mul(mask)
        p_e = p_e.mul(mask)
        p_s = p_s.view(self.text_seq_len, self.args_num).data.cpu().numpy()
        p_e = p_e.view(self.text_seq_len, self.args_num).data.cpu().numpy()
        return p_s, p_e
