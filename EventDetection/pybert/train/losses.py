from torch.nn import CrossEntropyLoss
from torch.nn import BCEWithLogitsLoss
from torch.nn import BCELoss
import torch

__call__ = ['CrossEntropy', 'BCEWithLogLoss']


class CrossEntropy(object):
    def __init__(self):
        self.loss_f = CrossEntropyLoss()

    def __call__(self, output, target):
        loss = self.loss_f(input=output, target=target)
        return loss


class BCEWithLogLoss(object):
    def __init__(self):
        self.loss_fn = BCEWithLogitsLoss()

    def __call__(self, output, target):
        output = output.float()
        target = target.float()
        loss = self.loss_fn(input=output, target=target)
        return loss


class BCEWithLogLossWithInversion(object):
    def __init__(self):
        self.loss_fn = BCEWithLogitsLoss()

    def __call__(self, output, target):
        output = output.float()
        target = target.float()
        loss = self.loss_fn(input=output, target=target)
        loss = torch.abs(loss - 0.001) + 0.001
        return loss


class BCETheLoss(object):
    '''
    Simply the BCELoss without sigmoid
    '''

    def __init__(self):
        self.loss_fn = BCELoss()

    def __call__(self, output, target):
        output = output.float()
        target = target.float()
        loss = self.loss_fn(input=output, target=target)
        return loss


class MultiLabelCategoricalCrossEntropy(object):
    def __init__(self):
        pass

    def __call__(self, output, target):
        '''
        :param output: predicted probabilities, which must be a real scalar. Do NOT add sigmoid or softmax.
        :param target: binary, 1 for target, 0 for others
        :return: the loss
        '''
        y_pred, y_true = output, target
        y_true = y_true.float()
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12

        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        loss = neg_loss + pos_loss
        loss = torch.mean(loss)
        return loss
