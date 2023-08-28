import torch
from torch.nn import L1Loss, MSELoss
import numpy as np


def get_device():
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    return device


def get_batch_size(x_input):
    if isinstance(x_input, list) or isinstance(x_input, tuple):
        return x_input[0].size(0)
    else:
        return x_input.size(0)


def loss_criterion(a, b, weights, eps=0.01, relative=True, reduction='mean'):
    ''' a: tensor, output of the model
    b: tensor, ground truth
    weights: list of floats, weight of the L1 and L2 loss
    eps: float, small positive value to avoid division by zero with relative loss
    relative: boolean, whether to compute relative or absolue loss
    '''

    # if a and b are composed of multiple tensors, compute the mean error over all tensors
    if isinstance(a, tuple):
        if relative:
            base = torch.abs(b[0]) + eps
            L1 = L1Loss(reduction=reduction)(a[0]/base, b[0]/base)
            L2 = MSELoss(reduction=reduction)(a[0]/base, b[0]/base)
        else:
            L1 = L1Loss(reduction=reduction)(a[0], b[0])
            L2 = MSELoss(reduction=reduction)(a[0], b[0])
            
        for ai, bi in zip(a[1:], b[1:]):
            if relative:
                base = torch.abs(bi) + eps
                L1 += L1Loss(reduction=reduction)(ai/base, bi/base)
                L2 += MSELoss(reduction=reduction)(ai/base, bi/base)
            else:
                L1 += L1Loss(reduction=reduction)(ai, bi)
                L2 += MSELoss(reduction=reduction)(ai, bi)

    else: # a and b are tensors
        if relative:
            base = torch.abs(b) + eps
            L1 = L1Loss(reduction=reduction)(a/base, b/base)
            L2 = MSELoss(reduction=reduction)(a/base, b/base)
        else:
            L1 = L1Loss(reduction=reduction)(a,b)
            L2 = MSELoss(reduction=reduction)(a,b)

    return weights[0] * L1 + weights[1] * L2


class RunningAverage:
    """Computes and stores the average
    """

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, value, n=1):
        self.count += n
        self.sum += value * n
        self.avg = self.sum / self.count


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
