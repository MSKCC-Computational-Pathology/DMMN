import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from pdb import set_trace


def cross_entropy(input, target, class_weights, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)

    loss = F.cross_entropy(
        input, target, weight=torch.FloatTensor(class_weights).cuda(), size_average=True, ignore_index=0
    )

    return loss
