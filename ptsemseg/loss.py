import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

#this file isn't used
def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h > ht and w > wt:  # upsample labels
        target = target.unsqueeze(1)
        target = F.upsample(target, size=(h, w), mode="nearest")
        target = target.squeeze(1)
    elif h < ht and w < wt:  # upsample images
        input = F.upsample(input, size=(ht, wt), mode="bilinear")
    elif h != ht and w != wt:
        raise Exception("Only support upsampling")

    d = {1: 15539.0, 2: 2887.0, 3: 7154.0, 4: 3700.0, 5: 17505.0, 6: 1016.0} #30slides_256_50label_10overlap
    d_sum = sum(d.values())
    class_weights = [0,1-d[1]/d_sum, 1-d[2]/d_sum, 1-d[3]/d_sum, 1-d[4]/d_sum, 1-d[5]/d_sum, 1-d[6]/d_sum]
    print(class_weights)
    
    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    #torch.nn.functional.cross_entropy This criterion combines log_softmax and nll_loss in a single function.
    loss = F.cross_entropy(
        input, target, weight=torch.FloatTensor(class_weights).cuda(), size_average=size_average, ignore_index=250
        #input, target, weight=weight, size_average=size_average, ignore_index=250
    )
    return loss


def bootstrapped_cross_entropy2d(input, target, K, weight=None, size_average=True):

    batch_size = input.size()[0]
    #uses nll_loss
    def _bootstrap_xentropy_single(input, target, K, weight=None, size_average=True):
        n, c, h, w = input.size()
        log_p = F.log_softmax(input, dim=1)
        log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        log_p = log_p[target.view(n * h * w, 1).repeat(1, c) >= 0]
        log_p = log_p.view(-1, c)

        mask = target >= 0
        target = target[mask]
        loss = F.nll_loss(
            log_p,
            target,
            weight=weight,
            ignore_index=250,
            reduce=False,
            size_average=False,
        )
        topk_loss, _ = loss.topk(K)
        reduced_topk_loss = topk_loss.sum() / K

        return reduced_topk_loss

    loss = 0.0
    # Bootstrap from each image not entire batch
    for i in range(batch_size):
        loss += _bootstrap_xentropy_single(
            input=torch.unsqueeze(input[i], 0),
            target=torch.unsqueeze(target[i], 0),
            K=K,
            weight=weight,
            size_average=size_average,
        )
    return loss / float(batch_size)


def multi_scale_cross_entropy2d(
    input, target, weight=None, size_average=True, scale_weight=None
):
    # Auxiliary training for PSPNet [1.0, 0.4] and ICNet [1.0, 0.4, 0.16]
    if scale_weight == None:  # scale_weight: torch tensor type
        n_inp = len(input)
        scale = 0.4
        scale_weight = torch.pow(scale * torch.ones(n_inp), torch.arange(n_inp))
        if input.is_cuda:
            scale_weight = scale_weight.cuda()

    loss = 0.0
    for i, inp in enumerate(input):
        loss = loss + scale_weight[i] * cross_entropy2d(
            input=inp, target=target, weight=weight, size_average=size_average
        )

    return loss
