import math
import torch
import torch.nn as nn
from copy import deepcopy
from torchmetrics.functional.classification import accuracy


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


class ModelEMA(object):
    def __init__(self, model, decay=0.9999, updates=0):
        # create EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000.))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()


def calc_accuracy(output, target, num_classes, device):
    acc1 = accuracy(output, target, task='multiclass', num_classes=num_classes, top_k=1).to(device)

    if num_classes >= 5:
        acc5 = accuracy(output, target, task='multiclass', num_classes=num_classes, top_k=5).to(device)
    else:
        acc5 = acc1

    return acc1, acc5
