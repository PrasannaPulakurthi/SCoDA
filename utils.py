"""
Minimal utilities for metrics and validation.
We keep dependencies light while still using tllib's accuracy/AverageMeter.
"""

import time
from typing import Tuple
import torch
import torch.nn.functional as F
from tqdm import tqdm
from tllib.utils.metric import accuracy
from tllib.utils.meter import AverageMeter


@torch.no_grad()
def validate(val_loader, model, device) -> float:
    """Compute top-1 accuracy on a dataloader."""
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    model.eval()
    end = time.time()
    for _, data, target in tqdm(val_loader):
        data, target = data.to(device), target.to(device)
        feats, output = model(data)
        loss = F.cross_entropy(output, target)
        acc1, = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), data.size(0))
        top1.update(acc1.item(), data.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

    print(f' * Acc@1 {top1.avg:.3f}')
    return top1.avg
