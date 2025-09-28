"""
Training script for:
Bridging Domain Shifts Through Self-contrastive Learning and Distribution Alignment (SCoDA)
- Source: SVHN
- Target: MNIST
This script implements:
  * Cross-entropy on source labels
  * MK-MMD alignment between source/target features (via tllib)
  * Self-contrastive loss on both domains (two views per sample)

Notes:
- This is a simplified version based on your digest. It removes the 'analysis' phase
  and any t-SNE/A-distance visualization to keep the code compact and reproducible.
"""

import os
import os.path as osp
import argparse
import random
import warnings
from typing import Tuple

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import pandas as pd

from tllib.alignment.dan import MultipleKernelMaximumMeanDiscrepancy
from tllib.modules.kernels import GaussianKernel
from tllib.utils.meter import AverageMeter
from tllib.utils.metric import accuracy

from datasets import (
    MNISTPair, SVHNPair,
    train_transform_mnist, test_transform_mnist,
    train_transform_svhn,  test_transform_svhn,
)
from model import ImageClassifier, info_nce_pairwise
from utils import validate


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    warnings.warn(
        "You set a seed; CUDNN deterministic mode is on and may slow down training."
    )


def build_dataloaders(batch_size: int,num_workers:int):
    """
    Build SVHN->MNIST paired dataloaders with equalized dataset lengths.
    """
    mnist_train = MNISTPair(root='data', train=True,
                            transform=train_transform_mnist, test_transform=test_transform_mnist, download=True)
    svhn_train = SVHNPair(root='data', split='train',
                          transform=train_transform_svhn, test_transform=test_transform_svhn, download=True)
    mnist_val = MNISTPair(root='data', train=False,
                          transform=test_transform_mnist, test_transform=test_transform_mnist, download=True)
    svhn_val = SVHNPair(root='data', split='test',
                        transform=test_transform_svhn, test_transform=test_transform_svhn, download=True)

    # Balance lengths for zipped training
    n = min(len(mnist_train), len(svhn_train))
    mnist_train = Subset(mnist_train, range(n))
    svhn_train = Subset(svhn_train, range(n))
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True, persistent_workers=True)

    mnist_loader = DataLoader(mnist_train, shuffle=True, drop_last=True,  **common)
    svhn_loader   = DataLoader(svhn_train, shuffle=True, drop_last=True,  **common)
    mnist_val_loader = DataLoader(mnist_val, shuffle=False, **common)
    svhn_val_loader  = DataLoader(svhn_val, shuffle=False, **common)

    return svhn_loader, mnist_loader, svhn_val_loader, mnist_val_loader


def train_one_epoch(
    source_loader, target_loader, model, mmd_loss, optimizer, lr_scheduler, epoch, trade_off: float
):
    """Single training epoch."""
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':3.1f')
    total_losses = AverageMeter('Loss', ':3.2f')
    transfer_losses = AverageMeter('MMD', ':5.4f')
    cls_accs = AverageMeter('ClsAcc', ':3.1f')
    contrastive_losses = AverageMeter('InfoNCE', ':5.4f')

    model.train()
    mmd_loss.train()

    loop = tqdm(zip(source_loader, target_loader), total=min(len(source_loader), len(target_loader)))
    for (xs1, xs2, ys), (xt1, xt2, _ ) in loop:
        xs1, xs2, ys = xs1.to(DEVICE), xs2.to(DEVICE), ys.to(DEVICE)
        xt1, xt2 = xt1.to(DEVICE), xt2.to(DEVICE)

        # Forward
        x_all = torch.cat([xs1, xs2, xt1, xt2], dim=0)           # (4B, C, H, W)
        feats_all, logits_all = model(x_all)                      # (4B, D), (4B, K)
        B = xs1.size(0)

        fs1, fs2, ft1, ft2 = feats_all[:B], feats_all[B:2*B], feats_all[2*B:3*B], feats_all[3*B:]
        logits_s = logits_all[:B]

        # Losses unchanged:
        ce   = F.cross_entropy(logits_s, ys)
        mmd  = mmd_loss(fs1, ft1)
        nce_src = info_nce_pairwise(fs1, fs2)
        nce_tgt = info_nce_pairwise(ft1, ft2)
        nce = nce_src+nce_tgt
        loss = ce + (mmd + nce) * trade_off

        # Metrics
        acc1, = accuracy(logits_s, ys, topk=(1,))
        cls_accs.update(acc1.item(), xs1.size(0))
        transfer_losses.update(mmd.item(), xs1.size(0))
        contrastive_losses.update(nce_src.item(), xs1.size(0))
        total_losses.update(loss.item(), xs1.size(0))

        # Step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        loop.set_description(f"Epoch {epoch}")
        loop.set_postfix(loss=total_losses.avg, mmd=transfer_losses.avg, nce=contrastive_losses.avg, acc=cls_accs.avg)

    return transfer_losses.avg, total_losses.avg, cls_accs.avg, contrastive_losses.avg


def main():
    parser = argparse.ArgumentParser(description="SCoDA: Self-contrastive + Distribution Alignment (SVHN→MNIST)")
    parser.add_argument("--batch-size", "-b", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--wd", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr-gamma", type=float, default=0.0003)
    parser.add_argument("--lr-decay", type=float, default=0.75)
    parser.add_argument("--bottleneck-dim", type=int, default=256)
    parser.add_argument("--scratch", action="store_true", help="If set, do not fine-tune backbone with lower LR.")
    parser.add_argument("--trade-off", type=float, default=1.0, help="Weight for MMD + contrastive loss.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logdir", type=str, default="runs_s2m")
    parser.add_argument("--val_epoch", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    set_seed(args.seed)
    cudnn.benchmark = True

    # Data
    svhn_loader, mnist_loader, svhn_val_loader, mnist_val_loader = build_dataloaders(args.batch_size,args.num_workers)

    # Model
    model = ImageClassifier(num_classes=10, bottleneck_dim=args.bottleneck_dim, finetune_backbone=not args.scratch).to(DEVICE)

    # Optimizer / Scheduler
    optimizer = AdamW(model.get_parameters(base_lr=args.lr), lr=args.lr, weight_decay=args.wd)
    lr_scheduler = LambdaLR(optimizer, lambda it: args.lr * (1.0 + args.lr_gamma * float(it)) ** (-args.lr_decay))

    # Alignment loss (MK-MMD)
    mmd_loss = MultipleKernelMaximumMeanDiscrepancy(kernels=[GaussianKernel(alpha=2)], linear=True)

    # Logs
    history = {
        "epoch": [], "train_loss": [], "src_val_acc": [], "tgt_val_acc": [],
        "mmd_loss": [], "contrastive_loss": []
    }

    best_tgt = 0.0
    best_path = osp.join(args.logdir, "best.pt")
    last_path = osp.join(args.logdir, "last.pt")

    # Train
    for epoch in range(1, args.epochs + 1):
        mmd_avg, loss_avg, acc_src_avg, nce_avg = train_one_epoch(
            svhn_loader, mnist_loader, model, mmd_loss, optimizer, lr_scheduler, epoch, trade_off=args.trade_off
        )

        if epoch%args.val_epoch==0:
            # Validation
            acc_src = validate(svhn_val_loader, model, DEVICE)
            acc_tgt = validate(mnist_val_loader, model, DEVICE)

            # Save
            history["epoch"].append(epoch)
            history["train_loss"].append(loss_avg)
            history["src_val_acc"].append(acc_src)
            history["tgt_val_acc"].append(acc_tgt)
            history["mmd_loss"].append(mmd_avg)
            history["contrastive_loss"].append(nce_avg)
            pd.DataFrame(history).to_csv(osp.join(args.logdir, "training_log.csv"), index=False)
            
            torch.save(model.state_dict(), last_path)
            if acc_tgt > best_tgt:
                torch.save(model.state_dict(), best_path)
                best_tgt = acc_tgt

            print(f"[Epoch {epoch}] tgt_acc={acc_tgt:.2f} (best={best_tgt:.2f})")

    # Final test on best checkpoint
    model.load_state_dict(torch.load(best_path, map_location="cpu"))
    final_acc = validate(mnist_val_loader, model, DEVICE)
    print(f"[FINAL] Best target Acc@1 = {final_acc:.2f}")


if __name__ == "__main__":
    main()
