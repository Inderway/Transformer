# -*- coding=utf8 -*-
# Aug 12, 2022
# the training and inference process of transformer
# Implemented by Alexander Rush
# Reconstructed by Wei Yin

from Transformer import subsequent_mask
import time
import torch.nn as nn
import torch


class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    def step(self):
        None

class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, tgt=None, pad=2):  # 2 = <blank>
        self.src = src
        # 2表示<blank>，若src中有元素等于2，则要置为false即0
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            # 第1维的数据去掉最后一个
            self.tgt = tgt[:, :-1]
            # 第1维的数据去掉第一个，因为第一个为起始符<s>？
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            # token个数为tgt_y中不为<blank>的个数，sum无视维度将所有元素相加
            # 使用.data使得计算不会进入ntokens的计算历史
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        # 把<blank>无视掉
        tgt_mask = (tgt != pad).unsqueeze(-2)
        # subsequent_mask兼顾无视<blanK>
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask

class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed

def run_epoch(
    data_iter,
    model,
    loss_compute,
    optimizer,
    scheduler,
    mode="train",
    accum_iter=1,
    train_state=TrainState(),
):
    """Train a single epoch"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        )
        # loss_compute待补充
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        # loss_node = loss_node / accum_iter
        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            # shape[0]的值表示batch中的输入句数量
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss / total_tokens, train_state


def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        # vocab的size
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        # target的shape为vocab
        # 不满足条件则抛出assertError
        assert x.size(1) == self.size
        # shape为ntokens x vocab
        # 先使true_dist具有x的shape
        true_dist = x.data.clone()
        # size-2表示去掉首尾
        true_dist.fill_(self.smoothing / (self.size - 2))
        # index的shape为vocab x 1
        # true_dist[i][index[i][j]]=1-smoothing
        # 给golden label赋上confidence
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # padding索引处赋0
        true_dist[:, self.padding_idx] = 0
        # nonzero返回由非0元素的坐标组成的列表，此处返回等于padding_idx的坐标
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            # 将对应position的prediction给padding掉
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        # 用正确分布与预测计算loss
        return self.criterion(x, true_dist.clone().detach())


