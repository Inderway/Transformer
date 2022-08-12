# -*- coding=utf8 -*-
# Aug 12, 2022
# first example of transformer, to implement a simple copy-task
# Implemented by Alexander Rush
# Reconstructed by Wei Yin

import torch
from train import Batch, run_epoch, DummyOptimizer, DummyScheduler, LabelSmoothing, rate
from Transformer import subsequent_mask, make_model
from torch.optim.lr_scheduler import LambdaLR

def data_gen(V, batch_size, nbatches):
    "Generate random data for a src-tgt copy task."
    # 共nbatch个batch
    for i in range(nbatches):
        # 指定data的shape为batch_size x 10，randint范围为[1,V)
        # 一个batch，batch_size个输入，每个输入10个token
        data = torch.randint(1, V, size=(batch_size, 10))
        # 令每个输入的第一个token为1
        data[:, 0] = 1
        # 不计算梯度
        src = data.requires_grad_(False).clone().detach()
        tgt = data.requires_grad_(False).clone().detach()
        # 0代表pad
        yield Batch(src, tgt, 0)


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = (
            self.criterion(
                x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
            )
            / norm
        )
        return sloss.data * norm, sloss


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    # 获得encode结果
    memory = model.encode(src, src_mask)
    # 先用start_symbol填充初始输出
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        # 获得概率最大的token的索引
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        # 扩充输出
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys


# Train the simple copy task.
def example_simple_model():
    # 词表包含[0,10]
    V = 11
    # 每个输入的第一个token pad掉
    # 不做label smoothing
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    # encoder和decoder各含2层子层
    model = make_model(V, V, N=2)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, model_size=model.src_embed[0].d_model, factor=1.0, warmup=400
        ),
    )

    batch_size = 80
    for epoch in range(20):
        model.train()
        run_epoch(
            # 20个批，每批80个输入
            data_gen(V, batch_size, 20),
            model,
            SimpleLossCompute(model.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train",
        )
        # 用于评估模型，不做优化
        model.eval()
        # 获得total_loss / total_tokens
        run_epoch(
            # 5个批，每批80个输入
            data_gen(V, batch_size, 5),
            model,
            SimpleLossCompute(model.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )[0]

    model.eval()
    src = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    # 获得句长
    max_len = src.shape[1]
    # 所有token都进入计算
    src_mask = torch.ones(1, 1, max_len)
    print(greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=0))

example_simple_model()