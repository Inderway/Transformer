# -*- coding=utf8 -*-
# Aug 12, 2022
# the core of transformer
# Implemented by Alexander Rush
# Reconstructed by Wei Yin



import torch
import torch.nn as nn
from torch.nn.functional import log_softmax
import math
import copy


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        # src_embed和tgt_embed用于将src, tgt转换为embedding
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        #线性层的输出维度为词典维度
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        # 将输入x经过线性层计算，softmax加log防止数据溢出
        return log_softmax(self.proj(x), dim=-1)

def clones(module, N):
    "Produce N identical layers."
    # deepcopy复制出一个独立新个体，原对象改变不会影响新对象
    # 在Transformer中，N取6
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        # 通过clones获得一个Module序列
        self.layers = clones(layer, N)
        # layer.size为hidden unit个数
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        # 传入输入并原地计算，输出作为下一层的输入
        # x的shape为1xnx512,mask的shape为1x1xn
        for layer in self.layers:
            x = layer(x, mask)
        # 返回LN后的结果
        return self.norm(x)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # features表示输入的维度
        # a2, b2对应LN论文中的g和b
        # eps用于防止除0
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        #求一个输入的mu和sigma用于LN
        #keepdim用于保留维度
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        # 首先对输入进行LN，再通过子层
        # 接着对子层的输出进行dropout和残差连接
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        '''
        Args:
            size: 维度
            self_attn: 自注意力层
            feed_forward: 全连接层
            dropout
        '''
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        #每个encoder层包含两个子层
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        #首先获得第一个子层的输出，再作为第二个子层的输入，最后返回
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        # 获得包含N个layer的队列
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            # 原地操作，上一层的输出作为下一层的输入
            x = layer(x, memory, src_mask, tgt_mask)
        #最后经过LN
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        # 每个decoder子层包含三个子层
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        # m为encoder的输出，应用于第二个子层
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    # 定义一个上三角阵，主对角线往上偏移1个单位
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    # shape为1xsizexsize
    # 会比对tensor中每个元素是否等于0，对于元素值只有0和1的tensor，相当于取反
    return subsequent_mask == 0

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    '''
    Args:
        query:nbatches x h x ntokens x d_k
        mask:1x1x1x10
    Return:
    '''
    # 获取query列表的维度
    d_k = query.size(-1)
    # 低维矩阵相乘并除常数
    # shape为nbatches x h x ntokens x ntokens
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        # 如果对应位置mask的元素为0，则将scores的元素置为-1e9
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # 返回attention的值和权值矩阵
    # 前者shape为nbatches x h x ntokens x d_k
    return torch.matmul(p_attn, value), p_attn 


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        # 复制4个线性层
        # ModuleList LinearModulex4，最后一个linear用于结果线性变换
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        '''
        Args:
            query:1xnx512
            mask: 1x1xn
        '''
        if mask is not None:
            # Same mask applied to all h heads.
            # mask的shape变为1x1x1xn
            mask = mask.unsqueeze(1)
        # 获得数据批数
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            # zip的结果为4个元组，每个元组由一个LinearModule和query/key/value组成
            for lin, x in zip(self.linears, (query, key, value))
            # lin(x)的shape为1xnx512，经过view后，变为1xnx8x64
            # 经转置后变为1x8xnx64
        ]
        # 记右侧临时列表为list，则query=list[0], key=list[1], value=list[2]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )
        # x的shape为nbatches x h x ntokens x d_k
        # 后者shape为nbatches x h x ntokens x ntokens

        # 3) "Concat" using a view and apply a final linear.
        x = (
            # shape变为nbatches x ntokens x h x d_k
            x.transpose(1, 2)
            .contiguous()
            # shape变为nbatches x ntokens x d_model
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        # 返回shape为nbatches x ntokens x d_model
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x的shape为nbatches x ntokens x d_model
        # 返回shape为nbatches x ntokens x d_model
        return self.w_2(self.dropout(self.w_1(x).relu()))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        # 将vocab维的单词表转换为d_model维
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        #x的shape为1xn
        # 返回的shape为1xnx512
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        # pe的shape为5000x512
        pe = torch.zeros(max_len, d_model)
        # position的shape为5000x1 position[i][0]=i, 0<=i<5000
        position = torch.arange(0, max_len).unsqueeze(1)
        # 分母项，实际为1/分母，exp(-2i/512*ln10000)=1/10000^(2i/512)
        # shape为256
        div_term = torch.exp(
            #[0,2,4,...,512]*
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # pe的shape变为5000x512
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe的shape变为1x5000x512
        pe = pe.unsqueeze(0)
        # 该函数注册一个buffer存储非模型参数的数据
        self.register_buffer("pe", pe)

    def forward(self, x):
        # 与位置编码相加，不计算梯度
        # x的1维度为token位置，长为token个数n
        # pe[:,x.size(1)]的shape为1xnx512
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

def make_model(
    src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    # deepcopy用于复制独立的层
    c = copy.deepcopy
    # 初始化各层
    attn = MultiHeadedAttention(h, d_model)
    # encoder和decoder子层中的FFN
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    # 位置编码
    position = PositionalEncoding(d_model, dropout)
    # 组装
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # 参数初始化
    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model