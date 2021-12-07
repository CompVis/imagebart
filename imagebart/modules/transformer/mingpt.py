"""
source: https://github.com/karpathy/minGPT/
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import top_k_top_p_filtering

from imagebart.modules.xtransformers.positional_embeddings import apply_rotary_pos_emb


def layers_from_width(d, a=5.039, b=5.55e-2):
    """as in https://arxiv.org/pdf/2006.12467.pdf"""
    return (math.log(d)-a)/b


class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, f"n_embd is {config.n_embd} but n_head is {config.n_head}."
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        mask = torch.tril(torch.ones(config.block_size,
                                     config.block_size))
        if hasattr(config, "n_unmasked"):
            mask[:config.n_unmasked, :config.n_unmasked] = 1
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None, rotary_pos_emb=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if rotary_pos_emb is not None:
            l = rotary_pos_emb.shape[-1]
            (ql, qr), (kl, kr), (vl, vr) = map(lambda t: (t[..., :l], t[..., l:]), (q, k, v))
            ql, kl = map(lambda t: apply_rotary_pos_emb(t, rotary_pos_emb), (ql, kl))
            q, k, v = map(lambda t: torch.cat(t, dim=-1), ((ql, qr), (kl, kr), (vl, vr)))

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

    def forward_with_past(self, x, layer_past=None, rotary_pos_emb=None):
        assert rotary_pos_emb is None, 'just for debugging'
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        if rotary_pos_emb is not None:
            l = rotary_pos_emb.shape[-1]
            (ql, qr), (kl, kr), (vl, vr) = map(lambda t: (t[..., :l], t[..., l:]), (q, k, v))
            ql, kl = map(lambda t: apply_rotary_pos_emb(t, rotary_pos_emb), (ql, kl))
            q, k, v = map(lambda t: torch.cat(t, dim=-1), ((ql, qr), (kl, kr), (vl, vr)))

        present = torch.stack((k, v))
        #present = torch.stack((k.clone(), v.clone()))  # (2, B, nh, 1, hs)

        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        # causal self-attention; Self-attend: (B, nh, Tq, hs) x (B, nh, hs, Tk) -> (B, nh, Tq, Tk)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if layer_past is None:
            pass

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, Tq, Tk) x (B, nh, Tk, hs) -> (B, nh, Tq, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, present


class CausalCrossAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, f"n_embd is {config.n_embd} but n_head is {config.n_head}."
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        block_size = config.block_size
        cond_length = config.n_unmasked
        data_length = block_size-cond_length+1
        mask = np.zeros((data_length, block_size), dtype=np.float32)
        mask[:,:cond_length]=1 # make conditioning visible
        submask=np.tril(np.ones((data_length-1,data_length-1), dtype=np.float32)) # causal submask
        mask[1:,cond_length:] = submask
        mask = torch.tensor(mask)

        self.register_buffer("mask", mask.view(1, 1, data_length, block_size))
        self.n_head = config.n_head

    def forward(self, x_q, x_kv, layer_past=None, rotary_pos_emb=None):
        B, T_q, C = x_q.size()
        _B, T_kv, _C = x_kv.size()
        assert B==_B and C==_C

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.query(x_q).view(B, T_q, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = self.key(x_kv).view(B, T_kv, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x_kv).view(B, T_kv, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if rotary_pos_emb is not None:
            l = rotary_pos_emb.shape[-1]
            (ql, qr), (kl, kr), (vl, vr) = map(lambda t: (t[..., :l], t[..., l:]), (q, k, v))
            ql, kl = map(lambda t: apply_rotary_pos_emb(t, rotary_pos_emb), (ql, kl))
            q, k, v = map(lambda t: torch.cat(t, dim=-1), ((ql, qr), (kl, kr), (vl, vr)))

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T_q,:T_kv] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T_q, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

    def forward_with_past(self, x, layer_past=None, rotary_pos_emb=None):
        raise NotImplementedError(":(")


class Block(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),  # nice
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, rotary_pos_emb=None):
        x = x + self.attn(self.ln1(x), rotary_pos_emb=rotary_pos_emb)
        x = x + self.mlp(self.ln2(x))
        return x

    def forward_with_past(self, x, rotary_pos_emb=None, layer_past=None):
        assert rotary_pos_emb is None, 'just for debugging'
        attn, present = self.attn.forward_with_past(self.ln1(x), rotary_pos_emb=rotary_pos_emb, layer_past=layer_past)
        # layer past: tuple of length two with B, nh, T, hs
        x = x + attn
        x = x + self.mlp(self.ln2(x))
        return x, present


class CrossBlock(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self, config):
        super().__init__()
        self.lnq = nn.LayerNorm(config.n_embd)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalCrossAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),  # nice
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x_q, x_kv, rotary_pos_emb=None):
        x = x_q + self.attn(x_q=self.lnq(x_q),
                            x_kv=self.ln1(x_kv),
                            rotary_pos_emb=rotary_pos_emb)
        x = x + self.mlp(self.ln2(x))
        return x

    def forward_with_past(self, x, rotary_pos_emb=None, layer_past=None):
        raise NotImplementedError()


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """
    def __init__(self, vocab_size, block_size, n_layer=12, n_head=8, n_embd=256,
                 embd_pdrop=0., resid_pdrop=0., attn_pdrop=0., n_unmasked=0,
                 input_vocab_size=None, autoset_layers=False):
        super().__init__()
        recommended_layers = int(np.around(layers_from_width(n_embd)))
        if autoset_layers:
            n_layer = recommended_layers
        print(f"Training with a width of n_embed = {n_embd} and L = {n_layer} layers. "
              f"https://arxiv.org/pdf/2006.12467.pdf suggest that one should use {recommended_layers} layers.")
        config = GPTConfig(vocab_size=vocab_size, block_size=block_size,
                           embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
                           n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                           n_unmasked=n_unmasked)

        # input embedding stem
        in_vocab_size = vocab_size if not input_vocab_size else input_vocab_size
        self.tok_emb = nn.Embedding(in_vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.config = config

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, embeddings=None, targets=None, return_layers=False, token_embeddings=None):
        # forward the GPT model
        if token_embeddings is None:
            token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
            if embeddings is not None:  # prepend explicit embeddings
                token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)

        t = token_embeddings.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)

        if return_layers:
            layers = [x]
            for block in self.blocks:
                x = block(x)
                layers.append(x)
            return layers

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def forward_with_past(self, idx, embeddings=None, targets=None, token_embeddings=None,
                          past=None, past_length=None):

        if token_embeddings is None:
            token_embeddings = self.tok_emb(idx)    # each index maps to a (learnable) vector
            if embeddings is not None:              # prepend explicit embeddings
                token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)
        assert not self.training
        if past is not None:
            assert past_length is not None
            past = torch.cat(past, dim=-2)   # n_layer, 2, b, nh, len_past, dim_head
            past_shape = list(past.shape)
            expected_shape = [self.config.n_layer, 2, idx.shape[0], self.config.n_head, past_length, self.config.n_embd//self.config.n_head]
            assert past_shape == expected_shape, f"{past_shape} =/= {expected_shape}"
            position_embeddings = self.pos_emb[:, past_length, :]  # each position maps to a (learnable) vector
        else:
            position_embeddings = self.pos_emb[:, :token_embeddings.shape[1], :]

        x = self.drop(token_embeddings + position_embeddings)

        presents = []  # accumulate over layers
        for i, block in enumerate(self.blocks):
            x, present = block.forward_with_past(x, layer_past=past[i, ...] if past is not None else None)  # take from layer
            presents.append(present)

        x = self.ln_f(x)
        logits = self.head(x)
        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, torch.stack(presents)  # _, _, n_layer, 2, b, nh, 1, dim_head


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed
        logits, _ = model(x_cond)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)

    return x


@torch.no_grad()
def sample_with_past(x, model, steps, temperature=1., sample_logits=True,
                     top_k=None, callback=None, guide=None, top_p=None,
                     embeddings=None):
    # x is conditioning
    sample = x
    cond_len = x.shape[1]
    if embeddings is not None:
        cond_len += embeddings.shape[1]
    past = None
    for n in range(steps):
        if callback is not None:
            callback(n)
        logits, _, present = model.forward_with_past(x, embeddings=embeddings, past=past, past_length=(n+cond_len-1))
        embeddings = None # only pass in first time
        if past is None:
            past = [present]
        else:
            past.append(present)
        if guide is not None:
            logits = logits + guide[:, [n]]
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            if top_p is not None:
                logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            else:
                logits = top_k_logits(logits, top_k)

        probs = F.softmax(logits, dim=-1)
        if not sample_logits:
            _, x = torch.topk(probs, k=1, dim=-1)
        else:
            x = torch.multinomial(probs, num_samples=1)
        # append to the sequence and continue
        sample = torch.cat((sample, x), dim=1)
    del past
    sample = sample[:, -steps:]  # cut conditioning off
    return sample


@torch.no_grad()
def sample_vanilla(x, model, steps, temperature=1.0, sample_logits=False,
                   top_k=None, embeddings=None):
    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed
        logits, _ = model(x_cond, embeddings=embeddings)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample_logits:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)
    x = x[:, -steps:]
    return x


def seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test1():
    import time
    #torch.use_deterministic_algorithms(True)
    SEED = 142
    h = w = 4
    b = 1
    clen = 2
    cb = lambda n: print(n)
    cb = None
    elen = 3

    SAMPLE = True
    DEVICE = "cpu"

    # test past
    device = torch.device(DEVICE)
    model = GPT(vocab_size=1024, block_size=h*w+(elen+clen-1), n_embd=32).to(device).eval()
    x = torch.randint(0, 1024, size=(b, clen)).to(device)  # start
    emb = torch.randn(b, elen, 32)

    print(f"in goes: {x}")

    # with past
    seed(SEED)
    t0 = time.time()
    s0 = sample_with_past(x, model, embeddings=emb, steps=h * w, sample_logits=SAMPLE, callback=cb)
    t1 = time.time()

    # without past
    seed(SEED)
    s1 = sample_vanilla(x, model, embeddings=emb, steps=h*w, sample_logits=SAMPLE)
    t2 = time.time()

    print(f"s0 (with past): time = {t1-t0:.2f}s")
    print(s0)
    print(f"s1 (no past): time = {t2-t1:.2f}s")
    print(s1)
    print("are equal:", torch.equal(s0, s1))
    print("done.")


if __name__ == "__main__":
    test1()
