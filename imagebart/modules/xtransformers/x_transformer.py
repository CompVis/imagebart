"""modified from https://github.com/lucidrains/x-transformers/tree/main/x_transformers"""

import torch
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial
from inspect import isfunction
from collections import namedtuple

from einops import rearrange, repeat, reduce

from imagebart.modules.xtransformers.autoregressive_wrapper import AutoregressiveWrapper
from imagebart.modules.xtransformers.positional_embeddings import apply_rotary_pos_emb, rotate_half
from imagebart.modules.xtransformers.positional_embeddings import DepthWiseConv1d, AbsolutePositionalEmbedding, \
    FixedPositionalEmbedding, RotaryEmbedding, RelativePositionBias


# constants

DEFAULT_DIM_HEAD = 64

Intermediates = namedtuple('Intermediates', [
    'pre_softmax_attn',
    'post_softmax_attn'
])

LayerIntermediates = namedtuple('Intermediates', [
    'hiddens',
    'attn_intermediates'
])


# helpers

def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def always(val):
    def inner(*args, **kwargs):
        return val
    return inner


def not_equals(val):
    def inner(x):
        return x != val
    return inner


def equals(val):
    def inner(x):
        return x == val
    return inner


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


# keyword argument helpers

def pick_and_pop(keys, d):
    values = list(map(lambda key: d.pop(key), keys))
    return dict(zip(keys, values))


def group_dict_by_key(cond, d):
    return_val = [dict(), dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)


def string_begins_with(prefix, str):
    return str.startswith(prefix)


def group_by_key_prefix(prefix, d):
    return group_dict_by_key(partial(string_begins_with, prefix), d)


def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs


# classes
class Scale(nn.Module):
    def __init__(self, value, fn):
        super().__init__()
        self.value = value
        self.fn = fn

    def forward(self, x, **kwargs):
        x, *rest = self.fn(x, **kwargs)
        return (x * self.value, *rest)


class Rezero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.zeros(1))

    def forward(self, x, **kwargs):
        x, *rest = self.fn(x, **kwargs)
        return (x * self.g, *rest)


class ScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class Residual(nn.Module):
    def forward(self, x, residual):
        return x + residual


class GRUGating(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gru = nn.GRUCell(dim, dim)

    def forward(self, x, residual):
        gated_output = self.gru(
            rearrange(x, 'b n d -> (b n) d'),
            rearrange(residual, 'b n d -> (b n) d')
        )

        return gated_output.reshape_as(x)


# feedforward

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


# attention.
class Attention(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=DEFAULT_DIM_HEAD,
            heads=8,
            causal=False,
            mask=None,
            talking_heads=False,
            sparse_topk=None,
            use_entmax15=False,
            num_mem_kv=0,
            dropout=0.,
            on_attn=False
    ):
        super().__init__()
        if use_entmax15:
            raise NotImplementedError("Check out entmax activation instead of softmax activation!")
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.causal = causal
        self.mask = mask

        inner_dim = dim_head * heads

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        # talking heads
        self.talking_heads = talking_heads
        if talking_heads:
            self.pre_softmax_proj = nn.Parameter(torch.randn(heads, heads))
            self.post_softmax_proj = nn.Parameter(torch.randn(heads, heads))

        # explicit topk sparse attention
        self.sparse_topk = sparse_topk

        # entmax
        #self.attn_fn = entmax15 if use_entmax15 else F.softmax
        self.attn_fn = F.softmax

        # add memory key / values
        self.num_mem_kv = num_mem_kv
        if num_mem_kv > 0:
            self.mem_k = nn.Parameter(torch.randn(heads, num_mem_kv, dim_head))
            self.mem_v = nn.Parameter(torch.randn(heads, num_mem_kv, dim_head))

        # attention on attention
        self.attn_on_attn = on_attn
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim * 2), nn.GLU()) if on_attn else nn.Linear(inner_dim, dim)

    def forward(
            self,
            x,
            context=None,
            mask=None,
            context_mask=None,
            rel_pos=None,
            sinusoidal_emb=None,
            rotary_pos_emb=None,
            prev_attn=None,
            mem=None
    ):
        b, n, _, h, talking_heads, device = *x.shape, self.heads, self.talking_heads, x.device
        kv_input = default(context, x)

        q_input = x
        k_input = kv_input
        v_input = kv_input

        if exists(mem):
            k_input = torch.cat((mem, k_input), dim=-2)
            v_input = torch.cat((mem, v_input), dim=-2)

        if exists(sinusoidal_emb):
            # in shortformer, the query would start at a position offset depending on the past cached memory
            offset = k_input.shape[-2] - q_input.shape[-2]
            q_input = q_input + sinusoidal_emb(q_input, offset=offset)
            k_input = k_input + sinusoidal_emb(k_input)

        q = self.to_q(q_input)
        k = self.to_k(k_input)
        v = self.to_v(v_input)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        if exists(rotary_pos_emb):
            l = rotary_pos_emb.shape[-1]
            (ql, qr), (kl, kr) = map(lambda t: (t[..., :l], t[..., l:]), (q, k))
            ql, kl = apply_rotary_pos_emb(ql, kl, rotary_pos_emb)
            q = torch.cat((ql, qr), dim=-1)
            k = torch.cat((kl, kr), dim=-1)

        input_mask = None
        if any(map(exists, (mask, context_mask))):
            q_mask = default(mask, lambda: torch.ones((b, n), device=device).bool())
            k_mask = q_mask if not exists(context) else context_mask
            k_mask = default(k_mask, lambda: torch.ones((b, k.shape[-2]), device=device).bool())
            q_mask = rearrange(q_mask, 'b i -> b () i ()')
            k_mask = rearrange(k_mask, 'b j -> b () () j')
            input_mask = q_mask * k_mask

        if self.num_mem_kv > 0:
            mem_k, mem_v = map(lambda t: repeat(t, 'h n d -> b h n d', b=b), (self.mem_k, self.mem_v))
            k = torch.cat((mem_k, k), dim=-2)
            v = torch.cat((mem_v, v), dim=-2)
            if exists(input_mask):
                input_mask = F.pad(input_mask, (self.num_mem_kv, 0), value=True)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = max_neg_value(dots)

        if exists(prev_attn):
            dots = dots + prev_attn

        pre_softmax_attn = dots

        if talking_heads:
            dots = einsum('b h i j, h k -> b k i j', dots, self.pre_softmax_proj).contiguous()

        if exists(rel_pos):
            dots = rel_pos(dots)

        if exists(input_mask):
            dots.masked_fill_(~input_mask, mask_value)
            del input_mask

        if self.causal:
            i, j = dots.shape[-2:]
            r = torch.arange(i, device=device)
            mask = rearrange(r, 'i -> () () i ()') < rearrange(r, 'j -> () () () j')
            mask = F.pad(mask, (j - i, 0), value=False)
            dots.masked_fill_(mask, mask_value)
            del mask

        if exists(self.sparse_topk) and self.sparse_topk < dots.shape[-1]:
            top, _ = dots.topk(self.sparse_topk, dim=-1)
            vk = top[..., -1].unsqueeze(-1).expand_as(dots)
            mask = dots < vk
            dots.masked_fill_(mask, mask_value)
            del mask

        attn = self.attn_fn(dots, dim=-1)
        post_softmax_attn = attn

        attn = self.dropout(attn)

        if talking_heads:
            attn = einsum('b h i j, h k -> b k i j', attn, self.post_softmax_proj).contiguous()

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        intermediates = Intermediates(
            pre_softmax_attn=pre_softmax_attn,
            post_softmax_attn=post_softmax_attn
        )

        return self.to_out(out), intermediates


class AttentionLayers(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            heads=8,
            causal=False,
            cross_attend=False,
            only_cross=False,
            use_scalenorm=False,
            use_rmsnorm=False,
            use_rezero=False,
            rel_pos_bias=False,
            rel_pos_num_buckets=32,
            rel_pos_max_distance=128,
            position_infused_attn=False,
            rotary_pos_emb=False,
            rotary_emb_dim=None,
            custom_layers=None,
            sandwich_coef=None,
            par_ratio=None,
            residual_attn=False,
            cross_residual_attn=False,
            macaron=False,
            pre_norm=True,
            gate_residual=False,
            **kwargs
    ):
        super().__init__()
        ff_kwargs, kwargs = groupby_prefix_and_trim('ff_', kwargs)
        attn_kwargs, _ = groupby_prefix_and_trim('attn_', kwargs)

        dim_head = attn_kwargs.get('dim_head', DEFAULT_DIM_HEAD)

        self.dim = dim
        self.depth = depth
        self.layers = nn.ModuleList([])

        self.has_pos_emb = position_infused_attn or rel_pos_bias or rotary_pos_emb
        self.pia_pos_emb = FixedPositionalEmbedding(dim) if position_infused_attn else None

        rotary_emb_dim = max(default(rotary_emb_dim, dim_head // 2), 32)
        self.rotary_pos_emb = RotaryEmbedding(rotary_emb_dim) if rotary_pos_emb else always(None)

        assert rel_pos_num_buckets <= rel_pos_max_distance, 'number of relative position buckets must be less than the relative position max distance'
        self.rel_pos = RelativePositionBias(causal=causal, heads=heads, num_buckets=rel_pos_num_buckets,
                                            max_distance=rel_pos_max_distance) if rel_pos_bias else None

        self.pre_norm = pre_norm

        self.residual_attn = residual_attn
        self.cross_residual_attn = cross_residual_attn

        norm_class = ScaleNorm if use_scalenorm else nn.LayerNorm
        norm_class = RMSNorm if use_rmsnorm else norm_class
        norm_fn = partial(norm_class, dim)

        norm_fn = nn.Identity if use_rezero else norm_fn
        branch_fn = Rezero if use_rezero else None

        if cross_attend and not only_cross:
            default_block = ('a', 'c', 'f')
        elif cross_attend and only_cross:
            default_block = ('c', 'f')
        else:
            default_block = ('a', 'f')

        if macaron:
            default_block = ('f',) + default_block

        if exists(custom_layers):
            layer_types = custom_layers
        elif exists(par_ratio):
            par_depth = depth * len(default_block)
            assert 1 < par_ratio <= par_depth, 'par ratio out of range'
            default_block = tuple(filter(not_equals('f'), default_block))
            par_attn = par_depth // par_ratio
            depth_cut = par_depth * 2 // 3  # 2 / 3 attention layer cutoff suggested by PAR paper
            par_width = (depth_cut + depth_cut // par_attn) // par_attn
            assert len(default_block) <= par_width, 'default block is too large for par_ratio'
            par_block = default_block + ('f',) * (par_width - len(default_block))
            par_head = par_block * par_attn
            layer_types = par_head + ('f',) * (par_depth - len(par_head))
        elif exists(sandwich_coef):
            assert sandwich_coef > 0 and sandwich_coef <= depth, 'sandwich coefficient should be less than the depth'
            layer_types = ('a',) * sandwich_coef + default_block * (depth - sandwich_coef) + ('f',) * sandwich_coef
        else:
            layer_types = default_block * depth

        self.layer_types = layer_types
        self.num_attn_layers = len(list(filter(equals('a'), layer_types)))

        for layer_type in self.layer_types:
            if layer_type == 'a':
                layer = Attention(dim, heads=heads, causal=causal, **attn_kwargs)
            elif layer_type == 'c':
                layer = Attention(dim, heads=heads, **attn_kwargs)
            elif layer_type == 'f':
                layer = FeedForward(dim, **ff_kwargs)
                layer = layer if not macaron else Scale(0.5, layer)
            else:
                raise Exception(f'invalid layer type {layer_type}')

            if isinstance(layer, Attention) and exists(branch_fn):
                layer = branch_fn(layer)

            if gate_residual:
                residual_fn = GRUGating(dim)
            else:
                residual_fn = Residual()

            self.layers.append(nn.ModuleList([
                norm_fn(),
                layer,
                residual_fn
            ]))

    def forward(
            self,
            x,
            context=None,
            mask=None,
            context_mask=None,
            mems=None,
            return_hiddens=False
    ):
        hiddens = []
        intermediates = []
        prev_attn = None
        prev_cross_attn = None

        mems = mems.copy() if exists(mems) else [None] * self.num_attn_layers

        rotary_pos_emb = self.rotary_pos_emb(x)

        for ind, (layer_type, (norm, block, residual_fn)) in enumerate(zip(self.layer_types, self.layers)):
            is_last = ind == (len(self.layers) - 1)

            if layer_type == 'a':
                hiddens.append(x)
                layer_mem = mems.pop(0)

            residual = x

            if self.pre_norm:
                x = norm(x)

            if layer_type == 'a':
                out, inter = block(x, mask=mask, sinusoidal_emb=self.pia_pos_emb, rel_pos=self.rel_pos,
                                   rotary_pos_emb=rotary_pos_emb, prev_attn=prev_attn, mem=layer_mem)
            elif layer_type == 'c':
                out, inter = block(x, context=context, mask=mask, context_mask=context_mask, prev_attn=prev_cross_attn)
            elif layer_type == 'f':
                out = block(x)

            x = residual_fn(out, residual)

            if layer_type in ('a', 'c'):
                intermediates.append(inter)

            if layer_type == 'a' and self.residual_attn:
                prev_attn = inter.pre_softmax_attn
            elif layer_type == 'c' and self.cross_residual_attn:
                prev_cross_attn = inter.pre_softmax_attn

            if not self.pre_norm and not is_last:
                x = norm(x)

        if return_hiddens:
            intermediates = LayerIntermediates(
                hiddens=hiddens,
                attn_intermediates=intermediates
            )

            return x, intermediates

        return x


class Encoder(AttentionLayers):
    def __init__(self, **kwargs):
        assert 'causal' not in kwargs, 'cannot set causality on encoder'
        super().__init__(causal=False, **kwargs)


class Decoder(AttentionLayers):
    def __init__(self, causal=True, **kwargs):
        #assert 'causal' not in kwargs, 'cannot set causality on decoder'
        if not causal: print("### WARNING: Decoder not causal. Sure? ###")
        super().__init__(causal=causal, **kwargs)


class CrossAttender(AttentionLayers):
    def __init__(self, **kwargs):
        super().__init__(cross_attend=True, only_cross=True, **kwargs)



class ConditioningChannelwiseWrapper(nn.Module):
    def __init__(self ,
                 *,
                num_tokens,
                max_seq_len,
                attn_layers,
                split_dim,
                emb_dim=None,
                max_mem_len=0.,
                emb_dropout=0.,
                num_memory_tokens=None,
                tie_embedding=False,
                use_pos_emb=True,
                use_effective_seq_len=False):
        super().__init__()
        assert isinstance(attn_layers, AttentionLayers), 'attention layers must be one of Encoder or Decoder'

        dim = attn_layers.dim
        emb_dim = default(emb_dim , dim)

        self.max_seq_len = max_seq_len
        self.max_mem_len = max_mem_len

        effective_seq_len = max(max_seq_len-split_dim, split_dim)

        self.split_dim = split_dim
        pad_value = abs(2*self.split_dim - self.max_seq_len)
        assert self.max_seq_len > self.split_dim, f'max_seq_len={self.max_seq_len} =< split_dim={self.split_dim}'


        self.token_emb = nn.Embedding(num_tokens,emb_dim)
        self.cond_emb = nn.Embedding(num_tokens,emb_dim)

        self.padder = partial(F.pad,pad=(0,0,pad_value,0),mode='constant',value=0)

        self.project_cat = nn.Linear(2*emb_dim,emb_dim)

        self.pos_emb = AbsolutePositionalEmbedding(emb_dim, effective_seq_len if use_effective_seq_len else max_seq_len) if (
                use_pos_emb and not attn_layers.has_pos_emb) else always(0)
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.project_emb = nn.Linear(emb_dim, dim) if emb_dim != dim else nn.Identity()
        self.attn_layers = attn_layers
        self.norm = nn.LayerNorm(dim)

        self.init_()

        self.to_logits = nn.Linear(dim, num_tokens) if not tie_embedding else lambda t: t @ self.token_emb.weight.t()

        # memory tokens (like [cls]) from Memory Transformers paper
        num_memory_tokens = default(num_memory_tokens, 0)
        self.num_memory_tokens = num_memory_tokens
        if num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, dim))

            # let funnel encoder know number of memory tokens, if specified
            # TODO: think of a cleaner solution
            if hasattr(attn_layers, 'num_memory_tokens'):
                attn_layers.num_memory_tokens = num_memory_tokens

    def init_(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)

    def forward(
            self,
            x,
            return_embeddings=False,
            mask=None,
            return_mems=False,
            return_attn=False,
            mems=None,
            **kwargs
    ):
        b, n, device, num_mem = *x.shape, x.device, self.num_memory_tokens
        split_dims = [self.split_dim,x.shape[1] - self.split_dim]
        c, x = torch.split(x,split_dims,1)

        x = self.token_emb(x)
        c = self.cond_emb(c)
        if x.shape[1] > c.shape[1]:
            c = self.padder(c)
        elif x.shape[1] < c.shape[1]:
            x = self.padder(x)

        x = torch.cat([x,c],dim=-1)
        x = self.project_cat(x)

        x += self.pos_emb(x)
        x = self.emb_dropout(x)

        x = self.project_emb(x)

        if num_mem > 0:
            mem = repeat(self.memory_tokens, 'n d -> b n d', b=b)
            x = torch.cat((mem, x), dim=1)

            # auto-handle masking after appending memory tokens
            if exists(mask):
                mask = F.pad(mask, (num_mem, 0), value=True)

        x, intermediates = self.attn_layers(x, mask=mask, mems=mems, return_hiddens=True, **kwargs)
        x = self.norm(x)

        mem, x = x[:, :num_mem], x[:, num_mem:]

        out = self.to_logits(x) if not return_embeddings else x

        if return_mems:
            hiddens = intermediates.hiddens
            new_mems = list(map(lambda pair: torch.cat(pair, dim=-2), zip(mems, hiddens))) if exists(mems) else hiddens
            new_mems = list(map(lambda t: t[..., -self.max_mem_len:, :].detach(), new_mems))
            return out, new_mems

        if return_attn:
            attn_maps = list(map(lambda t: t.post_softmax_attn, intermediates.attn_intermediates))
            return out, attn_maps

        return out


class TransformerWrapper(nn.Module):
    def __init__(
            self,
            *,
            num_tokens,
            max_seq_len,
            attn_layers,
            emb_dim=None,
            max_mem_len=0.,
            emb_dropout=0.,
            num_memory_tokens=None,
            tie_embedding=False,
            use_pos_emb=True
    ):
        super().__init__()
        assert isinstance(attn_layers, AttentionLayers), 'attention layers must be one of Encoder or Decoder'

        dim = attn_layers.dim
        emb_dim = default(emb_dim, dim)

        self.max_seq_len = max_seq_len
        self.max_mem_len = max_mem_len
        self.num_tokens = num_tokens

        self.token_emb = nn.Embedding(num_tokens, emb_dim)
        self.pos_emb = AbsolutePositionalEmbedding(emb_dim, max_seq_len) if (
                    use_pos_emb and not attn_layers.has_pos_emb) else always(0)
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.project_emb = nn.Linear(emb_dim, dim) if emb_dim != dim else nn.Identity()
        self.attn_layers = attn_layers
        self.norm = nn.LayerNorm(dim)

        self.init_()

        self.to_logits = nn.Linear(dim, num_tokens) if not tie_embedding else lambda t: t @ self.token_emb.weight.t()

        # memory tokens (like [cls]) from Memory Transformers paper
        num_memory_tokens = default(num_memory_tokens, 0)
        self.num_memory_tokens = num_memory_tokens
        if num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, dim))

            # let funnel encoder know number of memory tokens, if specified
            # TODO: think of a cleaner solution
            if hasattr(attn_layers, 'num_memory_tokens'):
                attn_layers.num_memory_tokens = num_memory_tokens

    def init_(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)

    def forward(
            self,
            x,
            return_embeddings=False,
            mask=None,
            return_mems=False,
            return_attn=False,
            mems=None,
            **kwargs
    ):
        b, n, device, num_mem = *x.shape, x.device, self.num_memory_tokens
        x = self.token_emb(x)
        x += self.pos_emb(x)
        x = self.emb_dropout(x)

        x = self.project_emb(x)

        if num_mem > 0:
            mem = repeat(self.memory_tokens, 'n d -> b n d', b=b)
            x = torch.cat((mem, x), dim=1)

            # auto-handle masking after appending memory tokens
            if exists(mask):
                mask = F.pad(mask, (num_mem, 0), value=True)

        x, intermediates = self.attn_layers(x, mask=mask, mems=mems, return_hiddens=True, **kwargs)
        x = self.norm(x)

        mem, x = x[:, :num_mem], x[:, num_mem:]

        out = self.to_logits(x) if not return_embeddings else x

        if return_mems:
            hiddens = intermediates.hiddens
            new_mems = list(map(lambda pair: torch.cat(pair, dim=-2), zip(mems, hiddens))) if exists(mems) else hiddens
            new_mems = list(map(lambda t: t[..., -self.max_mem_len:, :].detach(), new_mems))
            return out, new_mems

        if return_attn:
            attn_maps = list(map(lambda t: t.post_softmax_attn, intermediates.attn_intermediates))
            return out, attn_maps

        return out


class ContinuousTransformerWrapper(nn.Module):
    def __init__(
            self,
            *,
            max_seq_len,
            attn_layers,
            dim_in=None,
            dim_out=None,
            emb_dim=None,
            emb_dropout=0.,
            use_pos_emb=True
    ):
        super().__init__()
        assert isinstance(attn_layers, AttentionLayers), 'attention layers must be one of Encoder or Decoder'

        dim = attn_layers.dim

        self.max_seq_len = max_seq_len

        self.pos_emb = AbsolutePositionalEmbedding(dim, max_seq_len) if (
                    use_pos_emb and not attn_layers.has_pos_emb) else always(0)
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.project_in = nn.Linear(dim_in, dim) if exists(dim_in) else nn.Identity()

        self.attn_layers = attn_layers
        self.norm = nn.LayerNorm(dim)


        self.project_out = nn.Linear(dim, dim_out) if exists(dim_out) else nn.Identity()

    def forward(
            self,
            x,
            return_embeddings=False,
            mask=None,
            return_attn=False,
            mems=None,
            **kwargs
    ):
        b, n, _, device = *x.shape, x.device

        x = self.project_in(x)
        x += self.pos_emb(x)
        x = self.emb_dropout(x)

        x, intermediates = self.attn_layers(x, mask=mask, mems=mems, return_hiddens=True, **kwargs)
        x = self.norm(x)

        out = self.project_out(x) if not return_embeddings else x

        if return_attn:
            attn_maps = list(map(lambda t: t.post_softmax_attn, intermediates.attn_intermediates))
            return out, attn_maps

        return out


class XTransformer(nn.Module):
    def __init__(
            self,
            *,
            dim,
            tie_token_emb=False,
            wrap_decoder=True,
            causal_decoder=True,
            split_dim=None,
            use_effective_seq_len=False,
            **kwargs
    ):
        super().__init__()


        enc_kwargs, kwargs = groupby_prefix_and_trim('enc_', kwargs)
        dec_kwargs, kwargs = groupby_prefix_and_trim('dec_', kwargs)

        assert 'dim' not in enc_kwargs and 'dim' not in dec_kwargs, 'dimension of either encoder or decoder must be set with `dim` keyword'
        enc_transformer_kwargs = pick_and_pop(['num_tokens', 'max_seq_len'], enc_kwargs)
        enc_transformer_kwargs['num_memory_tokens'] = enc_kwargs.pop('num_memory_tokens', None)


        dec_transformer_kwargs = pick_and_pop(['num_tokens', 'max_seq_len'], dec_kwargs)
        if split_dim is not None:
            enc_transformer_kwargs.update({'split_dim': split_dim})
            enc_transformer_kwargs.update({'use_effective_seq_len': use_effective_seq_len})
            enc_wrapper = ConditioningChannelwiseWrapper
        else:
            enc_wrapper = TransformerWrapper

        self.encoder = enc_wrapper(
            **enc_transformer_kwargs,
            attn_layers=Encoder(dim=dim, **enc_kwargs)
        )

        self.decoder = TransformerWrapper(
            **dec_transformer_kwargs,
            attn_layers=Decoder(dim=dim, cross_attend=True, causal=causal_decoder, **dec_kwargs)
        )

        if tie_token_emb:
            self.decoder.token_emb = self.encoder.token_emb

        if wrap_decoder:
            print(f"{self.__class__.__name__}: wrapping decoder with AutoregressiveWrapper. yummy.")
            self.decoder = AutoregressiveWrapper(self.decoder)
        else:
            print(f"{self.__class__.__name__}: NOT wrapping decoder with AutoregressiveWrapper.")

    @torch.no_grad()
    def generate(self, seq_in, seq_out_start, seq_len, src_mask=None):
        encodings = self.encoder(seq_in, return_embeddings=True, mask=src_mask)
        return self.decoder.generate(seq_out_start, seq_len, context=encodings, context_mask=src_mask)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, enc=None):
        if enc is None:
            enc = self.encoder(src, mask=src_mask, return_embeddings=True)
        out = self.decoder(tgt, context=enc, mask=tgt_mask, context_mask=src_mask)
        return out, None


class ScaledXTransformer(XTransformer):
    def __init__(self, *args, **kwargs):
        self.scale_pos = kwargs.pop("scale_pos")
        self.n_scales = kwargs.pop("n_scales")
        super().__init__(*args, **kwargs)
        self.log_scalors = nn.Parameter(torch.zeros(self.n_scales))

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, enc=None):
        scale = src[:,self.scale_pos]-1 # scale comes in [1...n_scales]
        assert (0<=scale).all()
        scalors = torch.exp(self.log_scalors[scale])
        logits, loss = super().forward(src, tgt, src_mask, tgt_mask, enc=enc)
        logits = logits*scalors.reshape(logits.shape[0],1,1)
        return logits, loss


class FineScaledXTransformer(XTransformer):
    def __init__(self, *args, **kwargs):
        self.scale_pos = kwargs.pop("scale_pos")
        self.n_scales = kwargs.pop("n_scales")
        super().__init__(*args, **kwargs)
        self.log_scalors = nn.Parameter(torch.zeros(self.n_scales, self.decoder.max_seq_len))

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, enc=None):
        scale = src[:,self.scale_pos]-1 # scale comes in [1...n_scales]
        assert (0<=scale).all()
        scalors = torch.exp(self.log_scalors[scale])
        logits, loss = super().forward(src, tgt, src_mask, tgt_mask, enc=enc)
        logits = logits*scalors[:,:logits.shape[1],None]
        return logits, loss


class ResidualScaledXTransformer(XTransformer):
    def __init__(self, *args, **kwargs):
        self.scale_pos = kwargs.pop("scale_pos")
        self.n_scales = kwargs.pop("n_scales")
        self.xt_start = kwargs.pop("xt_start")
        self.xt_end = self.xt_start+kwargs.pop("xt_size")
        super().__init__(*args, **kwargs)
        self.log_scalors = nn.Parameter(torch.zeros(self.n_scales))

    def set_alphas(self, alphas):
        self.scale_alphas = alphas

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, enc=None):
        scale = src[:,self.scale_pos]-1 # scale comes in [1...n_scales]
        assert (0<=scale).all()
        scalors = torch.exp(self.log_scalors[scale])

        xt = src[:,self.xt_start:self.xt_end]
        xt = torch.nn.functional.one_hot(xt, self.decoder.num_tokens)
        alphas = self.scale_alphas[scale+1].to(device=src.device) # modeling posterior for scale=t-1
        alphas = alphas.reshape(alphas.shape[0],1,1)
        residual_logits = torch.log(alphas*xt+(1-alphas)*torch.ones_like(xt)/self.decoder.num_tokens)

        logits, loss = super().forward(src, tgt, src_mask, tgt_mask, enc=enc)
        logits = logits*scalors.reshape(logits.shape[0],1,1)
        logits = logits + residual_logits[:,:logits.shape[1],:]
        return logits, loss


class ResidualXTransformer(XTransformer):
    def __init__(self, *args, **kwargs):
        self.scale_pos = kwargs.pop("scale_pos")
        self.n_scales = kwargs.pop("n_scales")
        self.xt_start = kwargs.pop("xt_start")
        self.xt_end = self.xt_start+kwargs.pop("xt_size")
        super().__init__(*args, **kwargs)
        self.log_scalors = nn.Parameter(torch.zeros(self.n_scales))

    def set_alphas(self, alphas):
        self.scale_alphas = alphas

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, enc=None):
        scale = src[:,self.scale_pos]-1 # scale comes in [1...n_scales]
        assert (0<=scale).all()

        xt = src[:,self.xt_start:self.xt_end]
        xt = torch.nn.functional.one_hot(xt, self.decoder.num_tokens)
        alphas = self.scale_alphas[scale+1].to(device=src.device) # modeling posterior for scale=t-1
        alphas = alphas.reshape(alphas.shape[0],1,1)
        residual_logits = torch.log(alphas*xt+(1-alphas)*torch.ones_like(xt)/self.decoder.num_tokens)

        logits, loss = super().forward(src, tgt, src_mask, tgt_mask, enc=enc)
        logits = logits + residual_logits[:,:logits.shape[1],:]
        return logits, loss
