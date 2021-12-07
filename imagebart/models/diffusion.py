import os
import torch
import time
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from torch.optim.lr_scheduler import LambdaLR, ExponentialLR, CosineAnnealingLR
from einops import rearrange, repeat, reduce
from torchvision.utils import make_grid
from omegaconf import OmegaConf
from contextlib import contextmanager
from tqdm import tqdm
from collections import OrderedDict

from main import instantiate_from_config
from imagebart.modules.ema import LitEma
from imagebart.modules.betas import load_betas
from imagebart.util import log_txt_as_img
from imagebart.modules.transformer.mingpt import sample_with_past


class DenoisingTransformer(pl.LightningModule):
    """Discrete Autoregressive Denoising"""
    def __init__(self,
                 transformer_config,
                 first_stage_config,
                 n_scales,
                 ckpt_path=None,
                 ignore_keys=[],
                 first_stage_key="image",
                 scheduler_config=None,
                 monitor="val/loss",
                 top_k=None,
                 rescale_transformer_logits=False,
                 sos_token=0,
                 alpha=0.0,          # for mixing of soft and hard losses. alpha=0.0 is hard training only, alpha=1.0 soft only
                 soft_only=False,    # if true behaves like alpha=1.0 but soft loss is not rescaled by squared temperature
                 redraw_prob=None,   # if not None, use sequential forward process which randomizes entries with this probability
                 use_ema=False,
                 conditioner_config=None,
                 conditioner_key=None,
                 single_scale=None,
                 use_past_for_sampling=True,
                 ):
        super().__init__()
        self.vanilla_sampling = not use_past_for_sampling
        self.single_scale = single_scale
        self.n_scales = n_scales
        assert self.single_scale is None or (0 < self.single_scale <= n_scales)
        if monitor is not None:
            self.monitor = monitor
        self.register_buffer("sos_token", torch.tensor([sos_token]))
        self.first_stage_key = first_stage_key
        self.conditioner_key = conditioner_key
        self.conditioner_config = conditioner_config
        self.init_first_stage_from_ckpt(first_stage_config)
        self.init_cond_stage_from_ckpt(conditioner_config)
        # transformer_config['params'].update({'split_dim': conditioning_split_dim})
        self.transformer = instantiate_from_config(config=transformer_config)
        self.use_ema = use_ema
        if self.use_ema:
            self.transformer_ema = LitEma(self.transformer)
            print(f"Keeping EMAs of {len(list(self.transformer_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config
        self.top_k = top_k if top_k is not None else 100
        self.rescale_transformer_logits = rescale_transformer_logits
        assert not self.rescale_transformer_logits, "dropped support for this"
        self.soft_only = soft_only
        if self.soft_only:
            alpha = 1.0
        self.alpha = alpha
        assert 0. <= self.alpha <= 1.
        if self.single_scale == 1 and self.alpha > 0.0:
            print("######## WARNING: ######### Predicting data with soft loss, are you sure?")
        self.redraw_prob = redraw_prob
        self.set_noise_schedule()
        print(f"{self.__class__.__name__}: Running with temperature range {self.temperature_range}")
        assert self.redraw_prob is not None, "OMG!"

    def set_noise_schedule(self):
        print(f"{self.__class__.__name__}: Running with alpha = {self.alpha if not self.soft_only else 'soft_only'}")
        if self.redraw_prob is not None:
            print(f"{self.__class__.__name__}: Running with redraw_prob = {self.redraw_prob}")
            if self.redraw_prob == "geometric":  # jascha mode
                self.temperature_range = torch.cat((torch.tensor([0.0]),
                                                    1.0 / torch.arange(self.n_scales, 0, -1)))
            elif type(self.redraw_prob) == str and self.redraw_prob.startswith("alphacum:"):
                alphacum = self.redraw_prob[len("alphacum:"):].split(",")
                alphacum = [float(a) for a in alphacum]
                alpha = [at / atmo for at, atmo in zip(alphacum[1:], alphacum[:-1])]
                alpha = [1.0] + alpha
                beta = [1 - a for a in alpha]
                self.temperature_range = torch.tensor(beta)
            elif type(self.redraw_prob) == str:  # e.g. "bernoulli_FID" or "bernoulli_PSIM"
                parts = self.redraw_prob.split("_")
                assert len(parts) > 1, ("format: <dset>_<dist>_<metric>", self.redraw_prob)
                if len(parts) > 2:
                    dset, dist, metric = parts
                else:
                    dset = "in"
                    dist, metric = parts
                self.temperature_range = load_betas(f"data/{dset}_schedule_vs_metric.p",
                                                    dist, metric, self.n_scales + 1)
                self.temperature_range = torch.tensor(self.temperature_range)
            else:
                self.temperature_range = torch.cat((torch.tensor([0.0]),
                                                    self.redraw_prob * torch.ones(self.n_scales)))
            # temperature_range[t] corresponds to beta[t] of forward process
            # x[t] ~ q(x[t] | x[t-1]) = Cat(x[t] | (1-beta[t])*x[t-1] + beta[t]/K)
            # for t>0. Setting beta[0]=0 simplifies the corner case t=0

            # used to compute posterior for soft loss
            alpha = 1.0 - self.temperature_range
            alphacum = torch.cumprod(alpha, 0)
            # as buffer to place on right device
            self.init_or_change_buffer('alphats', alpha)
            self.init_or_change_buffer('alphacum', alphacum)

            print(f"{self.__class__.__name__}: Running with alphacum {self.alphacum}")
            if hasattr(self.transformer, "set_alphas"):
                self.transformer.set_alphas(alpha.detach().clone())

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.transformer_ema.store(self.transformer.parameters())
            self.transformer_ema.copy_to(self.transformer)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.transformer_ema.restore(self.transformer.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_or_change_buffer(self, key, val):
        named_buffers = [name for name, value in self.named_buffers()]
        if key in named_buffers:
            print(f'Overriding data of buffer "{key}"')
            print(f'Old value: {getattr(self, key)}')
            getattr(self, key).copy_(val)
            print(f'New value: {getattr(self, key)}')
        else:
            self.register_buffer(key, val)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in sd.keys():
            for ik in ignore_keys:
                if k.startswith(ik):
                    self.print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing keys and {len(unexpected)} unexpected keys.")
        if len(missing) > 0:
            print("Missing Keys:")
            print(missing)
        if len(unexpected) > 0:
            print("Unexpected Keys:")
            print(unexpected)

    def init_first_stage_from_ckpt(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()

    def init_cond_stage_from_ckpt(self, conditioner_config):
        self.conditioner = None
        if conditioner_config is not None:
            if conditioner_config == "__is_first_stage__":
                print("Using first stage also as cond stage.")
                self.conditioner = self.first_stage_model
            else:
                print(f"{self.__class__.__name__}: Instantiating conditioner from {conditioner_config}")
                self.conditioner = instantiate_from_config(conditioner_config)
                self.conditioner.eval()

    @torch.no_grad()
    def sample(self, x, c, steps, temperature=1.0, sample=False, top_k=None,
               callback=lambda k: None, embeddings=None, **kwargs):
        guide = kwargs.get("guide", None)
        x = torch.cat((c, x), dim=1)
        if self.vanilla_sampling:
            block_size = self.transformer.get_block_size()
            assert not self.transformer.training
            for k in range(steps):
                callback(k)
                assert x.size(1) <= block_size  # make sure model can see conditioning
                x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed
                logits, _ = self.transformer(x_cond, embeddings=embeddings)
                if guide is not None:
                    logits = logits + guide[:, :logits.shape[1]]
                # pluck the logits at the final step and scale by temperature
                logits = logits[:, -1, :] / temperature
                # optionally crop probabilities to only the top k options
                if top_k is not None:
                    logits = self.top_k_logits(logits, top_k)
                # apply softmax to convert to probabilities
                probs = F.softmax(logits, dim=-1)
                # sample from the distribution or take the most likely
                if sample:
                    ix = torch.multinomial(probs, num_samples=1)
                else:
                    _, ix = torch.topk(probs, k=1, dim=-1)
                # append to the sequence and continue
                x = torch.cat((x, ix), dim=1)
            # cut off conditioning
            x = x[:, c.shape[1]:]
        else:
            assert embeddings is None
            x = sample_with_past(x, self.transformer, steps, temperature=temperature, sample_logits=sample, top_k=top_k,
                                 callback=callback, guide=guide)
        return x

    @torch.no_grad()
    def decode_to_img(self, index, zshape):
        bhwc = (zshape[0], zshape[2], zshape[3], zshape[1])
        quant_z = self.first_stage_model.quantize.get_codebook_entry(
            index.reshape(-1), shape=bhwc)
        x = self.first_stage_model.decode(quant_z)
        return x

    @torch.no_grad()
    def sample_single_scale(self, c_indices, scale, temp_x, steps,
                            temperature=None, callback=None, top_k=None,
                            cond=None, guide=None):
        assert temp_x is None, "dropped support for this"
        # sample
        temp = 1. if temperature is None else temperature
        z_start_indices = c_indices[:, :0]  # empty
        t1 = time.time()
        if cond is not None:
            c_indices = torch.cat((cond, c_indices), dim=1)
        c_indices = torch.cat((scale, c_indices), dim=1)
        index_sample = self.sample(z_start_indices,
                                   c_indices,
                                   steps=steps,
                                   temperature=temp,
                                   sample=True,
                                   top_k=top_k if top_k is not None else 100,
                                   callback=callback if callback is not None else lambda k: None,
                                   guide=None,
                                   )
        if not hasattr(self, "sampling_time"):
            self.sampling_time = time.time() - t1
            print(f"Single scale sampling takes about {self.sampling_time:.2f} seconds.")

        return index_sample

    def get_diffusion_row(self, diffusion_h, scales):
        zq, _, info_x = self.first_stage_model.quantize(diffusion_h)
        indices = info_x[2]
        allindices = [indices]
        assert self.temperature_range[0].item() == 0.0, "Should I flip them?"

        bcshape = list(indices.shape)
        bcshape[0] = 1  # same noise for all
        for redraw_prob in self.temperature_range[1:]:
            randindices = torch.randint(0,
                                        self.first_stage_model.quantize.re_embed,
                                        bcshape,
                                        device=indices.device)
            redraw = torch.bernoulli(redraw_prob * torch.ones(bcshape)).to(
                device=indices.device)
            indices = (1 - redraw) * indices + redraw * randindices
            allindices.append(indices.to(dtype=torch.int64))

        allindices = torch.stack(allindices)
        assert len(allindices.shape) == 4, allindices.shape
        scale = scales[None, :, None, None].expand(
            -1, -1, allindices.shape[2], allindices.shape[3])
        indices_x = torch.gather(allindices, 0, scale)[0]

        x = rearrange(indices_x, 'b h w -> b (h w)')
        indices = x
        qshape = zq.shape
        return qshape, indices

    def get_input(self, batch, key):
        x = batch[key]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x

    @torch.no_grad()
    def get_h(self, x):
        h = self.first_stage_model.encode_to_prequant(x)
        return h

    @torch.no_grad()
    def get_conditioning(self, batch):
        if self.conditioner_key is not None:
            c = self.get_input(batch, self.conditioner_key)
            if c.device != self.conditioner.device:
                c = c.to(device=self.conditioner.device)
            quant_c, _, info = self.conditioner.encode(c)
            cond = info[2].view(quant_c.shape[0], -1)
        else:
            cond = self.conditioner(batch)
        return cond

    @torch.no_grad()
    def get_scales(self, pre_quant, force_random=False):
        if self.single_scale is None:
            if force_random:
                scale = torch.randint(1, self.temperature_range.shape[0], size=(pre_quant.shape[0],)).to(
                    device=pre_quant.device)
            else:
                scale = torch.randint(1, self.temperature_range.shape[0], size=(1,)).to(device=pre_quant.device)
                self.current_scale = scale.item()
                scale = repeat(scale, '1 -> b', b=pre_quant.shape[0])
        else:
            scale = (self.single_scale * torch.ones(size=(pre_quant.shape[0],))).to(dtype=torch.long,
                                                                                    device=pre_quant.device)

        zq_x, _, info_x = self.first_stage_model.quantize(pre_quant)
        indices = info_x[2]
        allindices = [indices]
        assert self.temperature_range[0].item() == 0.0, "Should I flip them?"

        for redraw_prob in self.temperature_range[1:]:
            randindices = torch.randint(0,
                                        self.first_stage_model.quantize.re_embed,
                                        indices.shape,
                                        device=indices.device)
            redraw = torch.bernoulli(redraw_prob * torch.ones(indices.shape)).to(
                device=indices.device)
            indices = (1 - redraw) * indices + redraw * randindices
            allindices.append(indices.to(dtype=torch.int64))

        allindices = torch.stack(allindices)
        assert len(allindices.shape) == 4, allindices.shape
        scale = repeat(scale, 'b -> 1 b h w',
                       b=allindices.shape[1], h=allindices.shape[2], w=allindices.shape[3])
        indices_x = torch.gather(allindices, 0, scale - 1)[0]
        indices_c = torch.gather(allindices, 0, scale)[0]

        x = rearrange(indices_x, 'b h w -> b (h w)')
        c = rearrange(indices_c, 'b h w -> b (h w)')

        qshape = zq_x.permute(0, 2, 3, 1).shape
        zq_x = self.first_stage_model.quantize.get_codebook_entry(x.reshape(-1),
                                                                  qshape)
        zq_c = self.first_stage_model.quantize.get_codebook_entry(c.reshape(-1),
                                                                  qshape)

        scale = scale[0, :, 0, 0]  # b

        # posterior
        # scale is t i.e. target is x_{t-1}
        # compute probabilities of posterior q(x_{t-1} | x_t, x_0)
        K = self.first_stage_model.quantize.re_embed
        onehot_xt = rearrange(torch.nn.functional.one_hot(indices_c,
                                                          num_classes=K),
                              "b h w n -> b n h w", n=K)
        onehot_x0 = rearrange(torch.nn.functional.one_hot(allindices[0, ...],
                                                          num_classes=K),
                              "b h w n -> b n h w", n=K)
        alphat = rearrange(torch.gather(self.alphats, 0, scale),
                           "b -> b 1 1 1")
        alphacumtmo = rearrange(torch.gather(self.alphacum, 0, scale - 1),
                                "b -> b 1 1 1")
        posterior = (alphacumtmo * onehot_x0 + (1 - alphacumtmo) / K) * (alphat * onehot_xt + (1 - alphat) / K)
        # normalize
        alphacumt = rearrange(torch.gather(self.alphacum, 0, scale),
                              "b -> b 1 1 1")
        Z = reduce(posterior, "b n h w -> b 1 h w", "sum", n=K)
        # better safe than sorry
        Z_alt = reduce((alphacumt * onehot_x0 + (1 - alphacumt) / K) * onehot_xt,
                       "b n h w -> b 1 h w",
                       "sum", n=K)
        assert Z.allclose(Z_alt)
        posterior = posterior / Z
        assert torch.allclose(reduce(posterior, "b n h w -> b 1 h w", "sum", n=K),
                              torch.ones_like(posterior))

        return zq_x, zq_c, x, c, scale[:, None], None, posterior

    def compute_loss(self, logits, targets, target_logits, temperature, recp_loss_weights=None, split="train"):
        if recp_loss_weights is None:
            recp_loss_weights = torch.ones_like(temperature.squeeze(-1).squeeze(-1))

        if len(target_logits.shape) == 4:
            target_logits = rearrange(target_logits, 'b n h w -> b (h w) n')
        assert logits.shape == target_logits.shape
        # loss_hard = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)).mean()

        hard_logits = rearrange(logits, 'b n c -> b c n')
        loss_hard = F.cross_entropy(hard_logits, targets, reduction='none')
        loss_hard = loss_hard.mean(-1)
        loss_hard = (loss_hard / recp_loss_weights).mean()

        # target logits are actually posterior probabilities
        target_probs = target_logits
        loss_soft = F.kl_div(F.log_softmax(logits, dim=-1),
                             target_probs, reduction="none").sum(-1).mean(-1)
        # teacher is posterior, student is our model
        loss = (loss_soft / recp_loss_weights).mean() * self.alpha + (1. - self.alpha) * loss_hard

        return loss, {f"{split}/loss": loss.detach(),
                      f"{split}/loss_hard": loss_hard.detach(),
                      f"{split}/loss_soft": loss_soft.mean().detach()}

    def forward(self, x, c, scale, temp=1.):
        # learn p(x|c, tau)
        z_indices = x
        c_indices = c
        cz_indices = torch.cat((scale, c_indices, z_indices), dim=1)
        target = z_indices
        # make the prediction
        logits, _ = self.transformer(cz_indices[:, :-1])
        logits = logits / temp
        # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c)
        logits = logits[:, c_indices.shape[1]:]
        return logits, target

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out

    def shared_step(self, batch, batch_idx):
        x = self.get_input(batch, self.first_stage_key)
        pre_quant = self.get_h(x)
        zq_x, zq_c, x, c, scale, _, logits_target = self.get_scales(pre_quant)
        temp = 1.
        if self.conditioner is not None:
            cond = self.get_conditioning(batch).to(x.device)
            c = torch.cat((cond, c), 1)
        logits, target = self(x, c, scale, temp=temp)
        dummy_temp = torch.ones(1, 1, 1, 1).to(device=x.device)
        return logits, target, logits_target, dummy_temp

    def training_step(self, batch, batch_idx):
        logits, target, logits_target, temp_target = self.shared_step(batch, batch_idx)
        loss, log_dict = self.compute_loss(logits, target, logits_target, temp_target.squeeze(-1), split="train")

        self.log("train/loss", loss,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)
        if hasattr(self.transformer, "log_scalors"):
            if len(self.transformer.log_scalors.shape) > 1:
                for k in range(self.transformer.log_scalors.shape[0]):
                    for j in [0, -1]:
                        self.log(f"train/log_scalors_{k}_{j}", self.transformer.log_scalors[k, j],
                                 prog_bar=False, on_step=True, on_epoch=False,
                                 logger=True)
            else:
                for k in range(self.transformer.log_scalors.shape[0]):
                    self.log(f"train/log_scalors_{k}", self.transformer.log_scalors[k],
                             prog_bar=False, on_step=True, on_epoch=False,
                             logger=True)
        return loss

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.transformer_ema(self.transformer)

    def validation_step(self, batch, batch_idx):
        logits, target, logits_target, temp_target = self.shared_step(batch, batch_idx)
        loss, log_dict = self.compute_loss(logits, target, logits_target, temp_target.squeeze(-1), split="val")
        for k in log_dict:
            self.log(k, log_dict[k], prog_bar=False, logger=True, on_step=False, on_epoch=True)
        with self.ema_scope():
            logits, target, logits_target, temp_target = self.shared_step(batch, batch_idx)
            loss, log_dict = self.compute_loss(logits, target, logits_target, temp_target.squeeze(-1),
                                               split="val_ema")
            for k in log_dict:
                self.log(k, log_dict[k], prog_bar=False, logger=True, on_step=False, on_epoch=True)

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        if self.use_ema:
            self.transformer_ema(self.transformer)

    @torch.no_grad()
    def log_images(self,
                   batch,
                   temperature=None,
                   top_k=None,
                   callback=None,
                   N=4,
                   sample=True,
                   gridify=True,
                   start_idx=None,
                   log_diffusion=True,
                   log_scale_step=1,
                   log_caption=False,
                   sample_full=False,
                   sample_half=False,
                   **kwargs):
        with self.ema_scope(context="Plotting"):
            top_k = top_k if top_k is not None else self.top_k
            log = dict()
            x = self.get_input(batch, self.first_stage_key).to(self.device)[:N, ...]
            cond = None
            if self.conditioner is not None:
                cond = self.get_conditioning(batch).to(self.device)[:N, ...]
            pre_quant = self.get_h(x)
            quant_z, quant_c, z_indices, c_indices, scale, _, logits_x = self.get_scales(pre_quant, force_random=True)

            if sample:
                # sample
                index_sample = self.sample_single_scale(c_indices, scale, None,
                                                        steps=z_indices.shape[1],
                                                        temperature=temperature,
                                                        callback=callback,
                                                        top_k=top_k,
                                                        cond=cond
                                                        )
                x_sample_nopix = self.decode_to_img(index_sample, quant_z.shape)
                log["samples_t"] = x_sample_nopix

            if sample_full:
                # only batch size 1
                Nb = 1
                t1 = time.time()
                scale_samples = list()
                # get first scale: (uniformly distributed)
                _, _, info_c = self.first_stage_model.quantize(pre_quant[:Nb])
                c_scale_indices = rearrange(info_c[2], 'b h w -> b (h w)')

                # replace by uniform sample and go
                c_scale_indices = torch.randint(0,
                                                self.first_stage_model.quantize.re_embed,
                                                c_scale_indices.shape,
                                                device=c_scale_indices.device)
                scale_samples.append(self.decode_to_img(c_scale_indices,
                                                        [Nb, quant_z.shape[1], quant_z.shape[2], quant_z.shape[3]]))
                current_scale = ((len(self.temperature_range) - 1) * torch.ones(1, 1)).to(c_scale_indices)
                for i in range(len(self.temperature_range) - 2, -1, -1):
                    c_scale_indices = self.sample_single_scale(c_scale_indices,
                                                               current_scale,
                                                               None,
                                                               steps=z_indices.shape[1],
                                                               temperature=temperature,
                                                               top_k=top_k,
                                                               cond=cond[[0], ...] if cond is not None else None
                                                               )
                    scale_samples.append(self.decode_to_img(c_scale_indices,
                                                            [Nb, quant_z.shape[1], quant_z.shape[2], quant_z.shape[3]]
                                                            )
                                         )
                    current_scale = (i * torch.ones(1, 1)).to(c_scale_indices)

                if not hasattr(self, "full_sampling_time"):
                    self.full_sampling_time = time.time() - t1
                    print(f"Full range sampling takes about {self.full_sampling_time:.2f} seconds.")
                # make a grid
                scale_samples = scale_samples[::log_scale_step]
                scale_samples = torch.cat(scale_samples, dim=0)
                if gridify:
                    grid = make_grid(scale_samples, nrow=scale_samples.shape[0])
                else:
                    grid = scale_samples
                log["denoise_row"] = grid

            # reconstructions
            x_rec_x = self.decode_to_img(z_indices, quant_z.shape)
            x_rec_c = self.decode_to_img(c_indices, quant_z.shape)

            # diffusion row
            if log_diffusion:
                n_scales = self.temperature_range.shape[0]
                diffusion_h = pre_quant[n_scales * [0], ...]
                scales = torch.flip(torch.arange(n_scales), dims=(0,)).to(device=diffusion_h.device)
                qshape, diffusion_indices = self.get_diffusion_row(diffusion_h, scales)

                tmpshape = list(qshape)
                tmpshape[0] = diffusion_indices[::log_scale_step].shape[0]
                diffusion_rec = self.decode_to_img(diffusion_indices[::log_scale_step],
                                                   tmpshape)
                if gridify:
                    grid = make_grid(diffusion_rec, nrow=diffusion_rec.shape[0])
                else:
                    grid = diffusion_rec
                log["diffusion_row"] = grid

                if sample_half and n_scales > 2:
                    t2 = time.time()
                    if start_idx is None:
                        start_idx = n_scales // 2
                    assert start_idx > 0
                    tmpshape = list(qshape)
                    tmpshape[0] = diffusion_indices[:start_idx, ...][::log_scale_step].shape[0]
                    pre_scale_samples = self.decode_to_img(diffusion_indices[:start_idx, ...][::log_scale_step],
                                                           tmpshape)
                    scale_samples = list()
                    c_scale_indices = diffusion_indices[[start_idx - 1], ...]
                    quantshape = list(qshape)
                    quantshape[0] = 1

                    for i in range(n_scales - start_idx):
                        c_scale_indices = self.sample_single_scale(c_scale_indices,
                                                                   scales[[start_idx - 1 + i], None].to(
                                                                       device=c_scale_indices.device),
                                                                   None,
                                                                   steps=z_indices.shape[1],
                                                                   temperature=temperature,
                                                                   top_k=top_k,
                                                                   cond=cond[[0], ...] if cond is not None else None)
                        scale_samples.append(self.decode_to_img(c_scale_indices, quantshape))

                    scale_samples = torch.cat(scale_samples[::log_scale_step], 0)
                    scale_samples = torch.cat((pre_scale_samples, scale_samples), 0)
                    if gridify:
                        grid = make_grid(scale_samples, nrow=scale_samples.shape[0])
                    else:
                        grid = scale_samples
                    log["denoise_uprow"] = grid

                    if not hasattr(self, "half_sampling_time"):
                        self.half_sampling_time = time.time() - t2
                        print(f"Half range sampling takes about {self.half_sampling_time:.2f} seconds.")

            log["inputs"] = x
            log["reconstructions_t"] = x_rec_x
            log["reconstructions_t+1"] = x_rec_c

            if self.conditioner_config == "__is_first_stage__":
                c_inp = self.get_input(batch, self.conditioner_key).to(self.device)[:cond.shape[0], ...]
                log["inputs_cond"] = c_inp
                c_rec = self.decode_to_img(cond, quant_z.shape)
                log["reconstructions_cond"] = c_rec

            if log_caption:
                cap_as_img = log_txt_as_img((x.shape[2], x.shape[3]), batch["caption"])
                log["caption"] = cap_as_img

            elif self.conditioner_key == "segmentation":
                # get image from segmentation mask
                segmap = self.conditioner.get_input(batch, self.conditioner_key)[:N].to(self.device)
                num_classes = segmap.shape[1]

                c = torch.argmax(segmap, dim=1, keepdim=True)
                c = F.one_hot(c, num_classes=num_classes)
                c = c.squeeze(1).permute(0, 3, 1, 2).float()
                c = self.conditioner.to_rgb(c)
                log["segmentation"] = c

            return log

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.transformer.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95))
        if self.use_scheduler:
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(optimizer, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [optimizer], scheduler
        return optimizer


class DecoderOnlyDenoiser(DenoisingTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.single_scale is not None, "I am a single scaler"
        assert self.single_scale == len(self.temperature_range) - 1, f"oh no, " \
                                                                     f"{self.single_scale} =/= {len(self.temperature_range) - 1}"

    def forward(self, z_indices, c_indices, temp=1.):
        # learn p(x|c, tau)
        # c_indices holds scale + optionally conditoning
        target = z_indices
        cz_indices = torch.cat((c_indices, z_indices), dim=1)

        # make the prediction
        logits, _ = self.transformer(cz_indices[:, :-1])
        logits = logits / temp
        # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c)
        logits = logits[:, c_indices.shape[1] - 1:]
        return logits, target

    def shared_step(self, batch, batch_idx):
        x = self.get_input(batch, self.first_stage_key)
        pre_quant = self.get_h(x)
        zq_x, zq_c, x, c, scale, _, logits_target = self.get_scales(pre_quant)
        temp = 1.
        c_indices = scale
        if self.conditioner is not None:
            cond = self.get_conditioning(batch).to(x.device)
            c_indices = torch.cat((c_indices, cond), 1)

        logits, target = self(x, c_indices, temp=temp)
        dummy_temp = torch.ones(1, 1, 1, 1).to(device=x.device)
        return logits, target, logits_target, dummy_temp

    @torch.no_grad()
    def sample_single_scale(self, c_indices, scale, temp_x, steps,
                            temperature=None, callback=None, top_k=None,
                            cond=None, guide=None):
        assert temp_x is None, "dropped support for this"

        # sample. c_indices will be ignored since they are pure uniform noise
        temp = 1. if temperature is None else temperature

        z_start_indices = c_indices[:, :0]  # empty
        t1 = time.time()
        c_indices = scale
        if cond is not None:
            c_indices = torch.cat((c_indices, cond), dim=1)
        index_sample = self.sample(z_start_indices,
                                   c_indices,
                                   steps=steps,
                                   temperature=temp,
                                   sample=True,
                                   top_k=top_k if top_k is not None else 100,
                                   callback=callback if callback is not None else lambda k: None,
                                   guide=guide,
                                   )
        if not hasattr(self, "sampling_time"):
            self.sampling_time = time.time() - t1
            print(f"Single scale sampling takes about {self.sampling_time:.2f} seconds.")
        return index_sample


class DenoisingXTransformer(DenoisingTransformer):
    """a.k.a BART (https://arxiv.org/abs/1910.13461). Also supports regression mode."""

    def __init__(self, *args, **kwargs):
        regression = kwargs.pop("regression", False)
        super().__init__(*args, **kwargs)
        self.regression = regression
        if self.regression: print(f"{self.__class__.__name__} running in pure regression mode.")

    def forward(self, x, c, scale, temp=1.):
        # learn p(x|c, tau)
        target = x
        sos = repeat(self.sos_token, '... -> b (...)', b=x.shape[0]).to(x)
        if not self.regression:
            z_indices = torch.cat((sos, x), 1)[:, :-1]
        else:
            z_indices = torch.cat((sos, torch.zeros_like(x)), 1)[:, :-1]
        c_indices = torch.cat((scale, c), dim=1)
        # make the prediction
        logits, _ = self.transformer(c_indices, z_indices)
        logits = logits / temp
        return logits, target

    @torch.no_grad()
    def sample(self, x, c, steps, temperature=1.0, sample=False, top_k=None,
               callback=lambda k: None, embeddings=None, **kwargs):
        assert not self.transformer.training
        guide = kwargs.get("guide", None)
        if not self.regression:
            enc = self.transformer.encoder(c, mask=None, return_embeddings=True)
            for k in range(steps):
                callback(k)
                # logits = self.transformer.decoder(x, context=enc, mask=None, context_mask=None)
                logits, _ = self.transformer(c, x, enc=enc)
                if guide is not None:
                    logits = logits + guide[:, :logits.shape[1]]
                # pluck the logits at the final step and scale by temperature
                logits = logits[:, -1, :] / temperature
                # optionally crop probabilities to only the top k options
                if top_k is not None:
                    logits = self.top_k_logits(logits, top_k)
                # apply softmax to convert to probabilities
                probs = F.softmax(logits, dim=-1)
                # sample from the distribution or take the most likely
                if sample:
                    ix = torch.multinomial(probs, num_samples=1)
                else:
                    _, ix = torch.topk(probs, k=1, dim=-1)
                # append to the sequence and continue
                x = torch.cat((x, ix), dim=1)
            x = x[:, 1:]  # cut of the sos token
        else:
            # x is cat(sos, zeros)
            logits, _ = self.transformer(c, x[:, :-1])
            if guide is not None:
                logits = logits + guide[:, :logits.shape[1]]
            logits = logits / temperature
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution or take the most likely
            if sample:
                shape = probs.shape
                probs = probs.reshape(shape[0] * shape[1], shape[2])
                ix = torch.multinomial(probs, num_samples=1)
                x = ix.reshape(shape[0], shape[1])
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
                x = ix.squeeze(-1)
        return x

    @torch.no_grad()
    def sample_single_scale(self, c_indices, scale, temp_x, steps,
                            temperature=None, callback=None, top_k=None,
                            cond=None, guide=None):
        assert temp_x is None, "dropped support for this"
        # sample
        temp = 1.
        sos = repeat(self.sos_token, '... -> b (...)', b=c_indices.shape[0]).to(c_indices)
        z_start_indices = sos if not self.regression else torch.cat((sos, torch.zeros_like(c_indices)), 1)
        t1 = time.time()
        if cond is not None:
            c_indices = torch.cat((cond, c_indices), 1)
        index_sample = self.sample(z_start_indices,
                                   torch.cat((scale, c_indices), dim=1),
                                   steps=steps,
                                   temperature=temp,
                                   sample=True,
                                   top_k=top_k if top_k is not None else 100,
                                   callback=callback if callback is not None else lambda k: None,
                                   guide=guide,
                                   )
        if not hasattr(self, "sampling_time"):
            self.sampling_time = time.time() - t1
            print(f"Single scale sampling takes about {self.sampling_time:.2f} seconds.")
        return index_sample

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.transformer.parameters(), lr=self.learning_rate, betas=(0.9, 0.95))
        if self.use_scheduler:
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(optimizer, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [optimizer], scheduler
        return optimizer


class ImageBARTSampler(torch.nn.Module):
    def __init__(self, paths, eval=True, topk: list = None, skip_ema_saving=False, prune_ckpt=True):
        super().__init__()
        self.is_pruned = prune_ckpt
        if prune_ckpt:
            paths = self.save_ema_weights(paths, eval, skip_ema_saving)
        else:
            print('Not pruning ckpt')
        models, configs = self.__load_models(paths, eval)
        self.configs = list(configs)
        self.models = torch.nn.ModuleList(models)
        self.codebook_size = self.models[-1].first_stage_model.quantize.re_embed
        self.is_conditional = 'conditioner_config' in self.configs[-1]['model']['params']

        if self.is_conditional:
            print(f'Sampling from conditional model.')
        if topk is not None:
            assert isinstance(topk, list) and len(topk) == len(
                self.models), 'topk has tp be a list of equal length as those of the models'
            self.topk = list(reversed(topk))
        else:
            self.topk = [None] * len(self.models)

        self.models[-1].first_stage_model.encoder = None

        self.z_dim = self.configs[-1]['model']['params']['first_stage_config']['params']['ddconfig']['z_channels']
        self.embed_dim = self.configs[-1]['model']['params']['first_stage_config']['params']['embed_dim']

        print(
            f'Loaded {len(self.models)}-scales model with topk values of {self.topk} (when counted from last to first scale).')

    def save_ema_weights(self, paths, eval, skip_saving):
        for i, (ckpt_path, config_path) in enumerate(tqdm(zip(paths['checkpoints'], paths['configs']),
                                                          desc='Saving EMA weights for all models.')):
            save_path = ckpt_path.split('.')[0] + '-ema_weights.ckpt'
            if skip_saving:
                assert os.path.isfile(
                    save_path), 'Please make sure that ema weights exist in same folder than complete weights'
            else:
                model, *_ = self.__load_model(ckpt_path, config_path, eval, True, load_global_step=True, zero_ema=False)
                if model.use_ema:
                    model.transformer_ema.copy_to(model.transformer)

                saved_sd = OrderedDict(
                    [(key, model.state_dict()[key]) for key in model.state_dict() if 'transformer_ema' not in key])
                torch.save({'state_dict': saved_sd}, save_path)
            # switch checkpoint path
            paths['checkpoints'][i] = save_path

        return paths

    def __load_model(self, ckpt_path, config_path, eval, init_first_stage, load_global_step=False, zero_ema=True):
        print(f"loading config from {config_path} and model from {ckpt_path}")
        config = OmegaConf.load(config_path)
        pl_sd = torch.load(ckpt_path, map_location="cpu")
        if load_global_step:
            global_step = pl_sd["global_step"]
            print(f"loaded model from global step {global_step}")

        model = self.__load_model_from_config(config, pl_sd["state_dict"], eval_mode=eval,
                                              init_first_stage=init_first_stage)
        if zero_ema:
            model.transformer_ema = None
        if load_global_step:
            return model, config, global_step
        else:
            return model, config

    def __load_models(self, paths, eval):
        models = list()
        configs = list()
        global_steps = list()
        for model_index, (ckpt_path, config_path) in enumerate(zip(paths["checkpoints"], paths["configs"])):
            model, config = self.__load_model(ckpt_path, config_path, eval, init_first_stage=model_index == 0,
                                              zero_ema=self.is_pruned)

            models.append(model)
            configs.append(config)

        return reversed(models), reversed(configs)

    def __load_model_from_config(self, config, sd, eval_mode, init_first_stage=False):
        print("config:")
        print(config.pretty())
        if not init_first_stage:
            print(f'Not initializing first stage model')
            config['model']['params']['first_stage_config']['target'] = 'braket.models.vqgan.VQSamplingDummyModel'
        model = instantiate_from_config(config["model"])
        if sd is not None:
            m, u = model.load_state_dict(sd, strict=False)
            if len(m) > 0:
                print(len(m))
                for mm in m:
                    if not mm.startswith("transformer_ema"): assert False
        if eval_mode:
            model.eval()
        return model

    @torch.no_grad()
    def sample(self, batch, decode_to_img=True, mask_upper_half=False):
        batch_size = batch[self.models[-1].first_stage_key].shape[0]

        index_shape = (batch_size, self.embed_dim)
        c_scale_indices = torch.randint(0, self.codebook_size, index_shape, device='cuda')
        for model, top_k in zip(self.models, self.topk):
            if mask_upper_half:
                # todo add masking
                pass

            current_scale = (model.single_scale) * torch.ones((batch_size, 1), device='cuda',
                                                              dtype=c_scale_indices.dtype)
            if self.is_conditional:
                cond = model.get_conditioning(batch).cuda()
            else:
                cond = None
            if self.is_pruned:
                c_scale_indices = model.sample_single_scale(c_scale_indices,
                                                            current_scale,
                                                            temp_x=None,
                                                            steps=self.embed_dim,
                                                            cond=cond,
                                                            top_k=top_k)
            else:
                with model.ema_scope(context='Sampling'):
                    c_scale_indices = model.sample_single_scale(c_scale_indices,
                                                                current_scale,
                                                                temp_x=None,
                                                                steps=self.embed_dim,
                                                                cond=cond,
                                                                top_k=top_k)

        decode_shape = [batch_size, self.z_dim, int(np.sqrt(self.embed_dim)), int(np.sqrt(self.embed_dim))]
        if decode_to_img:
            out = self.models[-1].decode_to_img(c_scale_indices, decode_shape)
        else:
            out = c_scale_indices

        return out
