import os, sys
import torch
import numpy as np
from omegaconf import OmegaConf
import streamlit as st
from streamlit import caching
from PIL import Image
from main import instantiate_from_config
from torch.utils.data.dataloader import default_collate
from torchvision.utils import make_grid
from tqdm import trange
from einops import repeat
from contextlib import contextmanager


from scripts.sample_imagebart import get_top_k_schedule, get_temperature_schedule


@torch.no_grad()
def sample_unconditional(models, batch_size, chain_schedule, temperature_schedule=None, top_k_schedule=None,
                         dim_z=256, h=16, w=16,
                         example=None, mask=None):
    ########### opts
    use_ema = True
    device = torch.device("cuda")
    model_index = len(models) - 1
    steps = len(models) * [h * w]
    #################

    # mask and input
    mask = mask.to(device=device)
    unmasked_input = torch.tensor(example["image"]).permute(0, 3, 1, 2).to(device=device)
    masked_inputs = mask * unmasked_input

    ##### clean for guidance
    with on_gpu(models[0]):
        pre_quant = models[0].get_h(unmasked_input)
        _, _, x_clean, _, _, _, _ = models[0].get_scales(pre_quant=pre_quant)

    model = models[model_index]
    c = torch.randint(0,
                      model.first_stage_model.quantize.re_embed,
                      x_clean.shape,
                      device=x_clean.device)

    mask = torch.nn.functional.interpolate(mask, size=(16, 16),
                                           mode="nearest")
    mask = mask.reshape(c.shape).to(device=c.device)
    orig_dtype = c.dtype
    c = (1 - mask) * c + mask * x_clean
    c = c.to(dtype=orig_dtype)

    guide = torch.nn.functional.one_hot(c, num_classes=model.first_stage_model.quantize.re_embed).to(
        dtype=torch.float32)
    guide = torch.log(guide)
    guide[mask < 0.5] = 0

    # start sampling
    c_scale_indices = c
    scale = model_index
    current_scale = (scale * torch.ones(batch_size, 1)).to(device).long()
    steppys = st.empty()
    cb = lambda x: steppys.write(f"{x}/{h * w}")
    scaleinfo = st.empty()
    n_scales = len(models)
    for scale_n, model in enumerate(models[:model_index + 1][::-1]):
        temperature = temperature_schedule[scale]
        top_k = top_k_schedule[scale]
        n_chains = chain_schedule[scale]
        with on_gpu(model):
            with ema_scope(model, active=use_ema):
                for chain_idx in range(n_chains):
                    scaleinfo.write(
                        f"sampling chain {chain_idx + 1}/{n_chains} for scale {n_scales - scale_n}/{n_scales}, "
                        f"temp = {temperature:.2f}, top-k = {top_k}")

                    chain_weight = 1 - chain_idx / n_chains
                    if chain_idx > 0:
                        # already reversed, run forward again
                        origdtype = c_scale_indices.dtype
                        randindices = torch.randint(
                            0,
                            model.first_stage_model.quantize.re_embed,
                            c_scale_indices.shape,
                            device=c_scale_indices.device)
                        redraw_prob = chain_weight * model.temperature_range[model.single_scale]
                        redraw = torch.bernoulli(
                            redraw_prob * torch.ones(c_scale_indices.shape)).to(
                            device=c_scale_indices.device)
                        c_scale_indices = (1 - redraw) * c_scale_indices + redraw * randindices
                        c_scale_indices = c_scale_indices.to(dtype=origdtype)
                    c_scale_indices = model.sample_single_scale(c_scale_indices,
                                                                current_scale + 1,
                                                                temp_x=None,
                                                                steps=steps[scale],
                                                                temperature=temperature,
                                                                top_k=top_k,
                                                                guide=guide,
                                                                callback=cb
                                                                )
        scale -= 1
        current_scale = (scale * torch.ones(batch_size, 1)).to(device).long()

    qzshape = [batch_size, dim_z, h, w]
    with on_gpu(model):
        sample = model.decode_to_img(c_scale_indices, qzshape)

    log = dict()
    log["samples"] = sample
    log["inputs"] = unmasked_input
    log["masked_inputs"] = masked_inputs
    return log


def generate_mask(masking_option, shape):
    bs, h, w, c = shape
    mask = np.array(Image.new('L', (h, w))).astype(np.bool)

    if masking_option == 'upper-half completion':
        mask[h // 2:] = np.logical_not(mask[h // 2:])

    elif masking_option == 'window-inpainting':
        window_size = st.number_input(f'Select size of quadratic window for {masking_option} '
                                      f'(note: divided by 16 in latent space)', min_value=10,
                                      max_value=h // 2, value=h // 4)
        mask = np.logical_not(mask)
        mask[
        (h - window_size) // 2:(h + window_size) // 2,
        (w - window_size) // 2:(w + window_size) // 2] = np.logical_not(
            mask[(h - window_size) // 2:(h + window_size) // 2,
            (w - window_size) // 2:(w + window_size) // 2])

    else:
        window_size = st.number_input(f'Select size of quadratic window for {masking_option} '
                                      f'(note: divided by 16 in latent space)', min_value=h // 2,
                                      max_value=h - 20, value=h // 2)
        mask[
        (h - window_size) // 2:(h + window_size) // 2,
        (w - window_size) // 2:(w + window_size) // 2] = np.logical_not(
            mask[(h - window_size) // 2:(h + window_size) // 2,
            (w - window_size) // 2:(w + window_size) // 2])
        st.warning('With outpainting enabled, you might have to increase the length of the chains')

    display_mask = mask
    # only for displaying reasons incase of inpainting and upper half completion
    for p in [0, h - 1]:
        display_mask[p] = False
        display_mask[:, p] = False
    st.image((255 * mask.astype(np.uint8)), f'Selected mask for {masking_option}')

    mask = torch.from_numpy(mask.astype(np.float32)).float()
    mask = repeat(mask, 'h w -> b 1 h w', b=batch_size)
    return mask


@torch.no_grad()
def run(models, dset, batch_size, temperature, top_k, chain_schedule, num_runs):
    img_spatial = models[0].first_stage_model.encoder.resolution
    img_shape = [batch_size, img_spatial, img_spatial, 3]

    masking_option = st.selectbox('Select masking option',
                                  ['upper-half completion', 'window-inpainting', 'window-outpainting'],
                                  index=0)

    mask = generate_mask(masking_option, img_shape)

    if st.button('Sample with chain'):

        for n in trange(num_runs, desc="Data"):
            indices = np.random.choice(len(dset), batch_size, replace=False)
            example = default_collate([dset[i] for i in indices])
            logs = sample_unconditional(models, batch_size=batch_size,
                                        temperature_schedule=temperature, top_k_schedule=top_k,
                                        example=example, mask=mask, chain_schedule=chain_schedule)

            log_to_st(logs, n)


def log_to_st(log, n):
    keys = ['inputs', 'masked_inputs', 'samples']
    bs = log[keys[0]].shape[0]

    flatgrid = torch.cat([torch.clamp(log[k].detach().cpu(), -1., 1.) for k in keys], dim=0)
    grid = make_grid(flatgrid, nrow=bs, normalize=True).permute(1, 2, 0).numpy()

    st.image(grid, f'Masked samples #{n + 1} (top: original, mid: masked input, bottom: sample)')


@contextmanager
def ema_scope(model, active=False, context=None):
    if active:
        model.transformer_ema.store(model.transformer.parameters())
        model.transformer_ema.copy_to(model.transformer)
        if context is not None:
            print(f"{context}: Switched to EMA weights")
    try:
        yield None
    finally:
        if active:
            model.transformer_ema.restore(model.transformer.parameters())
            if context is not None:
                print(f"{context}: Restored training weights")


@contextmanager
def on_gpu(model, context=None):
    model = model.cuda()
    if context is not None:
        print(f"{context}: Moved model to GPU")
    try:
        yield None
    finally:
        model = model.cpu()
        torch.cuda.empty_cache()
        if context is not None:
            print(f"{context}: Moved model to CPU")


def load_model_from_config(config, sd, gpu=True, eval_mode=True):
    print("config:")
    print(OmegaConf.to_yaml(config))
    model = instantiate_from_config(config["model"])
    if sd is not None:
        m, u = model.load_state_dict(sd, strict=False)
    if gpu:
        model.cuda()
    if eval_mode:
        model.eval()
    return {"model": model}


@st.cache(allow_output_mutation=True)
def get_data(config):
    # get data
    try:
        if config.data.params.train.target == "braket.data.faceshq.FFHQTrain":
            config.data.params.train.params.random_flip = False
            print("Disabled random flip for FFHQ train")
    except Exception:
        pass
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    return data


def get_config(path):
    config = OmegaConf.load(path)
    return config


@st.cache(allow_output_mutation=True)
def load_models(paths, gpu=False, eval_mode=True):
    assert not gpu, 'moving them later'
    models = list()
    configs = list()
    global_steps = list()
    for ckpt_path, config_path in zip(paths["checkpoints"], paths["configs"]):
        print(f"loading config from {config_path} and model from {ckpt_path}")
        config = get_config(config_path)
        pl_sd = torch.load(ckpt_path, map_location="cpu")
        global_step = pl_sd["global_step"]
        model = load_model_from_config(config, pl_sd["state_dict"], gpu=gpu, eval_mode=eval_mode)["model"]
        models.append(model)
        configs.append(config)
        global_steps.append(global_step)
        print(f"loaded model from global step {global_step}")
    if "repeat" in paths:
        n_models = list()
        n_configs = list()
        n_global_steps = list()
        for i, n in enumerate(paths["repeat"]):
            print(f"Repeating model {i} {n}x.")
            n_models += n * [models[i]]
            n_configs += n * [configs[i]]
            n_global_steps += n * [global_steps[i]]
        models = n_models
        configs = n_configs
        global_steps = n_global_steps
    return models, configs, global_steps


if __name__ == "__main__":
    sys.path.append(os.getcwd())
    if not st._is_running_with_streamlit:
        print("Not running with streamlit. Redefining st functions...")
        st.info = print
        st.write = print

    yaml_path = sys.argv[1]
    paths = OmegaConf.load(yaml_path)
    paths = OmegaConf.to_container(paths)

    gpu = True
    eval_mode = True

    models, configs, global_steps = load_models(paths, gpu=False, eval_mode=eval_mode)
    if models[0].conditioner is not None:
        raise NotImplementedError('Currently only available for unconditional models.')
    device = torch.device("cuda") if gpu else torch.device("cpu")
    dsets = get_data(configs[0])

    split = "validation"
    dset = dsets.datasets[split]
    print(f"Dataset size: {len(dset)}")

    codebook_size = models[0].first_stage_model.quantize.re_embed

    st.sidebar.write('Sampling options')
    n_runs = st.sidebar.number_input('Number of runs', 1, 100, 1)
    batch_size = st.sidebar.number_input('Batch size', 1, 20, 4)

    top_k = get_top_k_schedule(len(models), codebook_size=codebook_size)
    temperature = get_temperature_schedule(len(models))

    chain_schedule = []
    st.write('Define chain schedule')
    st.info(
        f'The n-th entry in the chain schedule defines the number of sucessive runs, '
        f'the n-th AR submodel should perfom before passing the output to the next submodel.')
    for n in range(len(models)):
        if models[n].redraw_prob != 'geometric':
            if n == len(models) - 1:
                def_chain_len = 1
            else:
                def_chain_len = 5
        else:
            def_chain_len = 3
        chain_n = st.number_input(f"Chain length for scale #{n + 1}", min_value=1, max_value=100, value=def_chain_len)
        chain_schedule.append(chain_n)

    chain_schedule = chain_schedule

    run(models, dset, batch_size, temperature=temperature, top_k=top_k, chain_schedule=chain_schedule, num_runs=n_runs)
