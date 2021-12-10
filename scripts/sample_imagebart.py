import os, sys
import hashlib
import torch
import numpy as np
from omegaconf import OmegaConf
import streamlit as st
from PIL import Image
from main import instantiate_from_config
from torchvision.utils import make_grid
from contextlib import contextmanager

rescale = lambda x: (x + 1.) / 2.


def bchw_to_st(x):
    return rescale(x.detach().cpu().numpy().transpose(0,2,3,1))

def chw_to_st(x):
    return rescale(x.detach().cpu().numpy().transpose(1,2,0))

def chw_to_pillow(x):
    return Image.fromarray(chw_to_np(x))

def chw_to_np(x):
    return (255 * rescale(x.detach().cpu().numpy().transpose(1, 2, 0))).clip(0, 255).astype(np.uint8)


def computeMD5hash(string):
    m = hashlib.md5()
    m.update(string.encode('utf-8'))
    return m.hexdigest()


class L1(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.abs(x-y).sum(dim=[1,2,3]).mean()



@torch.no_grad()
def custom_log_images(model, batch, temperature):
    log = dict()
    x = model.get_input(batch, model.image_key)
    x = x.to(model.device)
    # encode
    h = model.encoder(x)
    h = model.quant_conv(h)
    quant, _, _ = model.quantize(h, temp=temperature, rescale_logits=True)
    # decode
    x_rec = model.decode(quant)
    log["inputs"] = x
    log["reconstructions"] = x_rec
    return log


def grid2img(grid):
    grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
    grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
    grid = grid.numpy()
    grid = (grid * 255).astype(np.uint8)
    print(f"grid.max/grid.min {grid.max()}/{grid.min()}")
    grid = Image.fromarray(grid)
    return grid


def get_top_k_schedule(n_steps, codebook_size):
    tk_schedule = st.radio("top-k scheduling", ["constant", "linear", "user"], 2)
    if tk_schedule == "constant":
        tk = st.number_input("Constant Top-K Value", value=codebook_size)
        top_k_schedule = (np.ones(n_steps) * tk).astype(int)
    elif tk_schedule == "linear":
        tk_start = st.number_input("Start Top-K Value", value=codebook_size)
        tk_end = st.number_input("End Top-K Value", value=codebook_size//4)
        top_k_schedule = np.linspace(tk_start, tk_end, n_steps).astype(int)
    elif tk_schedule == "user":
        default = f"{codebook_size}," * n_steps
        tk_list = st.text_input(f"Top-K Values (comma separated, must be {n_steps}, counted in sampling order, i.e. from last to first scale)", f"{default[:-1]}")
        tks = tk_list.split(",")
        top_k_schedule = list()
        for tk in tks:
            top_k_schedule.append(int(tk))
        assert len(top_k_schedule) == n_steps
    else:
        return None
    return top_k_schedule

def get_temperature_schedule(n_steps):
    type = st.radio("temperature scheduling", ["constant", "linear", "user"], 2)
    if type == "constant":
        tk = st.number_input("Constant Temperature Value", value=1.0)
        schedule = (np.ones(n_steps) * tk)
    elif type == "linear":
        tk_start = st.number_input("Start Temperature Value", value=1.)
        tk_end = st.number_input("End Top-Temperature Value", value=0.1)
        schedule = np.linspace(tk_start, tk_end, n_steps)
    elif type == "user":
        default = f"{1.0}," * n_steps
        tk_list = st.text_input(f"Temperature Values (comma separated, must be {n_steps}, counted in sampling order, i.e. from last to first scale)", f"{default[:-1]}")
        tks = tk_list.split(",")
        schedule = list()
        for tk in tks:
            schedule.append(float(tk))
        assert len(schedule) == n_steps
    else:
        return None
    return schedule


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


@torch.no_grad()
def run(models, user_conditioning, batch_size, device=torch.device("cuda"),conditional=False):
    assert type(models) == list

    n_scales = len(models)
    codebook_size = models[0].first_stage_model.quantize.re_embed

    cond = None
    start_index = len(models) - 1
    model = models[start_index]


    n_downs= model.first_stage_model.encoder.num_resolutions - 1
    h = model.first_stage_model.encoder.resolution // (2**n_downs)
    w = model.first_stage_model.encoder.resolution // (2**n_downs)
    dim_z = model.first_stage_model.embed_dim


    index_shape = [batch_size, h * w]
    qzshape = [batch_size, dim_z, h, w]

    st.info(f'Latent shape is {qzshape}')
    if user_conditioning is not None:
        exmpl = {model.conditioner.key: user_conditioning}
        cond = model.get_conditioning(exmpl).to(device)
        st.sidebar.write(f"cond.shape: {cond.shape}")


    top_k_schedule = get_top_k_schedule(n_scales, codebook_size=codebook_size)[::-1]
    temperature_schedule = get_temperature_schedule(n_scales)[::-1]
    st.text(f"top-k schedule: {top_k_schedule}")
    st.text(f"temperature schedule: {temperature_schedule}")

    n_batches = st.number_input("number runs", value=1, min_value=1, max_value=1000)
    steps = n_scales * [h*w]


    if st.button("Sample", False):
        grid_ph = st.empty()
        final_samples = list()
        for n in range(n_batches):

            scaleinfo = st.empty()
            scale = start_index
            c_scale_indices = torch.randint(0,
                          model.first_stage_model.quantize.re_embed,
                          index_shape,
                          device=model.device)
            current_scale = (scale * torch.ones(batch_size, 1)).to(c_scale_indices).to(device)

            steppys = st.empty()
            cb = lambda x: steppys.write(f"{x}/{h*w}")
            for model_count, model in enumerate(models[::-1]):
                with on_gpu(model, "Sampling"):
                    temperature = temperature_schedule[scale]
                    top_k = top_k_schedule[scale]
                    scaleinfo.write(f"sampling scale {scale+1}/{n_scales}, temp = {temperature:.2f}, top-k = {top_k}")

                    with ema_scope(model, active=True, context="Sampling"):

                        assert (current_scale + 1)[0].item() == model.single_scale, \
                            f"{(current_scale + 1)[0].item()} =/= {model.single_scale} :("
                        c_scale_indices = model.sample_single_scale(c_scale_indices.to(device),
                                                                    (current_scale+1).to(device),
                                                                    temp_x=None,
                                                                    steps=steps[scale],
                                                                    temperature=temperature,
                                                                    top_k=top_k,
                                                                    cond=cond.to(model.device) if cond is not None else None,
                                                                    callback=cb
                                                                    )

                        if model_count == len(models) -1:
                            final_samples.append(model.decode_to_img(c_scale_indices,
                                                     [batch_size, qzshape[1], qzshape[2], qzshape[3]]))
                    scale -= 1
                    current_scale = (scale * torch.ones(batch_size, 1)).to(device).long()

            intermediate_grid = make_grid(final_samples[-1],nrow=batch_size,padding=0)
            st.image(chw_to_st(intermediate_grid),clamp=True,output_format='PNG')

        final_samples = torch.cat(final_samples, 0)
        grid = make_grid(final_samples, nrow=batch_size, padding=0)
        grid_ph.image(chw_to_st(grid), clamp=True, output_format="PNG")

@torch.no_grad()
def render_as_grid(scale_samples, batch_size, stack=True):
    # make a grid
    if stack:
        scale_samples = torch.stack(scale_samples, dim=0)
    assert batch_size == scale_samples.shape[1]
    grids = []
    for i in range(scale_samples.shape[1]):
        grid = make_grid(scale_samples[:, i, ...], nrow=scale_samples.shape[0])
        grids.append(grid)

    for k in range(len(grids)):
        st.image(chw_to_st(grids[k]), clamp=True, output_format="PNG", use_column_width=grids[k].shape[2] < 500)


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

    return models, configs, global_steps


if __name__ == "__main__":

    yaml_path = sys.argv[1]
    log_path = yaml_path.split(os.sep)[-1][:-5]
    paths = OmegaConf.load(yaml_path)
    print(OmegaConf.to_yaml(paths))
    paths = OmegaConf.to_container(paths)

    log_path =paths["metrics"]["savepath"] if 'metrics' in paths and 'savepath' in paths['metrics'] else os.path.join("logs", log_path)
    print(f"loading from .yaml at {yaml_path}")


    print(f'logging to {log_path}')

    st.sidebar.text("Options")
    gpu = st.sidebar.checkbox("GPU", value=True)

    models, configs, global_steps = load_models(paths, gpu=False)
    device = torch.device("cuda") if gpu else torch.device("cpu")

    # dsets = get_data(configs[0])
    step_info = ""
    st.write(step_info[:-2])

    batch_size = st.number_input("Batch size", min_value=1, value=4)
    conditional = models[0].conditioner is not None


    user_conditioning = None
    if conditional:
        st.info("Detected a conditional model.")
        user_inputs = []
        conditioner_key = models[0].conditioner.key

        if conditioner_key == "caption":
            for n in range(batch_size):
                user_input = st.text_input(f"user caption {n}", value=f"Example caption {n}")
                user_inputs.append(user_input)

            #example["caption"] = [user_inputs]
            user_conditioning = user_inputs

            st.write(f"Selected text-prompts are {user_conditioning}")
        elif conditioner_key == "class_label":

            cfd = os.path.dirname(os.path.abspath(__file__))
            integer2human = OmegaConf.load(os.path.join(cfd,'../data/imagenet_ids2labels.yaml'))

            format_fn = lambda x: integer2human[x]
            for n in range(batch_size):
                user_input = st.selectbox(f"user class label {n}", index=144,
                                            options=list(integer2human.keys()),
                                            format_func=format_fn)
                user_inputs.append(int(user_input))

            user_conditioning = torch.tensor(user_inputs)

            human_labels = [integer2human[str(l)] for l in user_inputs]
            st.write(f"Selected class labels are {human_labels}")
        else:
            raise NotImplementedError(f"Model with conditoner key {conditioner_key} not yet implemented.")

    run(models, user_conditioning, batch_size, device=device,conditional=conditional)
