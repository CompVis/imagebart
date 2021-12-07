import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


class ClassProvider(torch.nn.Module):
    def __init__(self, key="class"):
        super().__init__()
        self.key = key

    def forward(self, batch):
        c = batch[self.key][:, None]
        return c


class BasicTokenizer(torch.nn.Module):
    """
    Uses the 'simple tokenizer' of the CLIP model
    https://github.com/openai/CLIP
    """
    def __init__(self, device="cuda", key="caption"):
        super().__init__()
        from clip import tokenize
        self.tknz_fn = tokenize
        self.device = device
        self.key = key

    def forward(self, batch):
        text = batch[self.key]
        tokens = self.tknz_fn(text).to(self.device)
        return tokens


class KeyNotFoundError(Exception):
    def __init__(self, cause, keys=None, visited=None):
        self.cause = cause
        self.keys = keys
        self.visited = visited
        messages = list()
        if keys is not None:
            messages.append("Key not found: {}".format(keys))
        if visited is not None:
            messages.append("Visited: {}".format(visited))
        messages.append("Cause:\n{}".format(cause))
        message = "\n".join(messages)
        super().__init__(message)


def log_txt_as_img(wh, xc, size=10):
    # wh a tuple of (width, height)
    # xc a list of captions to plot
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        font = ImageFont.truetype('data/DejaVuSans.ttf', size=size)
        nc = int(40 * (wh[0]/256))
        lines = "\n".join(xc[bi][start:start+nc] for start in range(0, len(xc[bi]), nc))

        try:
            draw.text((0,0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            print("Cant encode string for logging. Skipping.")

        txt = np.array(txt).transpose(2,0,1)/127.5-1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = torch.tensor(txts)
    return txts

