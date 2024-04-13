import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from einops import rearrange
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)


def reshape(x: torch.Tensor) -> np.ndarray:
    return rearrange(x, 'b c h w -> h w (b c)').cpu().numpy()


def psnr(inp: torch.Tensor, target: torch.Tensor):
    """Calculate PSNR between inp and target.

    This method assumes that inp and target are torch.Tensor of shape BCHW or CHW.
    """
    if inp.shape != target.shape:
        raise ValueError(f"Shape mismatch: inp.shape = {inp.shape}, target.shape = {target.shape}")
    # 如果是 BCHW，则进行 reshape；如果是 CHW，则跳过 reshape
    if len(inp.shape) == 4:
        inp = reshape(inp)
        target = reshape(target)

    return peak_signal_noise_ratio(inp, target, data_range=1)


def ssim(inp: torch.Tensor, target: torch.Tensor):
    """Calculate SSIM between inp and target.
    
    This method assumes that inp and target are torch.Tensor of shape BCHW or CHW and data_range=1.
    """
    if inp.shape != target.shape:
        raise ValueError(f"Shape mismatch: inp.shape = {inp.shape}, target.shape = {target.shape}")
    # 如果是 BCHW，则进行 reshape；如果是 CHW，则跳过 reshape
    if len(inp.shape) == 4:
        inp = reshape(inp)
        target = reshape(target)

    return structural_similarity(inp, target, channel_axis=-1, data_range=1)


def save_test_img(img: torch.Tensor, path: str) -> None:
    """Save test image to file.
    
    This method assumes that img is a torch.Tensor of shape BCHW, and only the first image in the batch will be saved.
    """
    if len(img.shape) == 4:
        img = img[0]
    img = img.permute(1, 2, 0).cpu().numpy()
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def resize(img: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(img, size=size, mode='bilinear', align_corners=False)
