from glob import glob

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


def crop_resize(img: torch.Tensor, height, width) -> torch.Tensor:
    if img.shape[1] != height or img.shape[2] != width:
        img = img[:, :height, :width]
    return img


class PairedImageDataset(Dataset):
    def __init__(self, data_root, crop_size, mode='train'):
        if mode not in ['train', 'val']:
            raise ValueError(f"mode was set {mode}, expected 'train' or 'val'")
        self.mode = mode
        self.crop_size = crop_size
        self.data_root = data_root

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        scene_paths = sorted(glob(f"{data_root}/*"))
        self.img_path_list = []
        for scene_path in scene_paths:
            temp = sorted(glob(scene_path + '/*R-*.png'))
            self.img_path_list.extend(temp)

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        data_len = len(self.img_path_list)
        img_lq_path = self.img_path_list[index % data_len]
        scene_name = img_lq_path.split('/')[-2]
        img_gt_path = glob(f"{self.data_root}/{scene_name}/*C-000.png")[0]

        img_lq = Image.open(img_lq_path)
        img_gt = Image.open(img_gt_path)

        # normalize to [0, 1], WHC to CHW, to tensor
        img_lq = self.transform(img_lq)
        img_gt = self.transform(img_gt)

        # adjust to same size
        min_height = min(img_lq.shape[1], img_gt.shape[1])
        min_width = min(img_lq.shape[2], img_gt.shape[2])
        img_lq = crop_resize(img_lq, min_height, min_width)
        img_gt = crop_resize(img_gt, min_height, min_width)

        # pad if needed
        pad_width = self.crop_size - min_width if min_width < self.crop_size else 0
        pad_height = self.crop_size - min_height if min_height < self.crop_size else 0
        img_lq = TF.pad(img_lq, [0, 0, pad_width, pad_height], padding_mode='reflect')
        img_gt = TF.pad(img_gt, [0, 0, pad_width, pad_height], padding_mode='reflect')

        # random crop
        i, j, h, w = transforms.RandomCrop.get_params(img_lq, output_size=(self.crop_size, self.crop_size))
        img_lq = TF.crop(img_lq, i, j, h, w)
        img_gt = TF.crop(img_gt, i, j, h, w)

        if self.mode == 'train':
            # data augmentation
            if np.random.random() > 0.5:
                img_lq = TF.hflip(img_lq)
                img_gt = TF.hflip(img_gt)

        return img_lq, img_gt

class TestsDataset(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        scene_paths = sorted(glob(f"{data_root}/*"))
        self.img_list = []
        self.dehaze_img_list = []
        for scene_path in scene_paths:
            temp = sorted(glob(scene_path + '/*R-*.png'))
            self.img_list.extend(temp)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        data_len = len(self.img_list)
        img_lq_path = self.img_list[index % data_len]
        scene_name = img_lq_path.split('/')[-2]
        img_lq_name = img_lq_path.split('/')[-1]
        img_gt_path = glob(f"{self.data_root}/{scene_name}/*C-000.png")[0]

        img_lq = Image.open(img_lq_path)
        img_gt = Image.open(img_gt_path)

        width_orig = img_lq.size[0]
        height_orig = img_lq.size[1]

        # normalize to [0, 1], WHC to CHW, to tensor
        img_lq = self.transform(img_lq)
        img_gt = self.transform(img_gt)

        # adjust to same size
        height = min(img_lq.shape[1], img_gt.shape[1]) // 8 * 8
        width = min(img_lq.shape[2], img_gt.shape[2]) // 8 * 8
        img_lq = crop_resize(img_lq, height, width)
        img_gt = crop_resize(img_gt, height, width)

        return img_lq, img_gt, scene_name, img_lq_name, width_orig, height_orig
