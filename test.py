import os
import time
from glob import glob

from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets import TestsDataset
from model_archs.model import ExampleModel
from options import TestOptions
from utils.utils import *

opt = TestOptions().parse()

os.makedirs(f"{opt.result_dir}/", exist_ok=True)
scene_paths = sorted(glob(f"{opt.data_root}/*"))
for scene_path in scene_paths:
    scene_name = scene_path.split('/')[-1]
    os.makedirs(os.path.join(opt.result_dir, scene_name), exist_ok=True)

os.makedirs(f"{opt.result_dir}/", exist_ok=True)

test_dataset = TestsDataset(opt.data_root)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.num_workers, pin_memory=True)
print(f"Test dataset loaded. Length: {len(test_dataset)}")

model = ExampleModel().cuda()
model.load_state_dict(torch.load(opt.model_path))
print(f"Model loaded: {opt.model_path}")

psnr_lq = 0
ssim_lq = 0
psnr_predict = 0
ssim_predict = 0
time_average = 0
model.eval()
with torch.no_grad():
    for i, (img_lq, img_gt, scene_name, img_lq_name, width_orig, height_orig) in enumerate(tqdm(test_loader)):
        img_lq = img_lq.cuda()
        img_gt = img_gt.cuda()

        time_start = time.time()

        img_predict = model(img_lq)

        time_end = time.time()
        time_average += time_end - time_start

        psnr_lq += psnr(img_lq, img_gt)
        ssim_lq += ssim(img_lq, img_gt)
        psnr_predict += psnr(img_predict, img_gt)
        ssim_predict += ssim(img_predict, img_gt)

        img_predict = torch.clamp(img_predict, 0, 1)
        img_predict = resize(img_predict, (width_orig, height_orig))
        save_test_img(img_predict, f"{opt.result_dir}/{scene_name[0]}/{img_lq_name[0]}")

psnr_lq /= len(test_loader)
ssim_lq /= len(test_loader)
psnr_predict /= len(test_loader)
ssim_predict /= len(test_loader)
time_average /= len(test_loader)

print(f"PSNR (LQ): {psnr_lq:.4f}, SSIM (LQ): {ssim_lq:.4f}")
print(f"PSNR (Predict): {psnr_predict:.4f}, SSIM (Predict): {ssim_predict:.4f}")
print(f"Average inference time: {time_average:.4f} s")
