import os

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.datasets import PairedImageDataset
from model_archs.model import ExampleModel
from options import TrainOptions
from pytorch_msssim import SSIM
from utils.utils import *

opt = TrainOptions().parse()

set_random_seed(opt.seed)

os.makedirs(f"{opt.model_dir}/{opt.experiment}/", exist_ok=True)
os.makedirs(f"{opt.log_dir}/{opt.experiment}", exist_ok=True)

writer = SummaryWriter(f"{opt.log_dir}/{opt.experiment}")

train_dataset = PairedImageDataset(opt.train_data_root, opt.crop, mode='train')
train_loader = DataLoader(train_dataset, batch_size=opt.train_bs, shuffle=True, num_workers=opt.num_workers, pin_memory=True)
print(f"Train dataset loaded. Length: {len(train_dataset)}")

val_dataset = PairedImageDataset(opt.val_data_root, opt.crop, mode='val')
val_loader = DataLoader(val_dataset, batch_size=opt.val_bs, shuffle=False, num_workers=opt.num_workers, pin_memory=True)
print(f"Val dataset loaded. Length: {len(val_dataset)}")

model = ExampleModel().cuda()

if opt.data_parallel:
    model = torch.nn.DataParallel(model)

if opt.pretrained:
    print("Loading pretrained model:", opt.pretrained)
    model.load_state_dict(torch.load(opt.pretrained))
    print("Model loaded.")

optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, opt.cosine_warmup_epochs, T_mult=2, eta_min=opt.lr_min)

# Loss
ssim_loss = SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=3)
L1_loss = torch.nn.L1Loss()

# Training
model.train()
iteration = 0
for epoch in range(opt.n_epochs):
    print(f"Epoch: {epoch + 1}")
    for i, (img_lq, img_gt) in enumerate(tqdm(train_loader)):
        iteration += 1

        img_lq = img_lq.cuda()
        img_gt = img_gt.cuda()

        img_predict = model(img_lq)

        model.zero_grad()
        s_loss = 1 - ssim_loss(img_predict, img_gt)
        l_loss = L1_loss(img_predict, img_gt)

        loss = 0.6 * s_loss + 0.4 * l_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)

        optimizer.step()

        if i % opt.log_gap == 0:
            writer.add_scalars('Train/loss', {
                'Overall Loss': loss.item(),
                'L1 Loss': l_loss.item(),
                'Neg-SSIM Loss': s_loss.item()
            }, iteration)
        if i % opt.save_img_gap == 0:
            img_predict = torch.clamp(img_predict, 0, 1)
            img_list = [
                img_lq[0].unsqueeze(0),
                img_predict[0].unsqueeze(0),
                img_gt[0].unsqueeze(0)
            ]
            frame = torch.cat(img_list, dim=3)
            writer.add_images('Train/img', frame, iteration)

    writer.add_scalars('Train/LR', {
        'LR': optimizer.param_groups[0]['lr'],
    }, epoch + 1)

    scheduler.step()

    # Validation
    if epoch % opt.val_gap == 0:
        print(f"Validating on Epoch {epoch + 1}:")
        model.eval()
        psnr_lq = 0
        ssim_lq = 0
        psnr_predict = 0
        ssim_predict = 0
        with torch.no_grad():
            for i, (img_lq, img_gt) in enumerate(tqdm(val_loader)):
                img_lq = img_lq.cuda()
                img_gt = img_gt.cuda()

                img_predict = model(img_lq)

                psnr_lq += psnr(img_lq, img_gt) * img_gt.shape[0]
                ssim_lq += ssim(img_lq, img_gt) * img_gt.shape[0]
                psnr_predict += psnr(img_predict, img_gt) * img_gt.shape[0]
                ssim_predict += ssim(img_predict, img_gt) * img_gt.shape[0]

        psnr_lq /= len(val_loader)
        ssim_lq /= len(val_loader)
        psnr_predict /= len(val_loader)
        ssim_predict /= len(val_loader)

        writer.add_scalars('Val/PSNR', {
            'PSNR_lq': psnr_lq,
            'PSNR_predict': psnr_predict,
        }, epoch)
        writer.add_scalars('Val/SSIM', {
            'SSIM_lq': ssim_lq,
            'SSIM_predict': ssim_predict,
        }, epoch)

    img_predict = torch.clamp(img_predict, 0, 1)
    img_list = [
        img_lq[0].unsqueeze(0),
        img_predict[0].unsqueeze(0),
        img_gt[0].unsqueeze(0)
    ]
    frame = torch.cat(img_list, dim=3)
    writer.add_images('Val/img', frame, epoch + 1)

    torch.save(model.state_dict(), f"{opt.model_dir}/{opt.experiment}/model_{opt.experiment}_epoch{epoch}.pth")

    model.train()
