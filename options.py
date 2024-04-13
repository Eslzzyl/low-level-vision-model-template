import argparse

import torch


def show_options(opt):
    args = vars(opt)
    print('************ Options ************')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('************** End **************')


class TrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # ---------------------------------------- step 1/5 : parameters preparing... ----------------------------------------
        self.parser.add_argument("--seed", type=int, default=42, help="random seed")
        self.parser.add_argument("--pretrained", type=str, default=None, help="if specified, will load the pretrained g model")
        self.parser.add_argument("--model_dir", type=str, default='./model', help="path of saving models")
        self.parser.add_argument("--log_dir", type=str, default='./log', help="path of saving log")
        self.parser.add_argument("--experiment", type=str, default='experiment', help="name of experiment")

        # ---------------------------------------- step 2/5 : data loading... ------------------------------------------------
        self.parser.add_argument("--train_data_root", type=str, default='', required=True, help="training dataset root")
        self.parser.add_argument("--val_data_root", type=str, default='', required=True, help="validation dataset root")
        self.parser.add_argument("--train_bs", type=int, default=16, help="size of the training batches (train_bs per GPU)")
        self.parser.add_argument("--val_bs", type=int, default=1, help="size of the validating batches (val_bs per GPU)")
        self.parser.add_argument("--crop", type=int, default=128, help="image size after cropping")
        self.parser.add_argument("--num_workers", type=int, default=8, help="number of cpu threads to use during batch generation")

        # ---------------------------------------- step 3/5 : model defining... ------------------------------------------------
        self.parser.add_argument("--data_parallel", action='store_true', help="if specified, training by data paralleling")

        # ---------------------------------------- step 4/5 : requisites defining... ------------------------------------------------
        self.parser.add_argument("--lr", type=float, default=1e-4, help="initial learning rate")
        self.parser.add_argument("--lr_min", type=float, default=1e-6, help="minimal learning rate")
        self.parser.add_argument("--cosine_warmup_epochs", type=int, default=3, help="Epochs for cosine annealing with warmup")
        self.parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")

        # ---------------------------------------- step 5/5 : training... ------------------------------------------------
        self.parser.add_argument("--val_gap", type=int, default=1, help="the gap between two validations, also the gap between two saving operation, in epoch")
        self.parser.add_argument("--log_gap", type=int, default=10, help="the gap between two logs, in iteration")
        self.parser.add_argument("--save_img_gap", type=int, default=500, help="the gap between two saving images, in iteration")

    def parse(self, show=True):
        opt = self.parser.parse_args()

        if opt.data_parallel:
            opt.train_bs = opt.train_bs * torch.cuda.device_count()
            opt.val_bs = torch.cuda.device_count()
            opt.num_workers = opt.num_workers * torch.cuda.device_count()

        if show:
            show_options(opt)

        return opt


class TestOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # ---------------------------------------- step 1/4 : parameters preparing... ----------------------------------------
        self.parser.add_argument("--result_dir", type=str, default='./result', help="path of saving images")

        # ---------------------------------------- step 2/4 : data loading... ------------------------------------------------
        self.parser.add_argument("--data_root", type=str, default='', required=True, help="derained dataset root")
        self.parser.add_argument("--num_workers", type=int, default=8, help="number of cpu threads to use during batch generation")
        
        # ---------------------------------------- step 3/4 : model defining... ------------------------------------------------
        self.parser.add_argument("--model_path", type=str, default='', required=True, help="pretrained model path")

    def parse(self, show=True):
        opt = self.parser.parse_args()

        if show:
            show_options(opt)

        return opt
