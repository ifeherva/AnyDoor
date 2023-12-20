import argparse

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from datasets.redress import RedressDataset
from datasets.vitonhd import VitonHDDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from torch.utils.data import ConcatDataset
from cldm.hack import disable_verbosity, enable_sliced_attention
from omegaconf import OmegaConf


def parse_args():
    parser = argparse.ArgumentParser('Train AnyDoor model')

    # Data
    parser.add_argument('--data-root', type=str, default='/media/istvanfe/MLStuff/Datasets/Redress')

    # Model
    parser.add_argument('--model-conf', type=str, default='./configs/anydoor_redress.yaml')
    parser.add_argument('--checkpoint-path', type=str, default=None)

    # Training
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-5)

    # Compute
    parser.add_argument('--num-gpus', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--accumulation-steps', type=int, default=1)

    return parser.parse_args()


def train_ad(opt):
    print(opt)

    logger_freq = 1000
    sd_locked = False
    only_mid_control = False

    n_gpus = opt.num_gpus or torch.cuda.device_count()
    print(f'Using {n_gpus} GPUs')

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(opt.model_conf).cpu()

    if opt.checkpoint_path is not None:
        model.load_state_dict(load_state_dict(opt.checkpoint_path, location='cpu'), strict=False)

    model.learning_rate = opt.lr
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    dataset = RedressDataset(root=opt.data_root)
    dataloader = DataLoader(dataset, num_workers=opt.num_workers, batch_size=opt.batch_size, shuffle=True)

    logger = ImageLogger(batch_frequency=logger_freq)
    trainer = pl.Trainer(gpus=n_gpus, strategy="ddp", precision=16, accelerator="gpu", callbacks=[logger],
                         progress_bar_refresh_rate=1, accumulate_grad_batches=opt.accumulation_steps)

    # Run training
    trainer.fit(model, dataloader)


if __name__ == '__main__':
    train_ad(parse_args())
