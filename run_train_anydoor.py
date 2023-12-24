import argparse

from lightning_utilities.core.rank_zero import rank_zero_info
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from datasets.redress import RedressDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import wandb


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
    parser.add_argument('--iterations', type=int, default=1000000)

    # Compute
    parser.add_argument('--num-gpus', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--accumulation-steps', type=int, default=1)

    # Logging
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--log-frequency', type=int, default=1000)

    return parser.parse_args()


def train_ad(opt):
    rank_zero_info(opt)

    if not opt.no_wandb:
        logger = WandbLogger(project='AnyDoorTryon', config=opt, log_model='all', save_dir='checkpoints')
    else:
        logger = None

    n_gpus = opt.num_gpus or torch.cuda.device_count()
    rank_zero_info(f'Using {n_gpus} GPUs')

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(opt.model_conf).cpu()

    if opt.checkpoint_path is not None:
        missing_keys, unexpected_keys = model.load_state_dict(load_state_dict(opt.checkpoint_path, location='cpu'), strict=False)

    model.learning_rate = opt.lr

    dataset = RedressDataset(root=opt.data_root, num_devices=n_gpus, batch_size=opt.batch_size)
    dataloader = DataLoader(dataset, num_workers=opt.num_workers, batch_size=opt.batch_size, shuffle=True)

    image_logger = ImageLogger(batch_frequency=opt.log_frequency, use_wandb=not opt.no_wandb)

    max_epochs: int = opt.iterations // (len(dataset)//(n_gpus*opt.batch_size))
    rank_zero_info(f'Max epochs: {max_epochs}')

    trainer = pl.Trainer(
        strategy="ddp",
        precision='16-mixed',
        accelerator="gpu",
        callbacks=[image_logger],
        # progress_bar_refresh_rate=1,
        # accumulate_grad_batches=opt.accumulation_steps,
        logger=logger,
        max_epochs=max_epochs,
    )

    # Run training
    trainer.fit(model, dataloader)
    wandb.finish()


if __name__ == '__main__':
    train_ad(parse_args())
