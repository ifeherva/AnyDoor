import argparse

import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import wandb
import einops
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from cldm.ddim_hacked import DDIMSampler
from cldm.model import create_model, load_state_dict
from datasets.redress import RedressDataset


model_config = OmegaConf.load('./configs/inference.yaml')
model_ckpt = model_config.pretrained_model
model_config = model_config.config_file

model = create_model(model_config).cpu()
model.load_state_dict(load_state_dict(model_ckpt, location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)

metric = F.mse_loss

project_name = 'Anydoor+ 200k sweep'


def parse_args():
    parser = argparse.ArgumentParser('Optimize model parameters')
    parser.add_argument('--root', type=str, default='/media/istvanfe/MLStuff/Datasets/Redress')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=4)

    return parser.parse_args()


def seek_function():
    wandb.init(project=project_name)

    dataset = Subset(RedressDataset(
        '/media/istvanfe/MLStuff/Datasets/Redress',
        split='test',
        batch_size=1,
    ), np.arange(100))
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4)

    all_metrics = []
    for batch in tqdm(dataloader):
        ref = batch['ref']
        tar = batch['jpg'] * 127.5 + 127.5
        hint = batch['hint']  # 1x512x512x4

        control = hint.float().cuda()
        control = einops.rearrange(control, 'b h w c -> b c h w')

        clip_input = ref.float().cuda()
        clip_input = einops.rearrange(clip_input, 'b h w c -> b c h w')

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning(clip_input)]}
        un_cond = {"c_concat": [control],
                   "c_crossattn": [model.get_learned_conditioning([torch.zeros_like(clip_input)])]}

        # params
        H, W = 512, 512
        shape = (4, H // 8, W // 8)
        eta = 0

        control_strength = wandb.config.control_strength
        ddim_steps = wandb.config.ddim_steps
        guidance_scale = wandb.config.guidance_scale

        model.control_scales = ([control_strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, 1,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=guidance_scale,
                                                     unconditional_conditioning=un_cond,
                                                     disable_wandb=True)
        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu()
        pred = x_samples[0]
        pred = torch.clip(pred, 0, 255)#[1:, :, :]

        mse = metric(pred, tar[0]).item()
        all_metrics.append(mse)

    score = np.mean(all_metrics)
    wandb.log({'mse': score})
    return score


@torch.inference_mode()
def optimize_parameters(opt):
    print(opt)
    assert opt.batch_size == 1, 'Not implemented'

    sweep_configuration = {
        'method': 'bayes',
        'metric': {'goal': 'minimize', 'name': 'mse'},
        'parameters': {
            'ddim_steps': {'min': 20, 'max': 60},
            'control_strength': {'min': 0.0, 'max': 2.0},
            'guidance_scale': {'min': 0.1, 'max': 30.0},
        },
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name)

    wandb.agent(
        sweep_id,
        function=seek_function,
        # count=1,
    )


if __name__ == '__main__':
    optimize_parameters(parse_args())
