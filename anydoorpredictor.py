from typing import Union

import cv2
import einops
import torch
from PIL import Image
import numpy as np
from omegaconf import OmegaConf

from cldm.ddim_hacked import DDIMSampler
from cldm.model import create_model, load_state_dict
from datasets.data_utils import get_bbox_from_mask, expand_image_mask, pad_to_square, sobel, expand_bbox, box2squre, \
    box_in_box
from datasets.redress import RedressDataset


def crop_back( pred, tar_image,  extra_sizes, tar_box_yyxx_crop):
    H1, W1, H2, W2 = extra_sizes
    y1,y2,x1,x2 = tar_box_yyxx_crop
    pred = cv2.resize(pred, (W2, H2))
    m = 5  # maigin_pixel

    if W1 == H1:
        tar_image[y1+m :y2-m, x1+m:x2-m, :] =  pred[m:-m, m:-m]
        return tar_image

    if W1 < W2:
        pad1 = int((W2 - W1) / 2)
        pad2 = W2 - W1 - pad1
        pred = pred[:,pad1: -pad2, :]
    else:
        pad1 = int((H2 - H1) / 2)
        pad2 = H2 - H1 - pad1
        pred = pred[pad1: -pad2, :, :]

    gen_image = tar_image.copy()
    gen_image[y1+m :y2-m, x1+m:x2-m, :] =  pred[m:-m, m:-m]
    return gen_image


def process_pairs(ref_image, ref_mask, ref_dp_mask, tar_image, tar_mask, tar_dp_mask):
    # ========= Reference ===========
    # ref expand
    ref_box_yyxx = get_bbox_from_mask(ref_mask)

    # ref filter mask
    ref_mask_3 = np.stack([ref_mask, ref_mask, ref_mask], -1)
    masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1-ref_mask_3)

    y1, y2, x1, x2 = ref_box_yyxx
    masked_ref_image = masked_ref_image[y1:y2, x1:x2, :]
    ref_mask = ref_mask[y1:y2, x1:x2]

    ratio = np.random.randint(12, 13) / 10
    masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=ratio)
    ref_mask_3 = np.stack([ref_mask, ref_mask, ref_mask], -1)

    # to square and resize
    masked_ref_image = pad_to_square(masked_ref_image, pad_value=255, random=False)
    masked_ref_image = cv2.resize(masked_ref_image, (224, 224)).astype(np.uint8)

    if ref_dp_mask is not None:
        ref_dp_mask = pad_to_square(ref_dp_mask, pad_value=0, random=False)
        ref_dp_mask = cv2.resize(ref_dp_mask.astype(np.uint8), (224, 224),
                                 interpolation=cv2.INTER_NEAREST).astype(np.float32)
        # to one-hot
        ref_dp_mask_onehot = (np.arange(25) == ref_dp_mask[..., None]).astype(np.uint8)  # HxWx25
    else:
        ref_dp_mask_onehot = None

    if tar_dp_mask is not None:
        tar_dp_mask_t = pad_to_square(tar_dp_mask, pad_value=0, random=False)
        tar_dp_mask_t = cv2.resize(tar_dp_mask_t.astype(np.uint8), (224, 224),
                                   interpolation=cv2.INTER_NEAREST).astype(np.float32)
        # to one-hot
        tar_dp_mask_onehot = (np.arange(25) == tar_dp_mask_t[..., None]).astype(np.uint8)  # HxWx25
    else:
        tar_dp_mask_onehot = None

    ref_mask_3 = pad_to_square(ref_mask_3 * 255, pad_value=0, random=False)
    ref_mask_3 = cv2.resize(ref_mask_3, (224, 224)).astype(np.uint8)
    ref_mask = ref_mask_3[:, :, 0]

    # ref no aug
    masked_ref_image_aug = masked_ref_image

    # collage aug
    masked_ref_image_compose, ref_mask_compose = masked_ref_image, ref_mask
    masked_ref_image_aug = masked_ref_image_compose.copy()
    # ref_mask_3 = np.stack([ref_mask_compose,ref_mask_compose,ref_mask_compose],-1)
    ref_image_collage = sobel(masked_ref_image_compose, ref_mask_compose/255)

    # ========= Target ===========
    tar_box_yyxx = get_bbox_from_mask(tar_mask)
    tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=[1.1,1.2])

    # crop
    tar_box_yyxx_crop = expand_bbox(tar_image, tar_box_yyxx, ratio=[1.5, 3])    #1.2 1.6
    tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop)  # crop box
    y1, y2, x1, x2 = tar_box_yyxx_crop
    cropped_target_image = tar_image[y1:y2, x1:x2, :]
    if tar_dp_mask is not None:
        cropped_target_dp_mask = tar_dp_mask[y1:y2, x1:x2]
    else:
        cropped_target_dp_mask = None

    cropped_tar_mask = tar_mask[y1:y2, x1:x2]
    tar_box_yyxx = box_in_box(tar_box_yyxx, tar_box_yyxx_crop)
    y1, y2, x1, x2 = tar_box_yyxx

    # collage
    ref_image_collage = cv2.resize(ref_image_collage, (x2-x1, y2-y1))
    # ref_mask_compose = cv2.resize(ref_mask_compose.astype(np.uint8), (x2-x1, y2-y1))
    # ref_mask_compose = (ref_mask_compose > 128).astype(np.uint8)

    collage = cropped_target_image.copy()
    collage[y1:y2, x1:x2, :] = ref_image_collage

    collage_mask = cropped_target_image.copy() * 0.0
    collage_mask[y1:y2, x1:x2, :] = 1.0

    # the size before pad
    H1, W1 = collage.shape[0], collage.shape[1]

    cropped_target_image = pad_to_square(cropped_target_image, pad_value = 0, random = False).astype(np.uint8)
    collage = pad_to_square(collage, pad_value = 0, random = False).astype(np.uint8)
    collage_mask = pad_to_square(collage_mask, pad_value = -1, random = False).astype(np.uint8)

    # the size after pad
    H2, W2 = collage.shape[0], collage.shape[1]
    cropped_target_image = cv2.resize(cropped_target_image, (512, 512)).astype(np.float32)
    collage = cv2.resize(collage, (512, 512)).astype(np.float32)
    collage_mask = (cv2.resize(collage_mask, (512, 512)).astype(np.float32) > 0.5).astype(np.float32)
    if cropped_target_dp_mask is not None:
        cropped_target_dp_mask = pad_to_square(cropped_target_dp_mask, pad_value=0, random=False).astype(np.uint8)
        cropped_target_dp_mask = cv2.resize(cropped_target_dp_mask.astype(np.uint8), (512, 512),
                                            interpolation=cv2.INTER_NEAREST).astype(np.float32)
        cropped_target_dp_mask = cropped_target_dp_mask / 24  # number of DP classes -1

    masked_ref_image_aug = masked_ref_image_aug / 255
    cropped_target_image = cropped_target_image / 127.5 - 1.0
    collage = collage / 127.5 - 1.0
    collage = np.concatenate([collage, collage_mask[:, :, :1]], -1)

    if cropped_target_dp_mask is not None:
        collage = np.concatenate([collage, cropped_target_dp_mask[:, :, None]], -1)

    if ref_dp_mask_onehot is not None:
        masked_ref_image_aug = np.concatenate([masked_ref_image_aug, ref_dp_mask_onehot], -1)
    if tar_dp_mask_onehot is not None:
        masked_ref_image_aug = np.concatenate([masked_ref_image_aug, tar_dp_mask_onehot], -1)

    item = dict(ref=masked_ref_image_aug.copy(), jpg=cropped_target_image.copy(), hint=collage.copy(), extra_sizes=np.array([H1, W1, H2, W2]), tar_box_yyxx_crop=np.array( tar_box_yyxx_crop ) )
    return item


class AnydoorPredictor:
    def __init__(self,
                 inference_conf='./configs/inference.yaml'):
        config = OmegaConf.load(inference_conf)
        model_ckpt = config.pretrained_model
        model_config = config.config_file

        model = create_model(model_config).cpu()
        model.load_state_dict(load_state_dict(model_ckpt, location='cuda'))
        model = model.cuda()
        self.model = model
        self.ddim_sampler = DDIMSampler(model)

    def predict(
            self,
            ref_image_path: Union[str, np.ndarray, Image.Image],
            ref_mask: Union[str, np.ndarray, Image.Image],
            ref_dp_mask: Union[str, np.ndarray, Image.Image],
            tar_image_path: Union[str, np.ndarray, Image.Image],
            tar_mask: Union[str, np.ndarray, Image.Image],
            tar_dp_mask: Union[str, np.ndarray, Image.Image],
            btype: str,
            save_memory=False,
            guidance_scale: float = 8,
            eta=0.0,
            strength=1,
            ddim_steps=50,
            num_samples=1,
    ):
        ref_image, ref_mask, tar_image, tar_mask, ref_dp_mask, tar_dp_mask = RedressDataset.process_sample(
            ref_image_path,
            ref_mask,
            ref_dp_mask,
            tar_image_path,
            tar_mask,
            tar_dp_mask,
            btype,
        )

        item = process_pairs(ref_image, ref_mask, ref_dp_mask, tar_image, tar_mask, tar_dp_mask)

        # ref = item['ref'] * 255
        # tar = item['jpg'] * 127.5 + 127.5
        # hint = item['hint'] * 127.5 + 127.5

        # hint_image = hint[:, :, :-1]
        # hint_mask = item['hint'][:, :, -1] * 255
        # hint_mask = np.stack([hint_mask, hint_mask, hint_mask], -1)
        # ref = cv2.resize(ref.astype(np.uint8), (512, 512))

        if save_memory:
            self.model.low_vram_shift(is_diffusing=False)

        ref = item['ref']
        # tar = item['jpg']
        hint = item['hint']  # 512x512x4

        control = torch.from_numpy(hint.copy()).float().cuda()
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        clip_input = torch.from_numpy(ref.copy()).float().cuda()
        clip_input = torch.stack([clip_input for _ in range(num_samples)], dim=0)
        clip_input = einops.rearrange(clip_input, 'b h w c -> b c h w').clone()

        guess_mode = False
        H, W = 512, 512

        cond = {"c_concat": [control], "c_crossattn": [self.model.get_learned_conditioning(clip_input)]}
        un_cond = {"c_concat": None if guess_mode else [control],
                   "c_crossattn": [self.model.get_learned_conditioning([torch.zeros((1, 53, 224, 224))] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if save_memory:
            self.model.low_vram_shift(is_diffusing=True)

        guess_mode = False  # gr.Checkbox(label='Guess Mode', value=False)

        self.model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else (
                    [strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = self.ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=guidance_scale,
                                                     unconditional_conditioning=un_cond)
        if save_memory:
            self.model.low_vram_shift(is_diffusing=False)

        x_samples = self.model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples,
                                      'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy()

        pred = x_samples[0]
        pred = np.clip(pred, 0, 255)[1:, :, :]
        sizes = item['extra_sizes']
        tar_box_yyxx_crop = item['tar_box_yyxx_crop']
        gen_image = crop_back(pred, tar_image, sizes, tar_box_yyxx_crop)
        return Image.fromarray(gen_image)
