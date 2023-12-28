import os
from os.path import join, isdir, dirname, basename

import cv2
import einops
import numpy as np
import torch
import random

from diffusers import StableDiffusionLatentUpscalePipeline
from pytorch_lightning import seed_everything
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from cldm.hack import disable_verbosity, enable_sliced_attention
from datasets.data_utils import *
from datasets.redress import RedressDataset
from segmentation.densepose_segmenter import DenseposeSegmenter
from segmentation.face_segmenter import FaceSegmenter
from segmentation.human_segmenter import ClothSegmenterHF

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import albumentations as A
from omegaconf import OmegaConf
from PIL import Image


save_memory = False
# disable_verbosity()
if save_memory:
    enable_sliced_attention()


config = OmegaConf.load('./configs/inference.yaml')
model_ckpt = config.pretrained_model
model_config = config.config_file

model = create_model(model_config).cpu()
model.load_state_dict(load_state_dict(model_ckpt, location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)

segmenter = ClothSegmenterHF()
densepose_segmenter = DenseposeSegmenter()
face_segmenter = FaceSegmenter()


def aug_data_mask(image, mask):
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        ])
    transformed = transform(image=image.astype(np.uint8), mask = mask)
    transformed_image = transformed["image"]
    transformed_mask = transformed["mask"]
    return transformed_image, transformed_mask


def process_pairs(ref_image, ref_mask, ref_dp_mask, tar_image, tar_mask, tar_dp_mask, tar_image_facemask=None):
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
    # masked_ref_image_aug = masked_ref_image

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

    collage = cropped_target_image.copy()
    collage[y1:y2, x1:x2, :] = ref_image_collage
    collage_mask = np.zeros_like(cropped_target_image).astype(float)
    collage_mask[y1:y2, x1:x2, :] = 1.0

    # unmask face (experimental)
    if tar_image_facemask is not None:
        collage_mask[tar_image_facemask.astype(bool)] = 0
        # Remove the target bias from the target, seems to produce worse results, probably needs training
        collage = np.where(collage_mask.astype(bool), collage, cropped_target_image)

    # the size before pad
    H1, W1 = collage.shape[0], collage.shape[1]

    cropped_target_image = pad_to_square(cropped_target_image, pad_value=0, random=False).astype(np.uint8)
    collage = pad_to_square(collage, pad_value=0, random=False).astype(np.uint8)
    collage_mask = pad_to_square(collage_mask, pad_value=-1, random=False).astype(np.uint8)

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

    item = dict(
        ref=masked_ref_image_aug.copy(),
        jpg=cropped_target_image.copy(),
        hint=collage.copy(),
        extra_sizes=np.array([H1, W1, H2, W2]),
        tar_box_yyxx_crop=np.array(tar_box_yyxx_crop),
    )
    return item


def crop_back(pred, tar_image, extra_sizes, tar_box_yyxx_crop, upscaled_image=None):
    H1, W1, H2, W2 = extra_sizes
    y1, y2, x1, x2 = tar_box_yyxx_crop
    pred = cv2.resize(pred, (W2, H2))
    m = 5  # margin_pixel

    if W1 == H1:
        tar_image[y1+m:y2-m, x1+m:x2-m, :] = pred[m:-m, m:-m]
        # return tar_image
    else:
        if W1 < W2:
            pad1 = int((W2 - W1) / 2)
            pad2 = W2 - W1 - pad1
            pred = pred[:, pad1: -pad2, :]
        else:
            pad1 = int((H2 - H1) / 2)
            pad2 = H2 - H1 - pad1
            pred = pred[pad1: -pad2, :, :]

        gen_image = tar_image.copy()
        gen_image[y1+m: y2-m, x1+m:x2-m, :] = pred[m:-m, m:-m]

    if upscaled_image is not None:
        tar_image_upscaled = Image.fromarray(tar_image)
        upscaled_image = upscaled_image.resize((W2, H2))

        if W1 == H1:
            tar_image_upscaled.paste(upscaled_image.crop((m, m, W2 - m, H2 - m)), (m, m))
            return tar_image, tar_image_upscaled

        if W1 < W2:
            upscaled_image = upscaled_image.crop((pad1, 0, W2-pad2, H2))
        else:
            upscaled_image = upscaled_image.crop((0, pad1, W2, H2-pad2))

        tar_image_upscaled.paste(upscaled_image.crop((m, m, W2-m, H2-m)), (x1+m, y1+m))
    else:
        tar_image_upscaled = None

    return gen_image, tar_image_upscaled


def inference_single_image(
        ref_image, ref_mask, ref_dp_mask,
        tar_image, tar_mask, tar_dp_mask,
        tar_image_facemask=None,
        fix_mask=None,
        guidance_scale=5.0,
        ddim_steps=60,
):

    item = process_pairs(
        ref_image, ref_mask, ref_dp_mask,
        tar_image, tar_mask, tar_dp_mask,
        tar_image_facemask=tar_image_facemask)

    ref = item['ref'] * 255
    tar = item['jpg'] * 127.5 + 127.5
    hint = item['hint'] * 127.5 + 127.5

    hint_image = hint[:, :, :-1]
    hint_mask = item['hint'][:, :, -1] * 255
    hint_mask = np.stack([hint_mask, hint_mask, hint_mask], -1)
    ref = cv2.resize(ref.astype(np.uint8), (512, 512))

    seed = random.randint(0, 65535)
    if save_memory:
        model.low_vram_shift(is_diffusing=False)

    ref = item['ref']
    tar = item['jpg'] 
    hint = item['hint']  # 512x512x4
    num_samples = 1

    control = torch.from_numpy(hint.copy()).float().cuda() 
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()

    clip_input = torch.from_numpy(ref.copy()).float().cuda() 
    clip_input = torch.stack([clip_input for _ in range(num_samples)], dim=0)
    clip_input = einops.rearrange(clip_input, 'b h w c -> b c h w').clone()

    guess_mode = False
    H, W = 512, 512

    cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning(clip_input)]}
    un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([torch.zeros((1, 53, 224, 224))] * num_samples)]}
    shape = (4, H // 8, W // 8)

    if save_memory:
        model.low_vram_shift(is_diffusing=True)

    # ====
    num_samples = 1 #gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
    image_resolution = 512  #gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
    strength = 1  #gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
    guess_mode = False #gr.Checkbox(label='Guess Mode', value=False)
    #detect_resolution = 512  #gr.Slider(label="Segmentation Resolution", minimum=128, maximum=1024, value=512, step=1)
    ddim_steps = ddim_steps #gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
    scale = guidance_scale  #gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
    seed = -1  #gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
    eta = 0.0 #gr.Number(label="eta (DDIM)", value=0.0)

    model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
    samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                 shape, cond, verbose=False, eta=eta,
                                                 unconditional_guidance_scale=scale,
                                                 unconditional_conditioning=un_cond)
    if save_memory:
        model.low_vram_shift(is_diffusing=False)

    # latent upscaler
    upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained("stabilityai/sd-x2-latent-upscaler",
                                                                    torch_dtype=torch.float16)
    upscaler.to("cuda")

    upscaled_image = upscaler(
        prompt='',
        image=samples,  # 1x4x64x64
        num_inference_steps=50,
        guidance_scale=0,
        generator=torch.manual_seed(33),
    ).images[0]

    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy()

    # result = x_samples[0][:, :, ::-1]
    # result = np.clip(result,0,255)

    pred = x_samples[0]
    pred = np.clip(pred, 0, 255)[1:, :, :]
    sizes = item['extra_sizes']
    tar_box_yyxx_crop = item['tar_box_yyxx_crop'] 
    gen_image, upscaled_image = crop_back(pred, tar_image, sizes, tar_box_yyxx_crop, upscaled_image)

    gen_image = Image.fromarray(gen_image)
    if fix_mask is not None:
        erode_radius = 7 * max(1, int(sizes[-1]/512))
        fix_mask = cv2.erode(fix_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                                                  (2 * erode_radius + 1, 2 * erode_radius + 1))).astype('bool')
        upscaled_image.paste(Image.fromarray(tar_image), mask=Image.fromarray(fix_mask))
        gen_image.paste(Image.fromarray(tar_image), mask=Image.fromarray(fix_mask))

    return gen_image, upscaled_image


def run_inference(
    ref_image_path: str,
    tar_image_path: str,
    out_file: str,
    btype: str,
    scale: float = 9,
):
    # ref_image_path = '/media/istvanfe/MLStuff/Datasets/marketing/raw-images/art-7944154_lowres.jpg'
    # tar_image_path = '/media/istvanfe/MLStuff/Datasets/Scraped/aritzia/woman/tops/all_tied_up_sweater_104408_11420/f23_04_a03_104408_11420_on_a.jpg'
    # ref_image_path, tar_image_path = tar_image_path, ref_image_path
    # btype = 'upper_body'

    ref_mask = segmenter(ref_image_path)
    ref_dp_mask = densepose_segmenter(ref_image_path)

    tar_mask = segmenter(tar_image_path)
    tar_dp_mask = densepose_segmenter(tar_image_path)
    tar_face_mask = face_segmenter(tar_image_path)

    ref_image, ref_mask, tar_image, tar_mask, ref_dp_mask, tar_dp_mask, face_mask, hand_mask = RedressDataset.process_sample(
        ref_image_path,
        ref_mask,
        ref_dp_mask,
        tar_image_path,
        tar_mask,
        tar_dp_mask,
        btype,
    )
    face_mask = np.logical_and(face_mask, tar_face_mask)
    fix_mask = face_mask# np.logical_or(face_mask, hand_mask)

    gen_image, upscaled_image = inference_single_image(
        ref_image, ref_mask, ref_dp_mask,
        tar_image, tar_mask, tar_dp_mask,
        tar_image_facemask=None,#face_mask,
        fix_mask=fix_mask,
        guidance_scale=scale,
        ddim_steps=100,
    )

    if isdir(out_file):
        tar_filename = basename(tar_image_path).split('.')[0]
        ref_filename = basename(ref_image_path).split('.')[0]
        out_file = join(out_file, f'{tar_filename}_{ref_filename}.jpg')

    out_root = dirname(out_file)
    out_file_name = basename(out_file)

    out_root = join(out_root, 'scale_{:.1f}'.format(scale))
    os.makedirs(out_root, exist_ok=True)
    out_file = join(out_root, out_file_name)

    gen_image.save(out_file)
    upscaled_image.save(out_file.replace('.jpg', '_upscaled.jpg'))


if __name__ == '__main__':
    scales = [4.5, 6, 7, 8, 9, 10]
    for scale in scales:
        # art-7944154.jpg
        run_inference(
            tar_image_path='/media/istvanfe/MLStuff/Datasets/marketing/raw-images/art-7944154.jpg',
            ref_image_path='/media/istvanfe/MLStuff/Datasets/Scraped/aritzia/woman/blouses/lovefest_satin_bustier_113913_11420/f23_04_a02_113913_11420_on_a.jpg',
            btype='upper_body',
            out_file='/home/istvanfe/Downloads/model_output/anydoor/v1/',
            scale=scale,
        )

        # beautiful-1274051.jpg
        run_inference(
            tar_image_path='/media/istvanfe/MLStuff/Datasets/marketing/raw-images/beautiful-1274051.jpg',
            ref_image_path='/media/istvanfe/MLStuff/Datasets/Scraped/aritzia/woman/dresses/frenchy_dress_114423_11884/f23_04_a08_114423_11884_on_a.jpg',
            btype='full_body',
            out_file='/home/istvanfe/Downloads/model_output/anydoor/v1/',
            scale=scale,
        )

        run_inference(
            tar_image_path='/media/istvanfe/MLStuff/Datasets/marketing/raw-images/beautiful-1274051.jpg',
            ref_image_path='/media/istvanfe/MLStuff/Datasets/Scraped/aritzia/woman/dresses/only_slip_satin_midi_dress_75717_27311/f23_04_a08_75717_27311_on_a.jpg',
            btype='full_body',
            out_file='/home/istvanfe/Downloads/model_output/anydoor/v1/',
            scale=scale,
        )

        # girl-487065.jpg
        run_inference(
            tar_image_path='/media/istvanfe/MLStuff/Datasets/marketing/raw-images/girl-487065.jpg',
            ref_image_path='/media/istvanfe/MLStuff/Datasets/Scraped/aritzia/woman/blouses/the_2780s_comfy_denim_shirt_114523_31906/f23_10_a02_114523_31906_on_a.jpg',
            btype='upper_body',
            out_file='/home/istvanfe/Downloads/model_output/anydoor/v1/',
            scale=scale,
        )

        run_inference(
            tar_image_path='/media/istvanfe/MLStuff/Datasets/marketing/raw-images/girl-487065.jpg',
            ref_image_path='/media/istvanfe/MLStuff/Datasets/Scraped/aritzia/woman/dresses/audience_satin_dress_118262_15104/f23_01_a08_118262_15104_on_b.jpg',
            btype='full_body',
            out_file='/home/istvanfe/Downloads/model_output/anydoor/v1/',
            scale=scale,
        )

        # model-2758787.jpg
        run_inference(
            tar_image_path='/media/istvanfe/MLStuff/Datasets/marketing/raw-images/model-2758787.jpg',
            ref_image_path='/media/istvanfe/MLStuff/Datasets/Scraped/aritzia/woman/dresses/eyecatcher_dress_109178_1274/f23_01_a08_109178_1274_on_a.jpg',
            btype='full_body',
            out_file='/home/istvanfe/Downloads/model_output/anydoor/v1/',
            scale=scale,
        )

        run_inference(
            tar_image_path='/media/istvanfe/MLStuff/Datasets/marketing/raw-images/model-2758787.jpg',
            ref_image_path='/media/istvanfe/MLStuff/Datasets/Scraped/aritzia/woman/dresses/alchemy_dress_113695_1274/f23_04_a08_113695_1274_on_a.jpg',
            btype='full_body',
            out_file='/home/istvanfe/Downloads/model_output/anydoor/v1/',
            scale=scale,
        )

        # model-3296470.png
        run_inference(
            tar_image_path='/media/istvanfe/MLStuff/Datasets/marketing/raw-images/model-3296470.jpg',
            ref_image_path='/media/istvanfe/MLStuff/Datasets/Scraped/aritzia/woman/dresses/maclean_dress_103661_11420/f23_01_a08_103661_11420_on_d.jpg',
            btype='full_body',
            out_file='/home/istvanfe/Downloads/model_output/anydoor/v1/',
            scale=scale,
        )

        run_inference(
            tar_image_path='/media/istvanfe/MLStuff/Datasets/marketing/raw-images/model-3296470.jpg',
            ref_image_path='/media/istvanfe/MLStuff/Datasets/Scraped/aritzia/woman/dresses/lucille_dress_112826_1274/f23_04_a08_112826_1274_on_a.jpg',
            btype='full_body',
            out_file='/home/istvanfe/Downloads/model_output/anydoor/v1/',
            scale=scale,
        )

        # model-5277469.jpg
        run_inference(
            tar_image_path='/media/istvanfe/MLStuff/Datasets/marketing/raw-images/model-5277469.jpg',
            ref_image_path='/media/istvanfe/MLStuff/Datasets/Scraped/aritzia/woman/dresses/shimmer_satin_tube_dress_108165_1274/f23_02_a08_108165_1274_on_b.jpg',
            btype='full_body',
            out_file='/home/istvanfe/Downloads/model_output/anydoor/v1/',
            scale=scale,
        )

        run_inference(
            tar_image_path='/media/istvanfe/MLStuff/Datasets/marketing/raw-images/model-5277469.jpg',
            ref_image_path='/media/istvanfe/MLStuff/Datasets/Scraped/aritzia/woman/dresses/larchmont_satin_midi_dress_114575_15095/f23_01_a08_114575_15095_on_a.jpg',
            btype='full_body',
            out_file='/home/istvanfe/Downloads/model_output/anydoor/v1/',
            scale=scale,
        )

        # smile-1275668.jpg
        run_inference(
            tar_image_path='/media/istvanfe/MLStuff/Datasets/marketing/raw-images/smile-1275668.jpg',
            ref_image_path='/media/istvanfe/MLStuff/Datasets/Scraped/aritzia/woman/dresses/tempest_mini_dress_98811_11420/s23_04_a08_98811_11420_on_a.jpg',
            btype='full_body',
            out_file='/home/istvanfe/Downloads/model_output/anydoor/v1/',
            scale=scale,
        )

        run_inference(
            tar_image_path='/media/istvanfe/MLStuff/Datasets/marketing/raw-images/smile-1275668.jpg',
            ref_image_path='/media/istvanfe/MLStuff/Datasets/Scraped/aritzia/woman/dresses/hyde_satin_mini_dress_115693_15104/f23_01_a08_115693_15104_on_a.jpg',
            btype='full_body',
            out_file='/home/istvanfe/Downloads/model_output/anydoor/v1/',
            scale=scale,
        )

        # woman-4246938.jpg
        run_inference(
            tar_image_path='/media/istvanfe/MLStuff/Datasets/marketing/raw-images/woman-4246938.jpg',
            ref_image_path='/media/istvanfe/MLStuff/Datasets/Scraped/aritzia/woman/dresses/only_slip_mini_dress_81395_160/s23_04_a08_81395_160_on_a.jpg',
            btype='full_body',
            out_file='/home/istvanfe/Downloads/model_output/anydoor/v1/',
            scale=scale,
        )

        run_inference(
            tar_image_path='/media/istvanfe/MLStuff/Datasets/marketing/raw-images/woman-4246938.jpg',
            ref_image_path='/media/istvanfe/MLStuff/Datasets/Scraped/aritzia/woman/blouses/poppy_poplin_shirt_112819_1275/s23_02_a02_112819_1275_on_a.jpg',
            btype='upper_body',
            out_file='/home/istvanfe/Downloads/model_output/anydoor/v1/',
            scale=scale,
        )

        # woman-4390055.jpg
        run_inference(
            tar_image_path='/media/istvanfe/MLStuff/Datasets/marketing/raw-images/woman-4390055.jpg',
            ref_image_path='/media/istvanfe/MLStuff/Datasets/Scraped/aritzia/woman/dresses/secret_mini_slip_dress_114459_11420/f23_04_a08_114459_1274_on_d.jpg',
            btype='full_body',
            out_file='/home/istvanfe/Downloads/model_output/anydoor/v1/',
            scale=scale,
        )

        run_inference(
            tar_image_path='/media/istvanfe/MLStuff/Datasets/marketing/raw-images/woman-4390055.jpg',
            ref_image_path='/media/istvanfe/MLStuff/Datasets/Scraped/aritzia/woman/blouses/hyde_satin_blouse_113992_25167/f23_01_a02_113992_25167_on_a.jpg',
            btype='upper_body',
            out_file='/home/istvanfe/Downloads/model_output/anydoor/v1/',
            scale=scale,
        )

        # woman-4672683.jpg
        run_inference(
            tar_image_path='/media/istvanfe/MLStuff/Datasets/marketing/raw-images/woman-4672683.jpg',
            ref_image_path='/media/istvanfe/MLStuff/Datasets/Scraped/aritzia/woman/blouses/sail_oxford_shirt_112107_31570/f23_07_a02_112107_31570_on_b.jpg',
            btype='upper_body',
            out_file='/home/istvanfe/Downloads/model_output/anydoor/v1/',
            scale=scale,
        )

        run_inference(
            tar_image_path='/media/istvanfe/MLStuff/Datasets/marketing/raw-images/woman-4672683.jpg',
            ref_image_path='/media/istvanfe/MLStuff/Datasets/Scraped/aritzia/woman/blouses/kaia_top_108399_1275/s23_02_a01_108399_1275_on_a.jpg',
            btype='upper_body',
            out_file='/home/istvanfe/Downloads/model_output/anydoor/v1/',
            scale=scale,
        )

        # woman-4707545.jpg
        run_inference(
            tar_image_path='/media/istvanfe/MLStuff/Datasets/marketing/raw-images/woman-4707545.jpg',
            ref_image_path='/media/istvanfe/MLStuff/Datasets/Scraped/aritzia/woman/blouses/relaxed_shirt_102183_28813/f23_07_a02_102183_28813_on_d.jpg',
            btype='upper_body',
            out_file='/home/istvanfe/Downloads/model_output/anydoor/v1/',
            scale=scale,
        )

        run_inference(
            tar_image_path='/media/istvanfe/MLStuff/Datasets/marketing/raw-images/woman-4707545.jpg',
            ref_image_path='/media/istvanfe/MLStuff/Datasets/Scraped/aritzia/woman/blouses/sail_shirt_103791_28815/f23_07_a02_103791_28815_on_c.jpg',
            btype='upper_body',
            out_file='/home/istvanfe/Downloads/model_output/anydoor/v1/',
            scale=scale,
        )

        # woman-5679284.jpg
        run_inference(
            tar_image_path='/media/istvanfe/MLStuff/Datasets/marketing/raw-images/woman-5679284.jpg',
            ref_image_path='/media/istvanfe/MLStuff/Datasets/Scraped/aritzia/woman/dresses/larchmont_satin_dress_114213_29484/f23_01_a08_114213_29484_on_a.jpg',
            btype='full_body',
            out_file='/home/istvanfe/Downloads/model_output/anydoor/v1/',
            scale=scale,
        )

        run_inference(
            tar_image_path='/media/istvanfe/MLStuff/Datasets/marketing/raw-images/woman-5679284.jpg',
            ref_image_path='/media/istvanfe/MLStuff/Datasets/Scraped/aritzia/woman/dresses/shimmer_satin_dress_99544_2198/s23_02_a08_99544_2198_on_b.jpg',
            btype='full_body',
            out_file='/home/istvanfe/Downloads/model_output/anydoor/v1/',
            scale=scale,
        )

        # woman-6851973.jpg
        run_inference(
            tar_image_path='/media/istvanfe/MLStuff/Datasets/marketing/raw-images/woman-6851973.jpg',
            ref_image_path='/media/istvanfe/MLStuff/Datasets/Scraped/aritzia/woman/dresses/contour_tube_midi_dress_117216_1274/f23_01_a08_117216_1274_on_a.jpg',
            btype='full_body',
            out_file='/home/istvanfe/Downloads/model_output/anydoor/v1/',
            scale=scale,
        )

        run_inference(
            tar_image_path='/media/istvanfe/MLStuff/Datasets/marketing/raw-images/woman-6851973.jpg',
            ref_image_path='/media/istvanfe/MLStuff/Datasets/Scraped/aritzia/woman/dresses/genoa_dress_109461_1274/s23_04_a08_109461_1274_on_a.jpg',
            btype='full_body',
            out_file='/home/istvanfe/Downloads/model_output/anydoor/v1/',
            scale=scale,
        )
