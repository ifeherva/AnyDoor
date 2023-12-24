from typing import Union

import numpy
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation

from datasets.data_utils import get_palette


class ClothSegmenterHF:

    def __init__(self, model_path: str = '/home/istvanfe/Work/SLiMe/v1_B2_ATR5/5', device='cuda'):
        self.tokenizer = SegformerImageProcessor.from_pretrained(model_path)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(model_path).to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def __call__(self,
                 image: Union[str, np.ndarray, Image.Image],
                 pad2square=False,
                 input_size=None) -> np.ndarray:
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        item = self.tokenizer(image)

        inputs = torch.from_numpy(item.data["pixel_values"][0]).to(self.device).unsqueeze(0)
        outputs = self.model(inputs)

        logits = nn.functional.interpolate(
            outputs.logits,  # Detach to avoid saving gradients
            size=[image.height, image.width],
            mode="bilinear",
            align_corners=False,
        )

        pred_labels = logits.argmax(dim=1).cpu().numpy()[0].astype(np.uint8)

        return pred_labels


def draw_semantic_mask(image: Image.Image, mask: numpy.ndarray, label_format='atr'):
    palette = get_palette(11)  # ATR5

    mask_alpha = (mask != 0).astype(np.uint8)
    mask_alpha *= int(255 * 0.5)
    mask_alpha = Image.fromarray(mask_alpha, mode="L")

    mask = Image.fromarray(mask)
    mask.putpalette(palette)
    mask = mask.convert('RGBA')
    mask.putalpha(mask_alpha)

    image = image.copy()
    image = image.convert('RGBA')

    image = Image.alpha_composite(image, mask)
    image = image.convert("RGB")
    return image
