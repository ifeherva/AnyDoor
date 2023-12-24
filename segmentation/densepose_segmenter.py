from typing import Union

import cv2
import numpy as np
import torch
from PIL import Image
from densepose import add_densepose_config
from densepose.vis.extractor import DensePoseResultExtractor
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image, _apply_exif_orientation, convert_PIL_to_numpy
from detectron2.engine import DefaultPredictor
from huggingface_hub import hf_hub_download

from datasets.data_utils import get_palette

REPO_ID = "GooKSL/densepose"
BASE_CONFIG_FILENAME = 'Base-DensePose-RCNN-FPN.yaml'
CONFIG_FILENAME = 'densepose_rcnn_R_101_FPN_DL_s1x.yaml'
MODEL_FILENAME = 'model_final_844d15.pkl'


class DenseposeSegmenter:

    def __init__(self):
        hf_hub_download(repo_id=REPO_ID, filename=BASE_CONFIG_FILENAME)
        config_fpath = hf_hub_download(repo_id=REPO_ID, filename=CONFIG_FILENAME)
        model_fpath = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)

        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file(config_fpath)
        cfg.MODEL.WEIGHTS = model_fpath
        cfg.freeze()

        self.predictor = DefaultPredictor(cfg)
        self.extractor = DensePoseResultExtractor()

        self.interp_method_mask = cv2.INTER_NEAREST
        self.interp_method_matrix = cv2.INTER_LINEAR

    def _resize(self, mask, matrix, w, h):
        if (w != mask.shape[1]) or (h != mask.shape[0]):
            mask = cv2.resize(mask, (w, h), self.interp_method_mask)
        if (w != matrix.shape[1]) or (h != matrix.shape[0]):
            matrix = cv2.resize(matrix, (w, h), self.interp_method_matrix)
        return mask, matrix

    def detect_pose(self, image: Union[str, np.ndarray, Image.Image]):
        if isinstance(image, str):
            img = read_image(image, format="BGR")  # predictor expects BGR image.
        elif isinstance(image, Image.Image):
            img = _apply_exif_orientation(image)
            img = convert_PIL_to_numpy(img, format="BGR")
        else:
            raise RuntimeError('channel swapping needs to be implemented for numpy input')

        with torch.no_grad():
            outputs = self.predictor(img)["instances"]

        output = self.extractor(outputs)

        densepose_result, boxes_xywh = output
        assert densepose_result is not None and boxes_xywh is not None
        result, bbox_xywh = densepose_result[0], boxes_xywh.cpu().numpy()[0]
        iuv_array = result.labels[None].type(torch.uint8).cpu().numpy()
        matrix = iuv_array[0, :, :]
        segm = iuv_array[0, :, :]
        mask = np.zeros(matrix.shape, dtype=np.uint8)
        mask[segm > 0] = 1
        x, y, w, h = [int(v) for v in bbox_xywh]
        mask, matrix = self._resize(mask, matrix, w, h)
        mask_bg = np.tile((mask == 0)[:, :, np.newaxis], [1, 1, 3])
        image_target_bgr = np.zeros_like(img)
        matrix_vis = np.tile(matrix[:, :, np.newaxis], [1, 1, 3])
        matrix_vis[mask_bg] = image_target_bgr[y:y + h, x:x + w, :][mask_bg]
        alpha = 1.0
        image_target_bgr[y:y + h, x:x + w, :] = (image_target_bgr[y:y + h, x:x + w, :] *
                                                 (1.0 - alpha) + matrix_vis * alpha)

        uv_bgr = np.zeros((2, img.shape[0], img.shape[1]), dtype=np.float32)
        uv = result.uv.cpu().numpy()  # 2xHxW
        uv_bgr[:, y:y + h, x:x + w] = uv

        return image_target_bgr, uv_bgr

    @torch.no_grad()
    def __call__(self, image: Union[str, np.ndarray, Image.Image]):
        image_target_bgr, uv_bgr = self.detect_pose(image)

        return image_target_bgr  # HxWx3


def draw_densepose(image: Image.Image, mask: np.ndarray):
    palette = get_palette(25)

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


if __name__ == '__main__':
    segmenter = DenseposeSegmenter()
    segmenter('/media/istvanfe/MLStuff/Datasets/Redress/full_body/images/0000000/0000000_00.jpg')
