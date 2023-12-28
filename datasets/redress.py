from os.path import join
from typing import Union, Optional

import cv2
from PIL import Image
import numpy as np

from datasets.base import BaseDataset
from segmentation.face_segmenter import FaceSegmenter

ATR5_MAP = {
    'background': 0,
    'hair': 1,
    'face': 2,
    'upper_body_clothing': 3,
    'lower_body_clothing': 4,
    'arms': 5,
    'legs': 6,
    'scarf': 7,
    'belt': 8,
    'shoes': 9,
    'bag': 10,
}

DENSEPOSE_MAP = {
    'background': 0,
    'torso_front': 1,
    'torso_back': 2,
    'right_hand': 3,
    'left_hand': 4,
    'right_foot': 5,
    'left_foot': 6,
    'upper_leg_right_front': 7,
    'upper_leg_left_front': 8,
    'upper_leg_right_back': 9,
    'upper_leg_left_back': 10,
    'lower_leg_right_front': 11,
    'lower_leg_left_front': 12,
    'lower_leg_right_back': 13,
    'lower_leg_left_back': 14,
    'upper_arm_left_front': 15,
    'upper_arm_right_front': 16,
    'upper_arm_left_back': 17,
    'upper_arm_right_back': 18,
    'lower_arm_left_front': 19,
    'lower_arm_right_front': 20,
    'lower_arm_left_back': 21,
    'lower_arm_right_back': 22,
    'head_front': 23,
    'head_back': 24,
}

MASK_GARMENT_ITEMS = {
    'full_body': [
        ATR5_MAP['upper_body_clothing'],
        ATR5_MAP['lower_body_clothing'],
        ATR5_MAP['belt'],
    ],
    'upper_body': [
        ATR5_MAP['upper_body_clothing'],
    ],
    'lower_body': [
        ATR5_MAP['lower_body_clothing'],
        ATR5_MAP['belt'],
    ],
}

MASK_TARGET_ITEMS = {
    'full_body': [
        ATR5_MAP['upper_body_clothing'],
        ATR5_MAP['lower_body_clothing'],
        ATR5_MAP['belt'],
    ],
    'upper_body': [
        ATR5_MAP['upper_body_clothing'],
    ],
}


DP_MASK_TARGET_ITEMS = {
    'full_body': [
        # We exclude hands and head

        # torso
        DENSEPOSE_MAP['torso_front'],
        DENSEPOSE_MAP['torso_back'],

        # Arms
        DENSEPOSE_MAP['upper_arm_left_front'],
        DENSEPOSE_MAP['upper_arm_left_back'],
        DENSEPOSE_MAP['upper_arm_right_front'],
        DENSEPOSE_MAP['upper_arm_right_back'],

        DENSEPOSE_MAP['lower_arm_left_front'],
        DENSEPOSE_MAP['lower_arm_left_back'],
        DENSEPOSE_MAP['lower_arm_right_front'],
        DENSEPOSE_MAP['lower_arm_right_back'],

        # Legs
        DENSEPOSE_MAP['upper_leg_left_front'],
        DENSEPOSE_MAP['upper_leg_left_back'],
        DENSEPOSE_MAP['upper_leg_right_front'],
        DENSEPOSE_MAP['upper_leg_right_back'],

        DENSEPOSE_MAP['lower_leg_left_front'],
        DENSEPOSE_MAP['lower_leg_left_back'],
        DENSEPOSE_MAP['lower_leg_right_front'],
        DENSEPOSE_MAP['lower_leg_right_back'],

    ],
    'upper_body': [
        DENSEPOSE_MAP['torso_front'],
        DENSEPOSE_MAP['torso_back'],

        # Arms
        DENSEPOSE_MAP['upper_arm_left_front'],
        DENSEPOSE_MAP['upper_arm_left_back'],
        DENSEPOSE_MAP['upper_arm_right_front'],
        DENSEPOSE_MAP['upper_arm_right_back'],

        DENSEPOSE_MAP['lower_arm_left_front'],
        DENSEPOSE_MAP['lower_arm_left_back'],
        DENSEPOSE_MAP['lower_arm_right_front'],
        DENSEPOSE_MAP['lower_arm_right_back'],
    ],
    # 'lower_body': [
    #     ATR5_MAP['lower_body_clothing'],
    #     ATR5_MAP['belt'],
    # ],
}


class RedressDataset(BaseDataset):
    def __init__(self,
                 root: str,
                 num_devices: int = 1,
                 batch_size: int = 8,
                 split='train',
                 body_types=('full_body', 'upper_body'),
                 augment: Optional[bool] = None,
                 use_face_detect=False,
                 ):
        super().__init__()
        self.root = root
        self.pairs = []
        self.split = split
        self.num_devices = num_devices
        self.batch_size = batch_size
        self.augment = augment or (split == 'train')

        for btype in body_types:
            broot = join(self.root, btype)
            with open(join(broot, '{}_split.txt'.format(split)), 'r') as f:
                self.pairs += [(x.strip().split(','), btype)
                               for x in f.readlines()
                               if len(x.strip().split(',')) == 2]

        self.face_segmenter = FaceSegmenter() if use_face_detect else None

    def __len__(self) -> int:
        #return len(self.pairs)
        # this is fixed so we get shorter epochs
        return min(10000 * self.num_devices * self.batch_size, len(self.pairs))

    def process_sample(
            self,
            ref_image: Union[str, np.ndarray, Image.Image],
            ref_mask: Union[str, np.ndarray, Image.Image],
            ref_dp_mask: Union[str, np.ndarray, Image.Image],
            tar_image: Union[str, np.ndarray, Image.Image],
            tar_mask: Union[str, np.ndarray, Image.Image],
            tar_dp_mask: Union[str, np.ndarray, Image.Image],
            btype: str,
    ):
        assert btype in ['full_body', 'upper_body']

        # Prepare reference image
        if isinstance(ref_image, str):
            ref_image = cv2.imread(ref_image)  # BGR
            ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)  # HxWx3 - RGB
        elif isinstance(ref_image, Image.Image):
            ref_image = np.array(ref_image)
        else:
            assert ref_image.ndim == 3  # must be HxWxC
            assert ref_image.shape[-1] == 3  # must be RGB

        # Prepare reference mask
        if isinstance(ref_mask, str):
            ref_mask = np.array(Image.open(ref_mask).convert('P'))  # HxW
        elif isinstance(ref_mask, Image.Image):
            if ref_mask.mode != 'P':
                ref_mask = ref_mask.convert('P')
            ref_mask = np.array(ref_mask)
        else:
            assert ref_mask.ndim == 2  # must be HxW
        ref_mask = np.isin(ref_mask, MASK_GARMENT_ITEMS[btype]).astype(np.uint8)

        # Prepare reference densepose
        if isinstance(ref_dp_mask, str):
            ref_dp_mask = np.array(Image.open(ref_dp_mask).convert('L'))  # HxW
        elif isinstance(ref_dp_mask, Image.Image):
            if ref_dp_mask.mode != 'L':
                ref_dp_mask = ref_dp_mask.convert('L')
            ref_dp_mask = np.array(ref_dp_mask)
        else:
            if ref_dp_mask.ndim == 3:
                ref_dp_mask = ref_dp_mask[..., 0]
            assert ref_dp_mask.ndim == 2  # must be HxW

        # Prepare target image
        if isinstance(tar_image, str):
            tar_image = cv2.imread(tar_image)  # BGR
            tar_image = cv2.cvtColor(tar_image, cv2.COLOR_BGR2RGB)  # HxWx3 - RGB
        elif isinstance(tar_image, Image.Image):
            tar_image = np.array(tar_image)
        else:
            assert tar_image.ndim == 3  # must be HxWxC
            assert tar_image.shape[-1] == 3  # must be RGB

        # Load target mask
        if isinstance(tar_mask, str):
            tar_mask = np.array(Image.open(tar_mask).convert('P'))  # HxW
        elif isinstance(ref_image, Image.Image):
            if tar_mask.mode != 'P':
                tar_mask = tar_mask.convert('P')
            tar_mask = np.array(tar_mask)
        else:
            assert tar_mask.ndim == 2  # must be HxW

        # Prepare target densepose
        if isinstance(tar_dp_mask, str):
            tar_dp_mask = np.array(Image.open(tar_dp_mask).convert('L'))  # HxW
        elif isinstance(tar_dp_mask, Image.Image):
            if tar_dp_mask.mode != 'L':
                tar_dp_mask = tar_dp_mask.convert('L')
            tar_dp_mask = np.array(tar_dp_mask)
        else:
            if tar_dp_mask.ndim == 3:
                tar_dp_mask = tar_dp_mask[..., 0]
            assert tar_dp_mask.ndim == 2  # must be HxW

        # We use the atr_mask+densepose to determine where to insert
        tar_mask_dp = np.isin(tar_dp_mask, DP_MASK_TARGET_ITEMS[btype])
        tar_mask_atr = np.isin(tar_mask, MASK_TARGET_ITEMS[btype])

        face_mask = np.logical_or(
            np.isin(tar_dp_mask, [23, 24]),
            np.isin(tar_mask_atr, 2)
        ).astype(np.uint8)

        hand_mask = np.logical_and(
            np.isin(tar_dp_mask, [3, 4]),
            np.isin(tar_mask, 5)
        ).astype(np.uint8)

        if self.face_segmenter is not None:
            face_mask_square = self.face_segmenter(tar_image)
            face_mask = face_mask * face_mask_square

        tar_mask = np.logical_or(tar_mask_dp, tar_mask_atr).astype(np.uint8)

        return ref_image, ref_mask, tar_image, tar_mask, ref_dp_mask, tar_dp_mask, face_mask, hand_mask

    def get_sample(self, index):
        """
        Reference is what we paste, target is where we paste it
        :param index:
        :return:
        """
        pair, btype = self.pairs[index]
        root = join(self.root, btype)

        # randomly reverse
        if self.split == 'train':
            np.random.shuffle(pair)

        ref_image_name, tar_image_name = pair
        ref_image, ref_mask, tar_image, tar_mask, ref_dp_mask, tar_dp_mask, face_mask, hand_mask = self.process_sample(
            join(root, 'images', ref_image_name),
            join(root, 'atr5_labels', ref_image_name.replace('jpg', 'png')),
            join(root, 'densepose', ref_image_name.replace('jpg', 'png')),
            join(root, 'images', tar_image_name),
            join(root, 'atr5_labels', tar_image_name.replace('jpg', 'png')),
            join(root, 'densepose', tar_image_name.replace('jpg', 'png')),
            btype,
        )

        item_with_collage = self.process_pairs(
            ref_image, ref_mask,
            tar_image, tar_mask,
            max_ratio=1.0,
            ref_dp_mask=ref_dp_mask,
            tar_dp_mask=tar_dp_mask,
            do_aug=self.augment,
            face_mask=face_mask,
            hand_mask=hand_mask,
        )
        sampled_time_steps = self.sample_timestep()
        item_with_collage['time_steps'] = sampled_time_steps
        return item_with_collage


if __name__ == '__main__':
    ds = RedressDataset('/media/istvanfe/MLStuff/Datasets/Redress')
    ds.get_sample(0)