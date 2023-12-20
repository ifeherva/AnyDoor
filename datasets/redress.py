from os.path import join

import cv2
from PIL import Image
import numpy as np

from datasets.base import BaseDataset


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
                 split='train',
                 body_types=('full_body', 'upper_body')):
        super().__init__()
        self.root = root
        self.pairs = []
        self.split = split

        for btype in body_types:
            broot = join(self.root, btype)
            with open(join(broot, '{}_split.txt'.format(split)), 'r') as f:
                self.pairs += [(x.strip().split(','), btype)
                               for x in f.readlines()
                               if len(x.strip().split(',')) == 2]

    def __len__(self) -> int:
        return len(self.pairs)

    def get_sample(self, index):
        pair, btype = self.pairs[index]
        root = join(self.root, btype)

        # randomly reverse
        if self.split == 'train':
            np.random.shuffle(pair)

        ref_image_name, tar_image_name = pair  # reference is what we paste, target is where we paste it

        # Load reference image
        ref_image_path = join(root, 'images', ref_image_name)
        ref_image = cv2.imread(ref_image_path)  # BGR
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)  # HxWx3 - RGB

        # Load reference mask
        ref_mask_path = join(root, 'atr5_labels', ref_image_name.replace('jpg', 'png'))
        ref_mask = np.array(Image.open(ref_mask_path).convert('P'))  # HxW
        ref_mask = np.isin(ref_mask, MASK_GARMENT_ITEMS[btype]).astype(np.uint8)

        # Load reference densepose
        ref_dp_mask_path = join(root, 'densepose', ref_image_name.replace('jpg', 'png'))
        ref_dp_mask = np.array(Image.open(ref_dp_mask_path).convert('L'))  # HxW

        # Load target image
        tar_image_path = join(root, 'images', tar_image_name)
        tar_image = cv2.imread(tar_image_path)  # BGR
        tar_image = cv2.cvtColor(tar_image, cv2.COLOR_BGR2RGB)  # HxWx3 - RGB

        # Load target densepose
        tar_dp_mask_path = join(root, 'densepose', tar_image_name.replace('jpg', 'png'))
        tar_dp_mask = np.array(Image.open(tar_dp_mask_path).convert('L'))  # HxW

        # Load target mask
        tar_mask_path = join(root, 'atr5_labels', tar_image_name.replace('jpg', 'png'))
        # tar_mask = np.array(Image.open(tar_mask_path).convert('P'))  # HxW
        # We use the densepose to determine where to insert
        tar_mask = np.isin(tar_dp_mask, DP_MASK_TARGET_ITEMS[btype]).astype(np.uint8)

        item_with_collage = self.process_pairs(ref_image, ref_mask, tar_image, tar_mask, max_ratio=1.0,
                                               ref_dp_mask=ref_dp_mask, tar_dp_mask=tar_dp_mask)
        sampled_time_steps = self.sample_timestep()
        item_with_collage['time_steps'] = sampled_time_steps
        return item_with_collage


if __name__ == '__main__':
    ds = RedressDataset('/media/istvanfe/MLStuff/Datasets/Redress')
    ds.get_sample(0)