import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
from .data_utils import * 
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import albumentations as A


class BaseDataset(Dataset):
    def __init__(self, dynamic=2):
        self.dynamic = dynamic

    def __len__(self):
        # We adjust the ratio of different dataset by setting the length.
        raise NotImplementedError('len() must be implemented by subclass')

    def aug_data_back(self, image):
        transform = A.Compose([
            A.ColorJitter(p=0.5, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            A.ChannelShuffle()
            ])
        transformed = transform(image=image.astype(np.uint8))
        transformed_image = transformed["image"]
        return transformed_image
    
    def aug_data_mask(self, image, mask):
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            #A.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT,  value=(0,0,0)),
            ])

        transformed = transform(image=image.astype(np.uint8), mask=mask)
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]
        return transformed_image, transformed_mask

    def check_region_size(self, image, yyxx, ratio, mode = 'max'):
        pass_flag = True
        H,W = image.shape[0], image.shape[1]
        H,W = H * ratio, W * ratio
        y1,y2,x1,x2 = yyxx
        h,w = y2-y1,x2-x1
        if mode == 'max':
            if h > H or w > W:
                pass_flag = False
        elif mode == 'min':
            if h < H or w < W:
                pass_flag = False
        return pass_flag

    def __getitem__(self, idx):
        while True:
            try:
                idx = np.random.choice(range(len(self.pairs)), 1)[0]
                item = self.get_sample(idx)
                return item
            except Exception as a:
                idx = np.random.choice(range(len(self.pairs)), 1)[0]
                
    def get_sample(self, idx):
        # Implemented for each specific dataset
        raise NotImplementedError('get_sample() must be implemented by subclass')

    def sample_timestep(self, max_step =1000):
        if np.random.rand() < 0.3:
            step = np.random.randint(0,max_step)
            return np.array([step])

        if self.dynamic == 1:
            # coarse videos
            step_start = max_step // 2
            step_end = max_step
        elif self.dynamic == 0:
            # static images
            step_start = 0 
            step_end = max_step // 2
        else:
            # fine multi-view images/videos/3Ds
            step_start = 0
            step_end = max_step
        step = np.random.randint(step_start, step_end)
        return np.array([step])

    @staticmethod
    def check_mask_area(mask):
        height, width = mask.shape[0], mask.shape[1]
        ratio = mask.sum() / (height * width)
        if ratio > 0.8 * 0.8 or ratio < 0.1 * 0.1:
            return False
        else:
            return True 

    def process_pairs(self, ref_image, ref_mask, tar_image, tar_mask, max_ratio=0.8,
                      ref_dp_mask=None, tar_dp_mask=None, do_aug=True,
                      face_mask=None, hand_mask=None):
        assert mask_score(ref_mask) > 0.50
        assert self.check_mask_area(ref_mask)
        assert self.check_mask_area(tar_mask)

        # ========= Reference ===========

        # Get the outline box of the reference image
        ref_box_yyxx = get_bbox_from_mask(ref_mask)
        assert self.check_region_size(ref_mask, ref_box_yyxx, ratio=0.10, mode='min')
        
        # Filtering background for the reference image
        ref_mask_3 = np.stack([ref_mask, ref_mask, ref_mask], -1)
        masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1-ref_mask_3)

        y1, y2, x1, x2 = ref_box_yyxx
        masked_ref_image = masked_ref_image[y1:y2, x1:x2, :]
        ref_mask = ref_mask[y1:y2, x1:x2]

        ratio = np.random.randint(11, 15) / 10 
        masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=ratio)
        ref_mask_3 = np.stack([ref_mask, ref_mask, ref_mask], -1)

        # Padding reference image to square and resize to 224
        masked_ref_image = pad_to_square(masked_ref_image, pad_value=255, random=False)
        masked_ref_image = cv2.resize(masked_ref_image.astype(np.uint8), (224, 224)).astype(np.uint8)

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
        ref_mask_3 = cv2.resize(ref_mask_3.astype(np.uint8), (224, 224)).astype(np.uint8)
        ref_mask = ref_mask_3[:, :, 0]

        # Getting for high-freqency map
        if do_aug:
            masked_ref_image_compose, ref_mask_compose = self.aug_data_mask(masked_ref_image, ref_mask)
            masked_ref_image_aug = masked_ref_image_compose.copy()
        else:
            masked_ref_image_compose, ref_mask_compose = masked_ref_image, ref_mask
            masked_ref_image_aug = masked_ref_image_compose.copy()

        ref_image_collage = sobel(masked_ref_image_compose, ref_mask_compose/255)

        # ========= Training Target ===========
        tar_box_yyxx = get_bbox_from_mask(tar_mask)
        tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=[1.1, 1.2])  # 1.1  1.3
        assert self.check_region_size(tar_mask, tar_box_yyxx, ratio=max_ratio, mode='max')

        unmask_mask = np.zeros_like(tar_image[:, :, 0]).astype(np.uint8)
        if hand_mask is not None:
            unmask_mask = unmask_mask + hand_mask
        if face_mask is not None:
            unmask_mask = unmask_mask + face_mask
        unmask_mask = unmask_mask.clip(0, 1)

        # Cropping around the target object
        tar_box_yyxx_crop = expand_bbox(tar_image, tar_box_yyxx, ratio=[1.3, 3.0])
        tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop)  # crop box
        y1, y2, x1, x2 = tar_box_yyxx_crop
        cropped_target_image = tar_image[y1:y2, x1:x2, :]
        if tar_dp_mask is not None:
            cropped_target_dp_mask = tar_dp_mask[y1:y2, x1:x2]
        else:
            cropped_target_dp_mask = None

        unmask_mask = unmask_mask[y1:y2, x1:x2]
        cropped_tar_mask = tar_mask[y1:y2, x1:x2]

        tar_box_yyxx = box_in_box(tar_box_yyxx, tar_box_yyxx_crop)
        y1, y2, x1, x2 = tar_box_yyxx

        # Preparing collage image
        ref_image_collage = cv2.resize(ref_image_collage.astype(np.uint8), (x2-x1, y2-y1))
        # ref_mask_compose = cv2.resize(ref_mask_compose.astype(np.uint8), (x2-x1, y2-y1))
        # ref_mask_compose = (ref_mask_compose > 128).astype(np.uint8)

        collage = cropped_target_image.copy()
        collage[y1:y2, x1:x2, :] = ref_image_collage

        collage_mask = np.zeros_like(cropped_target_image)
        collage_mask[y1:y2, x1:x2, :] = 1.0

        if np.random.uniform(0, 1) < 0.7: 
            cropped_tar_mask = perturb_mask(cropped_tar_mask)
            collage_mask = np.stack([cropped_tar_mask, cropped_tar_mask, cropped_tar_mask], -1)

        H1, W1 = collage.shape[0], collage.shape[1]

        cropped_target_image = pad_to_square(cropped_target_image, pad_value=0, random=False).astype(np.uint8)
        collage = pad_to_square(collage, pad_value=0, random=False).astype(np.uint8)
        collage_mask = pad_to_square(collage_mask, pad_value=2, random=False).astype(np.uint8)
        H2, W2 = collage.shape[0], collage.shape[1]

        cropped_target_image = cv2.resize(cropped_target_image.astype(np.uint8), (512, 512)).astype(np.float32)
        collage = cv2.resize(collage.astype(np.uint8), (512, 512)).astype(np.float32)
        collage_mask = cv2.resize(collage_mask.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        collage_mask[collage_mask == 2] = -1

        if cropped_target_dp_mask is not None:
            cropped_target_dp_mask = pad_to_square(cropped_target_dp_mask, pad_value=0, random=False).astype(np.uint8)
            cropped_target_dp_mask = cv2.resize(cropped_target_dp_mask.astype(np.uint8), (512, 512),
                                                interpolation=cv2.INTER_NEAREST).astype(np.float32)
            cropped_target_dp_mask = cropped_target_dp_mask / 24  # number of DP classes -1

        unmask_mask = pad_to_square(unmask_mask, pad_value=0, random=False).astype(np.uint8)
        unmask_mask = cv2.resize(unmask_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        erode_radius = 3
        unmask_mask = cv2.erode(unmask_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                                       (2 * erode_radius + 1, 2 * erode_radius + 1))).astype('bool')
        collage = np.where(np.expand_dims(unmask_mask, 2), cropped_target_image, collage)
        collage_mask = collage_mask[:, :, 1]
        collage_mask[unmask_mask] = 0
        
        # Preparing dataloader items
        masked_ref_image_aug = masked_ref_image_aug / 255
        cropped_target_image = cropped_target_image / 127.5 - 1.0
        collage = collage / 127.5 - 1.0 
        collage = np.concatenate([collage, np.expand_dims(collage_mask, 2)], -1)

        if cropped_target_dp_mask is not None:
            collage = np.concatenate([collage, cropped_target_dp_mask[:, :, None]], -1)

        if ref_dp_mask_onehot is not None:
            masked_ref_image_aug = np.concatenate([masked_ref_image_aug, ref_dp_mask_onehot], -1)
        if tar_dp_mask_onehot is not None:
            masked_ref_image_aug = np.concatenate([masked_ref_image_aug, tar_dp_mask_onehot], -1)
        
        item = dict(
                ref=masked_ref_image_aug.copy(),  # for dino
                jpg=cropped_target_image.copy(),  # ground truth
                hint=collage.copy(),  # Collage for controlnet
                extra_sizes=np.array([H1, W1, H2, W2]), 
                tar_box_yyxx_crop=np.array(tar_box_yyxx_crop),
                )

        return item
