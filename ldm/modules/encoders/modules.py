import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Dinov2Model, CLIPProcessor, CLIPVisionModelWithProjection

from ldm.modules.diffusionmodules.util import conv_nd


class LayerNormFp32(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16 (by casting to float32 and back)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x.to(torch.float32), self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

    def freeze(self):
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False


class IdentityEncoder(AbstractEncoder):

    def encode(self, x):
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1):
        super().__init__()

        self.conv = conv_nd(2, in_channels, out_channels,
                            kernel_size, padding=padding, stride=stride)
        self.act = nn.SiLU()

    def forward(self, x):
        res = x
        x = self.conv(x)
        x = self.act(x)
        x = x + res
        return x


class FrozenDinoV2Encoder(AbstractEncoder):
    """
    Uses the DINOv2 encoder for image
    """
    def __init__(self,
                 freeze=True,
                 use_densepose=False,
                 ):
        super().__init__()
        self.model = Dinov2Model.from_pretrained("facebook/dinov2-giant")

        if freeze:
            self.freeze()

        self.register_buffer('image_mean', torch.tensor([0.485, 0.456, 0.406]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
        self.register_buffer('image_std', torch.tensor([0.229, 0.224, 0.225]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1))

        self.projector = nn.Linear(1536, 1024)
        self.use_densepose = use_densepose

        if use_densepose:
            # Make it separate class, add res connection, 4 blocks
            self.pose_projector = nn.Sequential(
                conv_nd(2, 25, 128, 7, padding=3, stride=2),
                nn.SiLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 56x56
                ResBlock(128, 128, 3, padding=1, stride=1),  # 56x56
                conv_nd(2, 128, 256, 3, padding=1, stride=2),  # 28x28
                nn.SiLU(),
                ResBlock(256, 256, 3, padding=1, stride=1),  # 28x28
                conv_nd(2, 256, 512, 3, padding=1, stride=2),  # 14x14
                nn.SiLU(),
                ResBlock(512, 512, 3, padding=1, stride=1),  # 14x14
                conv_nd(2, 512, 1024, 3, padding=1, stride=2),  # 7x7
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )

    def forward(self, image):
        if isinstance(image, list):
            image = torch.cat(image, 0)

        if self.use_densepose:
            image, dp_ref, dp_tar = image.split((3, 25, 25), 1)
            dense_proj = torch.cat([
                self.pose_projector(dp_ref).unsqueeze(1),  # Bx1x1024
                self.pose_projector(dp_tar).unsqueeze(1),  # Bx1x1024
            ], 1)

        with torch.no_grad():
            image = (image - self.image_mean) / self.image_std
            last_hidden_state, image_features = self.model(image, return_dict=False)
            tokens = last_hidden_state[:, 1:, :]

        hint = torch.cat([image_features.unsqueeze(1), tokens], 1)  # 8,257,1024
        hint = self.projector(hint)
        if self.use_densepose:
            hint = torch.cat([hint, dense_proj], 1)
        return hint

    def encode(self, image):
        return self(image)


class FrozenFashionClipEncoder(AbstractEncoder):
    def __init__(self,
                 freeze=True,
                 use_densepose=False,
                 ):
        super().__init__()
        self.model = CLIPVisionModelWithProjection.from_pretrained("patrickjohncyh/fashion-clip")
        self.fclip_processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

        if freeze:
            self.freeze()

        self.register_buffer('image_mean', torch.tensor([0.48145466, 0.4578275, 0.40821073]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
        self.register_buffer('image_std', torch.tensor([0.26862954, 0.26130258, 0.27577711]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1))

        self.clip_projector = torch.nn.Linear(self.model.visual_projection.in_features,
                                              self.model.visual_projection.out_features, bias=False)
        self.clip_projector.load_state_dict(self.model.visual_projection.state_dict())
        self.clip_projector.requires_grad_(True)

        self.projector = nn.Linear(self.model.visual_projection.out_features, 1024)

        self.use_densepose = use_densepose

        if use_densepose:
            # Make it separate class, add res connection, 4 blocks
            self.pose_projector = nn.Sequential(
                conv_nd(2, 25, 128, 7, padding=3, stride=2),
                nn.SiLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 56x56
                ResBlock(128, 128, 3, padding=1, stride=1),  # 56x56
                conv_nd(2, 128, 256, 3, padding=1, stride=2),  # 28x28
                nn.SiLU(),
                ResBlock(256, 256, 3, padding=1, stride=1),  # 28x28
                conv_nd(2, 256, 512, 3, padding=1, stride=2),  # 14x14
                nn.SiLU(),
                ResBlock(512, 512, 3, padding=1, stride=1),  # 14x14
                conv_nd(2, 512, 1024, 3, padding=1, stride=2),  # 7x7
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )

    def forward(self, image):
        if isinstance(image, list):
            image = torch.cat(image, 0)

        if self.use_densepose:
            image, dp_ref, dp_tar = image.split((3, 25, 25), 1)
            dense_proj = torch.cat([
                self.pose_projector(dp_ref).unsqueeze(1),  # Bx1x1024
                self.pose_projector(dp_tar).unsqueeze(1),  # Bx1x1024
            ], 1)

        with torch.no_grad():
            input_ids = self.fclip_processor(images=image,
                                             return_tensors='pt',
                                             do_resize=False,
                                             resample=False,
                                             do_center_crop=False,
                                             do_rescale=False,
                                             do_convert_rgb=False)

            last_hidden_states = self.model(**input_ids).last_hidden_state
            last_hidden_states_norm = self.model.vision_model.post_layernorm(last_hidden_states)
            image_features = self.clip_projector(last_hidden_states_norm)  # 1x50x512
            image_features = self.projector(image_features)  # 1x50x1024
        if self.use_densepose:
            image_features = torch.cat([image_features, dense_proj], 1)
        return image_features
