"""
Image preprocessing for DeepSeek-OCR-2.

Provides a clean ImageProcessor that handles global view padding and
local crop extraction without vLLM or tokenizer dependencies.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torchvision.transforms as T
from PIL import Image, ImageOps

from src.config import BASE_SIZE, IMAGE_SIZE, CROP_MODE
from .dynamic_cropping import dynamic_preprocess


class ImageTransform:
    """Apply ToTensor + optional normalization to a PIL image."""

    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        normalize: bool = True,
    ):
        self.mean = mean
        self.std = std
        self.normalize = normalize

        pipeline = [T.ToTensor()]
        if normalize:
            pipeline.append(T.Normalize(mean, std))
        self.transform = T.Compose(pipeline)

    def __call__(self, pil_img: Image.Image) -> torch.Tensor:
        return self.transform(pil_img)


class ImageProcessor:
    """
    Simplified image preprocessor for interpretability research.

    Handles:
    - Global view: pad image to base_size × base_size
    - Local views: dynamic crop into image_size × image_size tiles
    """

    def __init__(
        self,
        image_size: int = IMAGE_SIZE,
        base_size: int = BASE_SIZE,
        crop_mode: bool = CROP_MODE,
        image_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        image_std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    ):
        self.image_size = image_size
        self.base_size = base_size
        self.crop_mode = crop_mode
        self.image_transform = ImageTransform(mean=image_mean, std=image_std)

    def process_image(self, image: Image.Image) -> Dict[str, torch.Tensor]:
        """
        Preprocess a PIL image for DeepSeek-OCR-2.

        Returns a dict with keys:
            - pixel_values: [1, 1, 3, base_size, base_size]  — global view
            - images_crop:  [1, 1, n_patches, 3, image_size, image_size]  — local tiles (or zeros)
            - images_spatial_crop: [1, 1, 2]  — (width_tiles, height_tiles)
        """
        image = image.convert("RGB")
        orig_w, orig_h = image.size

        # Determine crop ratio
        if not self.crop_mode or (orig_w <= 768 and orig_h <= 768):
            crop_ratio = (1, 1)
            crop_tiles = []
        else:
            crop_tiles, crop_ratio = dynamic_preprocess(image, image_size=self.image_size)

        num_width_tiles, num_height_tiles = crop_ratio

        # Global view: pad to base_size
        pad_color = tuple(int(x * 255) for x in self.image_transform.mean)
        global_view = ImageOps.pad(image, (self.base_size, self.base_size), color=pad_color)
        pixel_values = self.image_transform(global_view)  # [3, base_size, base_size]

        # Local crops
        if crop_tiles:
            images_crop = torch.stack([self.image_transform(t) for t in crop_tiles], dim=0)
        else:
            # Dummy zeros — signals "no local crops" to the model
            images_crop = torch.zeros(1, 3, self.image_size, self.image_size)

        # Add batch + image dims to match model expectations
        # pixel_values: [n_images=1, batch=1, 3, H, W]
        pixel_values = pixel_values.unsqueeze(0).unsqueeze(0)
        # images_crop: [n_images=1, batch=1, n_patches, 3, H, W]
        images_crop = images_crop.unsqueeze(0).unsqueeze(0)
        # images_spatial_crop: [n_images=1, batch=1, 2]
        images_spatial_crop = torch.tensor(
            [[num_width_tiles, num_height_tiles]], dtype=torch.long
        ).unsqueeze(0)

        return {
            "pixel_values": pixel_values,
            "images_crop": images_crop,
            "images_spatial_crop": images_spatial_crop,
        }

    def process_batch(self, images: List[Image.Image]) -> Dict[str, torch.Tensor]:
        """
        Process a list of images. Each image is treated as a separate item.
        Returns dicts concatenated along the n_images dimension.
        """
        processed = [self.process_image(img) for img in images]
        return {
            "pixel_values": torch.cat([p["pixel_values"] for p in processed], dim=0),
            "images_crop": torch.cat([p["images_crop"] for p in processed], dim=0),
            "images_spatial_crop": torch.cat([p["images_spatial_crop"] for p in processed], dim=0),
        }
