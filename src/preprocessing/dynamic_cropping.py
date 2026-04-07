"""
Dynamic image cropping utilities for DeepSeek-OCR-2.

Extracted from process/image_process.py.
"""

from typing import List, Tuple
from PIL import Image

from src.config import MIN_CROPS, MAX_CROPS, IMAGE_SIZE


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: List[Tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> Tuple[int, int]:
    """Find the closest supported (w_tiles, h_tiles) ratio for an image."""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def count_tiles(
    orig_width: int,
    orig_height: int,
    min_num: int = MIN_CROPS,
    max_num: int = MAX_CROPS,
    image_size: int = IMAGE_SIZE,
) -> Tuple[int, int]:
    """
    Determine the (width_tiles, height_tiles) grid for dynamic cropping.

    Returns:
        (num_width_tiles, num_height_tiles)
    """
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    return find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = MIN_CROPS,
    max_num: int = MAX_CROPS,
    image_size: int = IMAGE_SIZE,
    use_thumbnail: bool = False,
) -> Tuple[List[Image.Image], Tuple[int, int]]:
    """
    Crop an image into tiles using the best-matching aspect ratio.

    Args:
        image: Input PIL image.
        min_num: Minimum number of total tiles.
        max_num: Maximum number of total tiles.
        image_size: Side length of each square tile.
        use_thumbnail: If True and multiple crops, also append a thumbnail.

    Returns:
        (list_of_tile_images, (num_width_tiles, num_height_tiles))
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))

    return processed_images, target_aspect_ratio
