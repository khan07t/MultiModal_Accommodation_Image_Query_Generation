"""Crop detected amenities with surrounding context."""

from PIL import Image

from .config import PAD_FRAC


def safe_open_image(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def crop_box(img: Image.Image, box, pad_frac: float = PAD_FRAC) -> Image.Image:
    """Crop [x1, y1, x2, y2] with proportional padding, clipped to the image."""
    w, h = img.size
    x1, y1, x2, y2 = box

    pad_x = pad_frac * (x2 - x1)
    pad_y = pad_frac * (y2 - y1)

    x1 = max(int(x1 - pad_x), 0)
    y1 = max(int(y1 - pad_y), 0)
    x2 = min(int(x2 + pad_x), w)
    y2 = min(int(y2 + pad_y), h)

    return img.crop((x1, y1, x2, y2))
