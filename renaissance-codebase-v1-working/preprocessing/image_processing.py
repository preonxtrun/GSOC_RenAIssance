"""
Image preprocessing for historical manuscripts.
Grayscale conversion, denoising, deskewing, and contrast enhancement.
"""
import cv2
import numpy as np
from PIL import Image
from typing import Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DENOISE_STRENGTH, CLAHE_CLIP_LIMIT, CLAHE_TILE_SIZE, SKEW_MAX_ANGLE


def pil_to_cv2(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV format (BGR)."""
    rgb = np.array(image)
    if len(rgb.shape) == 2:
        return rgb  # Already grayscale
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def cv2_to_pil(image: np.ndarray) -> Image.Image:
    """Convert OpenCV image (BGR or grayscale) to PIL Image."""
    if len(image.shape) == 2:
        return Image.fromarray(image)
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def grayscale(image: np.ndarray) -> np.ndarray:
    """Convert to grayscale if not already."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def denoise(image: np.ndarray, h: int = DENOISE_STRENGTH) -> np.ndarray:
    """
    Apply non-local means denoising.
    Works well for historical documents with age-related noise/stains.
    """
    if len(image.shape) == 3:
        return cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)
    return cv2.fastNlMeansDenoising(image, None, h, 7, 21)


def enhance_contrast(
    image: np.ndarray,
    clip_limit: float = CLAHE_CLIP_LIMIT,
    tile_size: tuple = CLAHE_TILE_SIZE,
) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Improves readability of faded historical ink.
    """
    gray = grayscale(image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return clahe.apply(gray)


def estimate_skew_angle(image: np.ndarray) -> float:
    """
    Estimate the skew angle of a document image using projection profile.
    Returns the angle in degrees.
    """
    gray = grayscale(image)

    # Threshold to binary
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find coordinates of all non-zero pixels
    coords = np.column_stack(np.where(binary > 0))

    if len(coords) < 100:
        return 0.0  # Not enough text to estimate

    # Use minAreaRect on the points
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]

    # Normalize angle
    if angle < -45:
        angle = -(90 + angle)
    elif angle > 45:
        angle = -(angle - 90)
    else:
        angle = -angle

    # Clamp to max angle to avoid wild rotations
    angle = max(-SKEW_MAX_ANGLE, min(SKEW_MAX_ANGLE, angle))

    return angle


def deskew(image: np.ndarray, angle: Optional[float] = None) -> np.ndarray:
    """
    Correct skew in a document image.
    If angle is None, it will be estimated automatically.
    """
    if angle is None:
        angle = estimate_skew_angle(image)

    if abs(angle) < 0.5:
        return image  # Skip if angle is negligible

    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # Rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Compute new bounding size
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    # Adjust the rotation matrix for the new center
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    # Use white background for rotated regions
    border_value = 255 if len(image.shape) == 2 else (255, 255, 255)
    rotated = cv2.warpAffine(
        image, M, (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )

    return rotated


def binarize(image: np.ndarray, adaptive: bool = True) -> np.ndarray:
    """
    Binarize the image using Otsu or adaptive thresholding.
    Helpful for segmentation but NOT for feeding into neural OCR models (use grayscale for that).
    """
    gray = grayscale(image)

    if adaptive:
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 10,
        )
    else:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary


def preprocess_image(
    image: Image.Image | np.ndarray,
    do_grayscale: bool = True,
    do_denoise: bool = True,
    do_deskew: bool = True,
    do_enhance: bool = True,
) -> np.ndarray:
    """
    Full preprocessing pipeline for a single page image.

    Args:
        image: PIL Image or numpy array.
        do_grayscale: Convert to grayscale.
        do_denoise: Apply denoising.
        do_deskew: Correct skew.
        do_enhance: Apply CLAHE contrast enhancement.

    Returns:
        Preprocessed image as numpy array.
    """
    if isinstance(image, Image.Image):
        img = pil_to_cv2(image)
    else:
        img = image.copy()

    if do_grayscale:
        img = grayscale(img)

    if do_denoise:
        img = denoise(img)

    if do_deskew:
        img = deskew(img)

    if do_enhance:
        img = enhance_contrast(img)

    return img
