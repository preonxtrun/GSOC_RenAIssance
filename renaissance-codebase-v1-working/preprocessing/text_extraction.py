"""
Text block extraction and line segmentation.
Isolates the main text body from marginalia, ornaments, and decorations.
"""
import cv2
import numpy as np
from PIL import Image
from typing import Optional

from .image_processing import grayscale, binarize, pil_to_cv2


def extract_text_block(
    image: Image.Image | np.ndarray,
    margin_ratio: float = 0.10,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Extract the main text block from a manuscript page, excluding marginalia.

    Strategy:
    1. Binarize and find all contours.
    2. Compute a density map via horizontal/vertical projection.
    3. Find the largest contiguous text region.
    4. Crop to that region with small padding.

    Args:
        image: Input page image.
        margin_ratio: Fraction of page width/height to exclude from edges
                      (where marginalia typically appears).

    Returns:
        Tuple of (cropped_image, (x, y, w, h) bounding box).
    """
    if isinstance(image, Image.Image):
        img = pil_to_cv2(image)
    else:
        img = image.copy()

    gray = grayscale(img)
    h, w = gray.shape

    # Binarize (inverted: text = white)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Define margin exclusion zone
    margin_x = int(w * margin_ratio)
    margin_y = int(h * margin_ratio * 0.5)  # Less vertical margin

    # Mask out the margins (exclude marginalia zones)
    mask = np.zeros_like(binary)
    mask[margin_y:h - margin_y, margin_x:w - margin_x] = 255
    binary_masked = cv2.bitwise_and(binary, mask)

    # Dilate to connect nearby text elements into blocks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 10))
    dilated = cv2.dilate(binary_masked, kernel, iterations=3)

    # Find contours of text blocks
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # Fallback: return center 80% of the page
        cx, cy = int(w * 0.1), int(h * 0.05)
        cw, ch = int(w * 0.8), int(h * 0.9)
        return img[cy:cy + ch, cx:cx + cw], (cx, cy, cw, ch)

    # Find the largest contour (assumed to be main text body)
    largest = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(largest)

    # If multiple large contours, union them
    min_area = 0.05 * w * h  # At least 5% of page
    big_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    if len(big_contours) > 1:
        all_points = np.vstack(big_contours)
        x, y, bw, bh = cv2.boundingRect(all_points)

    # Add small padding
    pad = 20
    x = max(0, x - pad)
    y = max(0, y - pad)
    bw = min(w - x, bw + 2 * pad)
    bh = min(h - y, bh + 2 * pad)

    cropped = img[y:y + bh, x:x + bw]
    return cropped, (x, y, bw, bh)


def segment_lines(
    image: np.ndarray,
    min_line_height: int = 20,
    max_line_height: int = 200,
) -> list[np.ndarray]:
    """
    Segment a text block image into individual text lines using horizontal projection.

    Args:
        image: Preprocessed text block (grayscale recommended).
        min_line_height: Minimum pixel height for a valid text line.
        max_line_height: Maximum pixel height for a valid text line.

    Returns:
        List of cropped line images, top to bottom.
    """
    gray = grayscale(image) if len(image.shape) == 3 else image
    h, w = gray.shape

    # Binarize (inverted)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Horizontal projection profile
    h_proj = np.sum(binary, axis=1) / 255  # Count of foreground pixels per row

    # Smooth the projection to handle noise
    kernel_size = 5
    h_proj_smooth = np.convolve(h_proj, np.ones(kernel_size) / kernel_size, mode="same")

    # Find line boundaries using threshold
    threshold = np.max(h_proj_smooth) * 0.05  # 5% of max density
    is_text = h_proj_smooth > threshold

    # Find transitions (start and end of text lines)
    lines = []
    in_line = False
    line_start = 0

    for row in range(len(is_text)):
        if is_text[row] and not in_line:
            line_start = row
            in_line = True
        elif not is_text[row] and in_line:
            line_end = row
            line_height = line_end - line_start

            if min_line_height <= line_height <= max_line_height:
                # Add small vertical padding
                pad = 5
                y1 = max(0, line_start - pad)
                y2 = min(h, line_end + pad)
                line_img = image[y1:y2, :]
                lines.append(line_img)

            in_line = False

    # Handle case where last line extends to bottom
    if in_line:
        line_height = h - line_start
        if min_line_height <= line_height <= max_line_height:
            pad = 5
            y1 = max(0, line_start - pad)
            line_img = image[y1:, :]
            lines.append(line_img)

    return lines


def get_line_images(
    page_image: Image.Image | np.ndarray,
    preprocess: bool = True,
) -> list[np.ndarray]:
    """
    Full pipeline: extract text block → segment into lines.

    Args:
        page_image: Full page image.
        preprocess: Whether to apply basic preprocessing before segmentation.

    Returns:
        List of individual line images ready for OCR.
    """
    if isinstance(page_image, Image.Image):
        img = pil_to_cv2(page_image)
    else:
        img = page_image

    # Extract main text block
    text_block, bbox = extract_text_block(img)

    # Segment into lines
    lines = segment_lines(text_block)

    if not lines:
        # Fallback: return the whole text block as a single "line"
        return [text_block]

    return lines
