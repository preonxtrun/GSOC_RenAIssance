"""
PyTorch Dataset for TrOCR fine-tuning on historical handwritten text.
"""
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TROCR_PROCESSOR_NAME, MAX_SEQ_LENGTH
from preprocessing.text_extraction import get_line_images
from preprocessing.image_processing import preprocess_image, cv2_to_pil


class HTRDataset(Dataset):
    """
    Dataset for fine-tuning TrOCR on aligned historical text data.

    Each sample is a (line_image, text_line) pair extracted from
    page-level alignments.
    """

    def __init__(
        self,
        aligned_data: list[dict],
        processor_name: str = TROCR_PROCESSOR_NAME,
        max_length: int = MAX_SEQ_LENGTH,
        preprocess: bool = True,
        is_train: bool = False,
    ):
        """
        Args:
            aligned_data: List of dicts with 'image' (PIL Image) and 'text' (str).
                          Only items where 'is_transcribed' is True are used.
            processor_name: HuggingFace processor name for TrOCR.
            max_length: Maximum sequence length for tokenization.
            preprocess: Whether to apply image preprocessing.
            is_train: Apply data augmentation if True.
        """
        from transformers import TrOCRProcessor

        self.processor = TrOCRProcessor.from_pretrained(processor_name)
        self.max_length = max_length
        self.preprocess = preprocess
        self.is_train = is_train

        # Build line-level samples from page-level data
        self.samples = self._build_samples(aligned_data)

    def _build_samples(self, aligned_data: list[dict]) -> list[dict]:
        """
        Convert page-level alignments to training samples.
        Uses page-level image/text pairs for reliable alignment.
        Line-level segmentation is skipped because segmenter counts
        rarely match the number of text lines, causing misaligned training.
        """
        samples = []

        for page_data in aligned_data:
            if not page_data.get("is_transcribed", False):
                continue
            if not page_data.get("text"):
                continue

            image = page_data["image"]
            text = page_data["text"]

            if self.preprocess:
                processed = preprocess_image(image)
            else:
                processed = np.array(image) if isinstance(image, Image.Image) else image

            # One clean page-level sample per transcribed page
            samples.append(
                {
                    "image": processed,
                    "text": text[:512],          # clip to reasonable length
                    "source": page_data.get("source", "unknown"),
                    "page_num": page_data.get("page_num", -1),
                }
            )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = sample["image"]

        # Convert to PIL RGB
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                image = Image.fromarray(image).convert("RGB")
            else:
                image = Image.fromarray(image).convert("RGB")
        elif image.mode != "RGB":
            image = image.convert("RGB")

        # Data augmentation
        if getattr(self, "is_train", False):
            import torchvision.transforms as T
            augment = T.Compose([
                T.ColorJitter(brightness=0.3, contrast=0.3),
                T.RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.95, 1.05))
            ])
            image = augment(image)

        # Process image for TrOCR
        pixel_values = self.processor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze(0)

        # Tokenize text
        labels = self.processor.tokenizer(
            sample["text"],
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        # Replace padding token id with -100 for loss computation
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }
