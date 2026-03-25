"""
Data alignment: match transcription text with corresponding page images.
Parses .docx transcriptions and aligns them with the initial pages of each PDF.
"""
import sys
from pathlib import Path
from typing import Optional

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    HANDWRITING_PDFS_DIR,
    HANDWRITING_TRANSCRIPTIONS_DIR,
    HANDWRITING_PAIRS,
    TRAIN_VAL_SPLIT,
    IMAGES_DIR,
)
from utils.helpers import parse_docx, parse_docx_paragraphs, setup_logger
from preprocessing.pdf_to_images import pdf_to_images

logger = setup_logger("alignment")


def align_transcriptions(
    pairs: list[dict] = HANDWRITING_PAIRS,
    pdfs_dir: Path = HANDWRITING_PDFS_DIR,
    transcriptions_dir: Path = HANDWRITING_TRANSCRIPTIONS_DIR,
    max_pages_per_doc: Optional[int] = None,
    dpi: int = 200,
) -> list[dict]:
    """
    Align transcription text with corresponding page images.

    Strategy:
    - .docx transcriptions cover only the INITIAL portion of each manuscript.
    - We estimate how many pages the transcription covers based on text length.
    - We split the transcription into roughly equal chunks per page.
    - This creates (page_image, text) pairs for supervised training.

    Args:
        pairs: List of PDF/transcription pair dicts from config.
        pdfs_dir: Directory containing PDF files.
        transcriptions_dir: Directory containing .docx files.
        max_pages_per_doc: Limit pages per document (for quick testing).
        dpi: DPI for PDF conversion.

    Returns:
        List of dicts with keys: 'image', 'text', 'source', 'page_num', 'split'.
    """
    aligned_data = []

    for pair in pairs:
        pdf_path = pdfs_dir / pair["pdf"]
        docx_path = transcriptions_dir / pair["transcription"]
        name = pair["name"]

        if not pdf_path.exists():
            logger.warning(f"PDF not found: {pdf_path}")
            continue
        if not docx_path.exists():
            logger.warning(f"Transcription not found: {docx_path}")
            continue

        logger.info(f"Processing: {name}")

        # Parse transcription and clean out modern archivist notes
        raw_paragraphs = parse_docx_paragraphs(docx_path)
        
        # Strip out metadata lines that aren't actually in the manuscript image
        paragraphs = []
        
        # English keywords uniquely inserted by the archivist in the notes section
        archivist_keywords = [
            "notes:", "interchangeably", "dictionary?", "accents are inconsistent",
            "horizontal “cap”", "line end hyphens", "modern z", "always modern"
        ]
        
        for p in raw_paragraphs:
            p_lower = p.strip().lower()
            
            # Skip paragraph if it contains any known archivist note strings
            if any(k in p_lower for k in archivist_keywords):
                continue
                
            # Filter out explicit page markers like "PDF p1"
            if p_lower.startswith("pdf p") or p_lower.startswith("[page"):
                continue
                
            if p.strip():
                paragraphs.append(p)

        full_text = "\n".join(paragraphs)

        if not paragraphs:
            logger.warning(f"Empty transcription for {name} after filtering notes")
            continue

        # Convert PDF to images
        images_dir = IMAGES_DIR / name
        images = pdf_to_images(pdf_path, output_dir=images_dir, dpi=dpi, save=True)

        if not images:
            logger.warning(f"No images from {name}")
            continue

        # Estimate how many pages the transcription covers
        # Heuristic: ~200-400 words per handwritten page
        word_count = len(full_text.split())
        estimated_pages = max(1, min(len(images), word_count // 250))

        if max_pages_per_doc:
            estimated_pages = min(estimated_pages, max_pages_per_doc)

        logger.info(
            f"  {name}: {len(paragraphs)} paragraphs, ~{word_count} words, "
            f"est. {estimated_pages} pages transcribed out of {len(images)}"
        )

        # Split transcription text across estimated pages
        # Distribute paragraphs roughly evenly
        if estimated_pages >= len(paragraphs):
            # One paragraph per page
            page_texts = paragraphs[:estimated_pages]
        else:
            # Multiple paragraphs per page
            paras_per_page = len(paragraphs) / estimated_pages
            page_texts = []
            for i in range(estimated_pages):
                start = int(i * paras_per_page)
                end = int((i + 1) * paras_per_page)
                page_text = "\n".join(paragraphs[start:end])
                page_texts.append(page_text)

        # Create aligned pairs (transcribed pages only)
        for page_idx, text in enumerate(page_texts):
            if page_idx >= len(images):
                break

            aligned_data.append(
                {
                    "image": images[page_idx],
                    "text": text,
                    "source": name,
                    "page_num": page_idx + 1,
                    "is_transcribed": True,
                }
            )

        # Mark remaining pages as untranscribed (for inference)
        for page_idx in range(estimated_pages, len(images)):
            aligned_data.append(
                {
                    "image": images[page_idx],
                    "text": None,  # No ground truth
                    "source": name,
                    "page_num": page_idx + 1,
                    "is_transcribed": False,
                }
            )

    # Split transcribed data into train/val
    transcribed = [d for d in aligned_data if d["is_transcribed"]]
    untranscribed = [d for d in aligned_data if not d["is_transcribed"]]

    split_idx = int(len(transcribed) * TRAIN_VAL_SPLIT)
    for i, item in enumerate(transcribed):
        item["split"] = "train" if i < split_idx else "val"

    for item in untranscribed:
        item["split"] = "test"

    logger.info(
        f"Alignment complete: {len(transcribed)} transcribed pages "
        f"({split_idx} train, {len(transcribed) - split_idx} val), "
        f"{len(untranscribed)} untranscribed pages"
    )

    return aligned_data


def get_train_val_test_split(
    aligned_data: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split aligned data into train, val, and test sets."""
    train = [d for d in aligned_data if d["split"] == "train"]
    val = [d for d in aligned_data if d["split"] == "val"]
    test = [d for d in aligned_data if d["split"] == "test"]
    return train, val, test
