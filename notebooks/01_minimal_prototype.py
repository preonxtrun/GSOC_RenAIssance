"""
═══════════════════════════════════════════════════════════════
  01 — MINIMAL PROTOTYPE
  RenAIssance HTR Pipeline — Quick Working Demo
═══════════════════════════════════════════════════════════════

This script demonstrates the minimal viable pipeline:
  PDF → Preprocessing → TrOCR OCR → Text Output

Works on CPU. No VLM or LLM required.
Expected runtime: ~5 minutes for one page.

Usage:
    python notebooks/01_minimal_prototype.py
"""
import sys
import os
from pathlib import Path

# Setup project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import HANDWRITING_PDFS_DIR, HANDWRITING_PAIRS, RESULTS_DIR


def main():
    print("=" * 60)
    print("  RenAIssance HTR — Minimal Prototype")
    print("  (PDF -> Preprocess -> TrOCR -> Text)")
    print("=" * 60)

    # ─── Step 1: Convert first page of first PDF ───
    print("\n[1/4] Converting PDF to image...")
    from preprocessing.pdf_to_images import pdf_to_images

    pair = HANDWRITING_PAIRS[0]
    pdf_path = HANDWRITING_PDFS_DIR / pair["pdf"]

    if not pdf_path.exists():
        print(f"ERROR: PDF not found at {pdf_path}")
        print("Make sure the dataset is in the correct directory.")
        return

    # Convert just the first page at lower DPI for speed
    images = pdf_to_images(pdf_path, dpi=200, save=False)
    page_image = images[0]
    print(f"  ✓ Got {len(images)} pages. Using page 1.")

    # ─── Step 2: Preprocess ───
    print("\n[2/4] Preprocessing image...")
    from preprocessing.image_processing import preprocess_image, cv2_to_pil

    processed = preprocess_image(page_image)
    print(f"  ✓ Preprocessed: {processed.shape}")

    # Save preprocessed image for visual inspection
    preprocessed_img = cv2_to_pil(processed)
    output_path = RESULTS_DIR / "preprocessed_sample.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preprocessed_img.save(str(output_path))
    print(f"  ✓ Saved preprocessed sample → {output_path}")

    # ─── Step 3: Extract text lines ───
    print("\n[3/4] Segmenting text lines...")
    from preprocessing.text_extraction import get_line_images

    line_images = get_line_images(processed, preprocess=False)
    print(f"  ✓ Found {len(line_images)} text lines")

    # Save a sample line image
    if line_images:
        from PIL import Image
        import numpy as np

        sample_line = line_images[0]
        if isinstance(sample_line, np.ndarray):
            if len(sample_line.shape) == 2:
                sample_line_pil = Image.fromarray(sample_line)
            else:
                sample_line_pil = Image.fromarray(sample_line)
        else:
            sample_line_pil = sample_line
        sample_line_pil.save(str(RESULTS_DIR / "sample_line.png"))
        print(f"  ✓ Saved sample line → {RESULTS_DIR / 'sample_line.png'}")

    # ─── Step 4: OCR with TrOCR ───
    print("\n[4/4] Running TrOCR OCR...")
    from models.ocr_backbone import TrOCREngine

    engine = TrOCREngine()
    page_result = engine.recognize_page(line_images)

    print(f"\n{'=' * 60}")
    print(f"  TRANSCRIPTION RESULT")
    print(f"{'=' * 60}")
    print(f"  Lines recognized: {page_result['num_lines']}")
    print(f"  Average confidence: {page_result['avg_confidence']:.4f}")
    print(f"  Uncertain lines: {len(page_result['uncertain_lines'])}")
    print(f"{'─' * 60}")
    print(f"\n{page_result['full_text']}")
    print(f"\n{'=' * 60}")

    # Save transcription
    output_file = RESULTS_DIR / f"transcription_{pair['name']}_page1.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(page_result["full_text"])
    print(f"\n  ✓ Transcription saved → {output_file}")

    # Show line-by-line confidence
    print(f"\n  Line-by-line confidence:")
    for lr in page_result["line_results"]:
        marker = "⚠️" if lr.get("is_uncertain") else "✓"
        print(
            f"    {marker} Line {lr['line_index'] + 1}: "
            f"conf={lr['confidence']:.3f} | {lr['text'][:60]}..."
        )

    # ─── Compare with ground truth ───
    print(f"\n[BONUS] Comparing with ground truth...")
    from config import HANDWRITING_TRANSCRIPTIONS_DIR
    from utils.helpers import parse_docx

    docx_path = HANDWRITING_TRANSCRIPTIONS_DIR / pair["transcription"]
    if docx_path.exists():
        gt_text = parse_docx(docx_path)
        gt_preview = gt_text[:500]
        print(f"\n  Ground truth preview:")
        print(f"  {gt_preview}...")

        from evaluation.metrics import compute_cer, compute_wer

        # Compare first few lines specifically to avoid length mismatch
        # The GT is truncated to 500 chars, so we must truncate the pred to ~500 chars
        pred_preview = page_result["full_text"][:len(gt_preview)]

        cer = compute_cer([pred_preview], [gt_preview])
        wer = compute_wer([pred_preview], [gt_preview])
        print(f"\n  CER (vs GT first section): {cer:.4f}")
        print(f"  WER (vs GT first section): {wer:.4f}")

    print(f"\n{'=' * 60}")
    print("  Minimal prototype complete! ✓")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
