"""
═══════════════════════════════════════════════════════════════
  02 — FULL PIPELINE
  RenAIssance HTR — Complete Submission Version
═══════════════════════════════════════════════════════════════

This script runs the full pipeline suitable for final GSoC submission:
  1. Preprocessing (all documents)
  2. Data alignment (transcription ↔ page images)
  3. TrOCR fine-tuning (optional, needs GPU)
  4. Dual-track inference (TrOCR + VLM)
  5. LLM fusion/correction
  6. Evaluation (CER, WER, confidence, uncertainty)
  7. Ablation experiment

Usage:
    python notebooks/02_full_pipeline.py [--skip-training] [--skip-vlm]
"""
import sys
import argparse
import json
from pathlib import Path
import warnings

# Suppress huge HuggingFace model loading outputs
import transformers
transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    HANDWRITING_PDFS_DIR,
    HANDWRITING_TRANSCRIPTIONS_DIR,
    HANDWRITING_PAIRS,
    RESULTS_DIR,
    PDF_DPI,
    NUM_EPOCHS
)


def parse_args():
    parser = argparse.ArgumentParser(description="RenAIssance HTR Full Pipeline")
    parser.add_argument(
        "--skip-training", action="store_true",
        help="Skip TrOCR fine-tuning (use pretrained model)"
    )
    parser.add_argument(
        "--skip-vlm", action="store_true",
        help="Skip VLM (use TrOCR + LLM only)"
    )
    parser.add_argument(
        "--max-pages", type=int, default=None,
        help="Max pages per document to process (default: all)"
    )
    parser.add_argument(
        "--max-docs", type=int, default=None,
        help="Max documents to process (default: all)"
    )
    parser.add_argument(
        "--dpi", type=int, default=PDF_DPI,
        help="DPI for PDF conversion"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("  RenAIssance HTR — Full Pipeline (GSoC Submission)")
    print("  VLM/LLM-Centered Handwritten Historical Text Recognition")
    print("=" * 70)

    # ─────────────────────────────────────────
    #  PHASE 1: Data Preparation
    # ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PHASE 1: Data Preparation")
    print("=" * 70)

    print("\n[1.1] Aligning transcriptions with page images...")
    from training.data_alignment import align_transcriptions, get_train_val_test_split

    aligned_data = align_transcriptions(
        pairs=HANDWRITING_PAIRS[:args.max_docs],
        max_pages_per_doc=args.max_pages,
        dpi=args.dpi,
    )

    train_data, val_data, test_data = get_train_val_test_split(aligned_data)
    print(f"  Train: {len(train_data)} pages")
    print(f"  Val:   {len(val_data)} pages")
    print(f"  Test:  {len(test_data)} untranscribed pages")

    # ─────────────────────────────────────────
    #  PHASE 2: Training (Optional)
    # ─────────────────────────────────────────
    if not args.skip_training:
        print("\n" + "=" * 70)
        print("  PHASE 2: TrOCR Fine-Tuning")
        print("=" * 70)

        from training.trocr_finetune import train as trocr_train

        try:
            trocr_train(
                max_pages_per_doc=args.max_pages,
                dpi=args.dpi,
                num_epochs=NUM_EPOCHS,
            )
            print("  V Fine-tuning complete")
        except Exception as e:
            print(f"  ⚠ Training failed: {e}")
            print("  Continuing with pretrained model...")
    else:
        print("\n[Skipping TrOCR training — --skip-training flag passed]")

    # ─────────────────────────────────────────
    #  PHASE 3: Inference
    # ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PHASE 3: Inference")
    print("=" * 70)

    from inference.pipeline import RenaissanceHTRPipeline

    if args.skip_vlm:
        mode = "standard"
        print("\n  [!] skip-vlm passed. Falling back to TrOCR -> LLM mode.")
    else:
        mode = "full"
        print("\n  [!] Using FULL mode. Dual-track pipeline: TrOCR + VLM -> LLM Fusion.")

    print(f"\n[3.1] Running pipeline in '{mode}' mode...")

    pipeline = RenaissanceHTRPipeline(mode=mode, load_models=True)

    # Process transcribed pages (for evaluation)
    eval_data = val_data if val_data else train_data[:args.max_pages]
    print(f"\n  Processing {len(eval_data)} transcribed pages in batch...")
    
    # Process batch automatically handles memory management stage-by-stage
    eval_results = pipeline.process_batch(eval_data)
    
    all_predictions = [r["transcription"] for r in eval_results]
    all_references = [p["text"] for p in eval_data]

    for i, (page_data, result) in enumerate(zip(eval_data, eval_results)):
        print(f"\n    Page {i + 1}/{len(eval_data)} [{page_data['source']}, p.{page_data['page_num']}]")
        print(f"    Method: {result['method']}")
        if 'confidence' in result:
            print(f"    Confidence: {result['confidence']:.4f}")
        print(f"    Preview: {result['transcription'][:80]}...")

    # Process untranscribed pages
    if test_data:
        test_pages = test_data[:args.max_pages]
        print(f"\n[3.2] Inferring on {len(test_pages)} untranscribed pages in batch...")
        
        test_results = pipeline.process_batch(test_pages)
        
        for i, (page_data, result) in enumerate(zip(test_pages, test_results)):
            print(f"  Page {page_data['page_num']}: {result['transcription'][:60]}...")
            out_file = RESULTS_DIR / f"inferred_{page_data['source']}_p{page_data['page_num']}.txt"
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(result["transcription"])

    # ─────────────────────────────────────────
    #  PHASE 4: Evaluation
    # ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PHASE 4: Evaluation")
    print("=" * 70)

    from evaluation.metrics import (
        compute_cer, compute_wer, uncertainty_analysis, format_metrics_report
    )

    cer = compute_cer(all_predictions, all_references)
    wer = compute_wer(all_predictions, all_references)
    unc = uncertainty_analysis(all_predictions, all_references)

    print(format_metrics_report(cer, wer, unc, f"Full Pipeline ({mode} mode)"))

    # Save detailed results
    eval_results = {
        "mode": mode,
        "cer": cer,
        "wer": wer,
        "num_pages": len(all_predictions),
        "uncertainty": unc,
        "pages": [
            {
                "prediction_preview": pred[:200],
                "reference_preview": ref[:200],
                "cer": compute_cer([pred], [ref]),
                "wer": compute_wer([pred], [ref]),
            }
            for pred, ref in zip(all_predictions, all_references)
        ],
    }

    results_file = RESULTS_DIR / "full_pipeline_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    print(f"\n  V Results saved -> {results_file}")

    # ─────────────────────────────────────────
    #  PHASE 5: Ablation Experiment
    # ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PHASE 5: Ablation Experiment")
    print("=" * 70)

    from evaluation.ablation import run_ablation

    try:
        ablation = run_ablation(
            max_pages=args.max_pages,
            max_docs=args.max_docs,
            dpi=args.dpi,
        )
        print("  V Ablation experiment complete")
    except Exception as e:
        print(f"  ⚠ Ablation failed: {e}")

    # ─────────────────────────────────────────
    #  Summary
    # ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Mode:       {mode}")
    print(f"  CER:        {cer:.4f} ({cer * 100:.2f}%)")
    print(f"  WER:        {wer:.4f} ({wer * 100:.2f}%)")
    print(f"  Pages:      {len(all_predictions)} evaluated, {len(test_data)} inferred")
    print(f"  Results:    {RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
