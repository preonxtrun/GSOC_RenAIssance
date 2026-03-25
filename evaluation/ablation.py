"""
Ablation experiment: measuring the impact of LLM correction on OCR quality.
Compares pipeline performance with and without LLM correction.
"""
import sys
import json
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import HANDWRITING_PDFS_DIR, HANDWRITING_PAIRS, RESULTS_DIR
from utils.helpers import setup_logger
from evaluation.metrics import (
    compute_cer,
    compute_wer,
    uncertainty_analysis,
    format_metrics_report,
)

logger = setup_logger("ablation")


def run_ablation(
    max_pages: int = 2,
    max_docs: int = 2,
    dpi: int = 200,
    save_results: bool = True,
) -> dict:
    """
    Ablation study: compare pipeline with and without LLM correction.

    Conditions:
    1. BASELINE (TrOCR only)
    2. + LLM correction (TrOCR → LLM)
    3. + VLM + LLM (TrOCR + VLM → LLM fusion) [if GPU allows]

    Uses transcribed pages with known ground truth for evaluation.

    Args:
        max_pages: Max pages per document to process.
        max_docs: Max documents to process.
        dpi: DPI for PDF conversion.
        save_results: Whether to save results to disk.

    Returns:
        Dict with ablation results.
    """
    from training.data_alignment import align_transcriptions, get_train_val_test_split
    from inference.pipeline import RenaissanceHTRPipeline

    # Get ground truth data
    logger.info("Loading ground truth data...")
    aligned = align_transcriptions(max_pages_per_doc=max_pages, dpi=dpi)

    # Use validation set for ablation (don't contaminate training evaluation)
    _, val_data, _ = get_train_val_test_split(aligned)
    if not val_data:
        # If no val data, use first few training samples
        train_data, _, _ = get_train_val_test_split(aligned)
        val_data = train_data[:max_pages * max_docs]

    logger.info(f"Using {len(val_data)} pages for ablation")

    results = {}

    # ─── Condition 1: Baseline (Minimal Prototype / TrOCR only) ───
    logger.info("\n{'='*50}")
    logger.info("Condition 1: BASELINE (Minimal Prototype - TrOCR only)")
    logger.info("{'='*50}")

    pipeline_baseline = RenaissanceHTRPipeline(mode="minimal", load_models=True)
    baseline_results = pipeline_baseline.process_batch(val_data)
    
    baseline_preds = [r["transcription"] for r in baseline_results]
    baseline_refs = [page["text"] for page in val_data]

    baseline_cer = compute_cer(baseline_preds, baseline_refs)
    baseline_wer = compute_wer(baseline_preds, baseline_refs)

    results["baseline"] = {
        "cer": baseline_cer,
        "wer": baseline_wer,
        "method": "Minimal Prototype (TrOCR only)",
        "uncertainty": uncertainty_analysis(baseline_preds, baseline_refs),
    }

    print(format_metrics_report(baseline_cer, baseline_wer,
          results["baseline"]["uncertainty"], "BASELINE (TrOCR only)"))


    # ─── Condition 2: VLM-Centric Pipeline ───
    logger.info("\n{'='*50}")
    logger.info("Condition 2: PROPOSED PIPELINE (VLM + LLM Fusion)")
    logger.info("{'='*50}")

    try:
        pipeline_vlm = RenaissanceHTRPipeline(mode="vlm_centric", load_models=True)
        vlm_results = pipeline_vlm.process_batch(val_data)
        
        vlm_preds = [r["transcription"] for r in vlm_results]

        vlm_cer = compute_cer(vlm_preds, baseline_refs)
        vlm_wer = compute_wer(vlm_preds, baseline_refs)

        cer_improvement = ((baseline_cer - vlm_cer) / max(baseline_cer, 1e-8)) * 100
        wer_improvement = ((baseline_wer - vlm_wer) / max(baseline_wer, 1e-8)) * 100

        results["proposed_pipeline"] = {
            "cer": vlm_cer,
            "wer": vlm_wer,
            "method": "VLM + LLM Fusion",
            "cer_improvement_pct": cer_improvement,
            "wer_improvement_pct": wer_improvement,
            "uncertainty": uncertainty_analysis(vlm_preds, baseline_refs),
        }

        print(format_metrics_report(vlm_cer, vlm_wer,
              results["proposed_pipeline"]["uncertainty"], "PROPOSED PIPELINE (VLM + LLM)"))
        print(f"  CER improvement vs baseline: {cer_improvement:+.1f}%")
        print(f"  WER improvement vs baseline: {wer_improvement:+.1f}%")

    except Exception as e:
        logger.warning(f"Could not run proposed pipeline condition: {e}")
        results["proposed_pipeline"] = {"error": str(e)}

    # ─── Summary Table ───
    print("\n" + "=" * 60)
    print("  ABLATION SUMMARY (PROPOSED vs MINIMAL)")
    print("=" * 60)
    print(f"  {'Condition':<35} {'CER':>8} {'WER':>8} {'Δ CER':>8}")
    print(f"  {'─' * 59}")

    for key in ["baseline", "proposed_pipeline"]:
        if key in results and "error" not in results[key]:
            r = results[key]
            delta = f"{r.get('cer_improvement_pct', 0):+.1f}%" if key != "baseline" else "—"
            print(f"  {r['method']:<35} {r['cer']:>8.4f} {r['wer']:>8.4f} {delta:>8}")

    print("=" * 60)

    # Save results
    if save_results:
        serializable = {}
        for k, v in results.items():
            serializable[k] = {
                sk: sv for sk, sv in v.items()
                if isinstance(sv, (str, int, float, dict, list, bool))
            }

        results_path = RESULTS_DIR / "ablation_results.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(serializable, f, indent=2)
        logger.info(f"Results saved to {results_path}")

    return results

if __name__ == "__main__":
    run_ablation(max_pages=2, max_docs=2)
