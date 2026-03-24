"""
Evaluation metrics for historical text recognition.
CER, WER, confidence scoring, and uncertainty analysis.
"""
import numpy as np
from typing import Optional


def compute_cer(predictions: list[str], references: list[str]) -> float:
    """
    Compute Character Error Rate (CER).

    CER = edit_distance(pred_chars, ref_chars) / len(ref_chars)

    Args:
        predictions: List of predicted strings.
        references: List of reference (ground truth) strings.

    Returns:
        Average CER across all pairs.
    """
    try:
        from jiwer import cer
        return cer(references, predictions)
    except ImportError:
        # Fallback: manual CER computation
        import editdistance

        total_dist = 0
        total_len = 0
        for pred, ref in zip(predictions, references):
            total_dist += editdistance.eval(pred, ref)
            total_len += max(len(ref), 1)
        return total_dist / max(total_len, 1)


def compute_wer(predictions: list[str], references: list[str]) -> float:
    """
    Compute Word Error Rate (WER).

    WER = edit_distance(pred_words, ref_words) / len(ref_words)

    Args:
        predictions: List of predicted strings.
        references: List of reference (ground truth) strings.

    Returns:
        Average WER across all pairs.
    """
    try:
        from jiwer import wer
        return wer(references, predictions)
    except ImportError:
        import editdistance

        total_dist = 0
        total_len = 0
        for pred, ref in zip(predictions, references):
            pred_words = pred.split()
            ref_words = ref.split()
            total_dist += editdistance.eval(pred_words, ref_words)
            total_len += max(len(ref_words), 1)
        return total_dist / max(total_len, 1)


def confidence_score(token_probabilities: list[float]) -> dict:
    """
    Compute aggregate confidence metrics from per-token probabilities.

    Args:
        token_probabilities: List of softmax probabilities for each token.

    Returns:
        Dict with 'mean', 'min', 'std', 'below_threshold' metrics.
    """
    if not token_probabilities:
        return {"mean": 0.0, "min": 0.0, "std": 0.0, "below_threshold": 0}

    probs = np.array(token_probabilities)
    threshold = 0.7

    return {
        "mean": float(np.mean(probs)),
        "min": float(np.min(probs)),
        "std": float(np.std(probs)),
        "below_threshold": int(np.sum(probs < threshold)),
        "total_tokens": len(probs),
        "pct_uncertain": float(np.mean(probs < threshold) * 100),
    }


def uncertainty_analysis(
    predictions: list[str],
    references: Optional[list[str]] = None,
    token_scores: Optional[list[list[float]]] = None,
) -> dict:
    """
    Comprehensive uncertainty analysis of model outputs.

    Analyzes:
    1. Token-level confidence distribution
    2. Correlation between confidence and error rate (if references available)
    3. Per-line uncertainty classification

    Args:
        predictions: List of predicted strings.
        references: Optional ground truth for correlation analysis.
        token_scores: Optional per-token confidence scores for each prediction.

    Returns:
        Dict with uncertainty analysis results.
    """
    results = {
        "num_predictions": len(predictions),
        "avg_prediction_length": np.mean([len(p) for p in predictions]) if predictions else 0,
    }

    # Token-level analysis
    if token_scores:
        all_scores = [s for scores in token_scores for s in scores]
        results["token_confidence"] = confidence_score(all_scores)

        # Per-line confidence
        line_confidences = []
        for scores in token_scores:
            if scores:
                line_confidences.append(np.mean(scores))
            else:
                line_confidences.append(0.0)

        results["per_line_confidence"] = {
            "mean": float(np.mean(line_confidences)),
            "std": float(np.std(line_confidences)),
            "min": float(np.min(line_confidences)),
            "max": float(np.max(line_confidences)),
        }

        # Count lines by confidence bucket
        confs = np.array(line_confidences)
        results["confidence_buckets"] = {
            "high (>0.9)": int(np.sum(confs > 0.9)),
            "medium (0.7-0.9)": int(np.sum((confs >= 0.7) & (confs <= 0.9))),
            "low (<0.7)": int(np.sum(confs < 0.7)),
        }

    # Error correlation analysis
    if references:
        line_cers = []
        for pred, ref in zip(predictions, references):
            line_cer = compute_cer([pred], [ref])
            line_cers.append(line_cer)

        results["error_analysis"] = {
            "mean_cer": float(np.mean(line_cers)),
            "std_cer": float(np.std(line_cers)),
            "median_cer": float(np.median(line_cers)),
            "max_cer": float(np.max(line_cers)),
            "zero_error_lines": int(np.sum(np.array(line_cers) == 0)),
        }

        # Correlation between confidence and CER
        if token_scores and len(line_confidences) == len(line_cers):
            correlation = np.corrcoef(line_confidences, line_cers)
            results["confidence_error_correlation"] = float(correlation[0, 1])

    return results


def format_metrics_report(
    cer_val: float,
    wer_val: float,
    uncertainty: dict,
    title: str = "Evaluation Results",
) -> str:
    """Format metrics into a readable report string."""
    lines = [
        f"{'=' * 50}",
        f"  {title}",
        f"{'=' * 50}",
        f"  CER:  {cer_val:.4f}  ({cer_val * 100:.2f}%)",
        f"  WER:  {wer_val:.4f}  ({wer_val * 100:.2f}%)",
        f"{'─' * 50}",
    ]

    if "token_confidence" in uncertainty:
        tc = uncertainty["token_confidence"]
        lines.extend([
            f"  Token Confidence:",
            f"    Mean: {tc['mean']:.4f}",
            f"    Min:  {tc['min']:.4f}",
            f"    Std:  {tc['std']:.4f}",
            f"    Uncertain tokens: {tc['below_threshold']} / {tc['total_tokens']} "
            f"({tc['pct_uncertain']:.1f}%)",
        ])

    if "confidence_buckets" in uncertainty:
        cb = uncertainty["confidence_buckets"]
        lines.extend([
            f"  Line Confidence Distribution:",
            f"    High (>0.9):    {cb.get('high (>0.9)', 0)}",
            f"    Medium (0.7-0.9): {cb.get('medium (0.7-0.9)', 0)}",
            f"    Low (<0.7):     {cb.get('low (<0.7)', 0)}",
        ])

    if "confidence_error_correlation" in uncertainty:
        corr = uncertainty["confidence_error_correlation"]
        lines.append(
            f"  Confidence-Error Correlation: {corr:.4f}"
        )

    lines.append(f"{'=' * 50}")
    return "\n".join(lines)
