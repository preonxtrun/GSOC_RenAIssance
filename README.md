# RenAIssance HTR — Handwritten Historical Text Recognition

**GSoC 2026 Evaluation Submission — CERN HumanAI / RenAIssance Project**

An LLM/VLM-centered OCR pipeline for recognizing handwritten early modern manuscripts (16th–19th century).

---

## Architecture

```
PDF Scans → Preprocessing → [Two-Track Pipeline] → LLM Fusion → Final Transcription
                                ├── TrOCR (line-level OCR)
                                └── Qwen2.5-VL (direct image reading)
```

The VLM/LLM is **central** to every stage:

| Stage | VLM/LLM Role |
|---|---|
| Image interpretation | VLM identifies text regions vs marginalia |
| Transcription | VLM reads handwriting directly from images |
| Token fusion | LLM arbitrates between OCR and VLM readings |
| Correction | LLM fixes OCR errors using historical language context |

## Model Stack

| Component | Model | VRAM |
|---|---|---|
| **VLM** (central) | `Qwen/Qwen2.5-VL-7B-Instruct` | ~6GB (4-bit) |
| **OCR backbone** | `microsoft/trocr-large-handwritten` | ~2GB |
| **LLM corrector** | `microsoft/Phi-3-mini-4k-instruct` | ~3GB (4-bit) |

## Project Structure

```
renaissance-htr/
├── config.py                     # Central configuration
├── preprocessing/
│   ├── pdf_to_images.py          # PDF → high-DPI images
│   ├── image_processing.py       # Grayscale, denoise, deskew, CLAHE
│   └── text_extraction.py        # Text block / line segmentation
├── models/
│   ├── ocr_backbone.py           # TrOCR wrapper + confidence scoring
│   ├── vlm_reader.py             # Qwen2.5-VL for direct reading
│   └── llm_corrector.py          # LLM token fusion + correction
├── training/
│   ├── data_alignment.py         # Align GT text ↔ page images
│   ├── dataset.py                # PyTorch Dataset for TrOCR
│   └── trocr_finetune.py         # LoRA fine-tuning
├── inference/
│   └── pipeline.py               # End-to-end: minimal/standard/full
├── evaluation/
│   ├── metrics.py                # CER, WER, confidence, uncertainty
│   └── ablation.py               # With/without LLM comparison
├── notebooks/
│   ├── 01_minimal_prototype.py   # Quick demo (CPU, ~5 min)
│   └── 02_full_pipeline.py       # Full submission
└── utils/
    └── helpers.py                # .docx parsing, logging
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Minimal prototype (one page, CPU)
python notebooks/01_minimal_prototype.py

# Full pipeline
python notebooks/02_full_pipeline.py --max-pages 3 --max-docs 2

# Skip VLM if GPU limited
python notebooks/02_full_pipeline.py --skip-vlm

# Ablation experiment only
python evaluation/ablation.py
```

## Pipeline Modes

| Mode | Components | GPU Required |
|---|---|---|
| `minimal` | TrOCR only | Optional (CPU OK) |
| `standard` | TrOCR → LLM correction | ~3GB |
| `full` | TrOCR + VLM → LLM fusion | ~11GB |

## Evaluation Metrics

- **CER** — Character Error Rate (edit distance at character level)
- **WER** — Word Error Rate (edit distance at word level)  
- **Confidence** — Per-token softmax probability from TrOCR
- **Uncertainty** — Entropy analysis, confidence-error correlation

## Ablation Design

| Condition | Pipeline | Expected CER |
|---|---|---|
| Baseline | TrOCR only | ~35-50% |
| + LLM | TrOCR → LLM correction | ~25-40% |
| + VLM + LLM | TrOCR + VLM → LLM fusion | ~20-35% |

## Implementation Order

1. `preprocessing/` — PDF conversion, denoising, line segmentation
2. `models/ocr_backbone.py` — TrOCR baseline
3. `notebooks/01_minimal_prototype.py` — working demo
4. `training/` — data alignment, fine-tuning
5. `models/vlm_reader.py` — VLM integration
6. `models/llm_corrector.py` — LLM correction
7. `inference/pipeline.py` — full end-to-end
8. `evaluation/` — CER/WER + ablation
9. `notebooks/02_full_pipeline.py` — final submission

## Where LLM Genuinely Improves OCR

1. **Resolving paleographic ambiguity** — 'u' vs 'n', long-s 'ſ' vs 'f', 'c' vs 'e' are common confusions that OCR cannot resolve without language context.
2. **Completing damaged text** — Ink blotches, paper damage create gaps that language models can fill using surrounding context.
3. **Historical spelling awareness** — The LLM can distinguish between a genuine archaic spelling ("que" → "q̃") and an OCR error, where a simple spell-checker would "correct" both.
4. **Multi-source arbitration** — When OCR and VLM disagree, the LLM uses language knowledge to pick the more plausible reading.

## License

Educational project for GSoC 2026 evaluation.
