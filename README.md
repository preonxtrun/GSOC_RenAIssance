# RenAIssance HTR — Handwritten Historical Text Recognition

**GSoC 2026 Evaluation Submission — CERN HumanAI / RenAIssance Project**

A VLM/LLM-centered OCR pipeline for recognizing handwritten early modern manuscripts (16th–19th century).

---

## Architecture

The pipeline supports **four operating modes**, with the VLM-centric mode as the proposed approach:

```
                    ┌─────────────────────────────────────────────────────┐
                    │           VLM-Centric Pipeline (Proposed)          │
  PDF Scans ──►     │  Qwen2.5-VL ──► Layout Analysis ──► Transcription  │ ──► LLM Correction ──► Output
                    └─────────────────────────────────────────────────────┘

                    ┌─────────────────────────────────────────────────────┐
                    │                 Full Dual-Track                     │
  PDF Scans ──►     │  TrOCR (line-level OCR) ─────────────┐             │ ──► LLM Fusion ──► Output
                    │  Qwen2.5-VL (direct reading) ────────┘             │
                    └─────────────────────────────────────────────────────┘
```

The VLM/LLM is **central** to every stage:

| Stage | VLM/LLM Role |
|---|---|
| Layout analysis | VLM identifies main text vs marginalia, stamps, archivist notes |
| Direct transcription | VLM reads handwriting directly from full-page images |
| OCR post-processing | CLIP reranks TrOCR beam candidates for semantic consistency |
| Token fusion | LLM arbitrates between OCR and VLM readings using few-shot examples |
| Correction | LLM fixes OCR errors using historical paleography context |

## Model Stack

| Component | Model | VRAM |
|---|---|---|
| **VLM** (central) | `Qwen/Qwen2.5-VL-7B-Instruct` | ~6GB (4-bit) |
| **OCR backbone** | `microsoft/trocr-large-handwritten` | ~2GB |
| **CLIP reranker** | `openai/clip-vit-base-patch32` | ~0.4GB |
| **LLM corrector** | `Qwen/Qwen2.5-7B-Instruct` | ~5GB (4-bit) |

> Models are loaded and unloaded sequentially (stage-by-stage) to fit within a single T4 GPU (15GB VRAM).

## Project Structure

```
renaissance-htr/
├── config.py                     # Central configuration (paths, models, hyperparams)
├── requirements.txt              # Python dependencies
├── preprocessing/
│   ├── pdf_to_images.py          # PDF → high-DPI page images (PyMuPDF)
│   ├── image_processing.py       # Grayscale, denoise, deskew, CLAHE
│   └── text_extraction.py        # Text line segmentation via projection profiles
├── models/
│   ├── ocr_backbone.py           # TrOCR wrapper + CLIP beam reranking + confidence
│   ├── vlm_reader.py             # Qwen2.5-VL: layout analysis → focused transcription
│   └── llm_corrector.py          # Few-shot LLM fusion and correction
├── training/
│   ├── data_alignment.py         # Align GT .docx ↔ page images (archivist note filtering)
│   ├── dataset.py                # PyTorch Dataset for TrOCR line-level training
│   └── trocr_finetune.py         # LoRA fine-tuning with cosine LR + gradient accumulation
├── inference/
│   └── pipeline.py               # End-to-end: minimal / standard / full / vlm_centric
├── evaluation/
│   ├── metrics.py                # CER, WER, confidence scoring, uncertainty analysis
│   └── ablation.py               # Baseline vs proposed pipeline comparison
├── notebooks/
│   ├── 01_minimal_prototype.py   # Quick demo (CPU, ~5 min, one page)
│   ├── 02_full_pipeline.py       # Full submission with training + evaluation
│   └── GSoC_RenAIssance_Colab.ipynb   # Google Colab notebook
└── utils/
    └── helpers.py                # .docx parsing, device detection, logging
```

## Quick Start

### Local

```bash
# Install dependencies
pip install -r requirements.txt

# Minimal prototype — one page, CPU, no VLM/LLM required
python notebooks/01_minimal_prototype.py

# Full pipeline — training + dual-track inference + evaluation
python notebooks/02_full_pipeline.py --max-pages 3 --max-docs 2

# Skip VLM if GPU limited (TrOCR → LLM correction only)
python notebooks/02_full_pipeline.py --skip-vlm

# Skip training (use pretrained TrOCR)
python notebooks/02_full_pipeline.py --skip-training

# Ablation experiment only (baseline vs proposed)
python evaluation/ablation.py
```

### Cloud (Recommended)

The pipeline is designed to run on a free-tier GPU:

| Platform | Notebook | GPU |
|---|---|---|
| **Colab** | `notebooks/GSoC_RenAIssance_Colab.ipynb` | T4 (15GB) |

## Pipeline Modes

| Mode | Components | GPU Required | Use Case |
|---|---|---|---|
| `minimal` | TrOCR only | Optional (CPU OK) | Quick baseline demo |
| `standard` | TrOCR → LLM correction | ~5GB | Limited GPU |
| `full` | TrOCR + VLM → LLM fusion | ~11GB | Best quality (dual-track) |
| `vlm_centric` | VLM → LLM correction | ~11GB | **Proposed pipeline** |

## Key Design Decisions

### Sequential Model Loading

All models cannot fit in VRAM simultaneously. The pipeline processes data in **phases**, loading one model at a time:

```
Phase 1: Load TrOCR → OCR all pages → Unload TrOCR
Phase 2: Load VLM → Read all pages → Unload VLM
Phase 3: Load LLM → Correct/fuse all pages → Unload LLM
```

### VLM Two-Stage Reading

The VLM doesn't just read text — it first analyzes the page layout, then uses that layout context to focus transcription on the main text body, filtering out marginalia, stamps, and archivist annotations.

### Few-Shot Paleography Prompts

The LLM fusion prompt includes few-shot examples that teach the model paleographic conventions:
- TrOCR: `"fi hiziera"` | VLM: `"si hiciera"` → Fused: `"si hiziera"` (VLM grammar + TrOCR historical spelling)
- Preserve `ç`, long-s `ſ`, and `u`/`v` interchangeability
- Never modernize archaic spellings

### Ground Truth Cleaning

Transcription `.docx` files contain modern archivist notes (e.g., "Notes:", "dictionary?", "accents are inconsistent"). The data alignment pipeline automatically filters these out before training and evaluation.

### CLIP Beam Reranking

TrOCR generates multiple beam candidates. CLIP scores each candidate against the original line image and picks the one most semantically consistent with what the image shows — catching OCR hallucinations that beam search alone misses.

## Evaluation Metrics

- **CER** — Character Error Rate (Levenshtein distance at character level)
- **WER** — Word Error Rate (Levenshtein distance at word level)
- **Confidence** — Per-token softmax probability from TrOCR
- **Uncertainty** — Entropy analysis, confidence-error correlation, per-line confidence buckets

## Ablation Design

The ablation compares the minimal prototype (TrOCR only) against the proposed VLM-centric pipeline:

| Condition | Pipeline | Description |
|---|---|---|
| **Baseline** | `minimal` — TrOCR only | OCR backbone without any LLM/VLM |
| **Proposed** | `vlm_centric` — VLM + LLM | VLM direct reading → LLM correction |

The ablation uses the validation split to avoid contaminating training evaluation.

## Where LLM/VLM Genuinely Improves OCR

1. **Resolving paleographic ambiguity** — `u` vs `n`, long-s `ſ` vs `f`, `c` vs `e` are common confusions that OCR cannot resolve without language context.
2. **Completing damaged text** — Ink blotches, paper damage create gaps that language models can fill using surrounding context.
3. **Historical spelling awareness** — The LLM can distinguish between a genuine archaic spelling ("que" → "q̃") and an OCR error, where a simple spell-checker would "correct" both.
4. **Multi-source arbitration** — When OCR and VLM disagree, the LLM uses language knowledge to pick the more plausible reading.
5. **Layout-aware reading** — The VLM separates main text from marginalia and stamps, which traditional OCR pipelines cannot do without separate layout detection models.

## License

Educational project for GSoC 2026 evaluation.
