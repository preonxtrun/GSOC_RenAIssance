"""
Central configuration for the RenAIssance HTR Pipeline.
All paths, model names, and hyperparameters in one place.
"""

import os
from pathlib import Path

# Kaggle Auto-Detection
IS_KAGGLE = os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None

# ──────────────────────────────────────────────
# Project Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT.parent

if IS_KAGGLE:
    # Kaggle environment mapping
    HANDWRITING_PDFS_DIR = Path("/kaggle/input/renaissance-dataset/Handwriting")
    HANDWRITING_TRANSCRIPTIONS_DIR = Path("/kaggle/input/renaissance-dataset/Handwriting_Transcriptions")
    PRINT_PDFS_DIR = Path("/kaggle/input/renaissance-dataset/Print")
    PRINT_TRANSCRIPTIONS_DIR = Path("/kaggle/input/renaissance-dataset/Print_Transcriptions")
    OUTPUT_DIR = Path("/kaggle/working/output")
else:
    # Dataset directories
    HANDWRITING_PDFS_DIR = DATA_DIR / "OneDrive_1_3-15-2026" / "Handwriting"
    HANDWRITING_TRANSCRIPTIONS_DIR = DATA_DIR / "OneDrive_1_3-15-2026 (1)" / "Handwriting"
    PRINT_PDFS_DIR = DATA_DIR / "OneDrive_1_3-15-2026" / "Print"
    PRINT_TRANSCRIPTIONS_DIR = DATA_DIR / "OneDrive_1_3-15-2026 (1)" / "Print"
    
    # Output directories
    OUTPUT_DIR = PROJECT_ROOT / "output"

IMAGES_DIR = OUTPUT_DIR / "images"
PREPROCESSED_DIR = OUTPUT_DIR / "preprocessed"
RESULTS_DIR = OUTPUT_DIR / "results"
MODELS_DIR = OUTPUT_DIR / "models"

for d in [OUTPUT_DIR, IMAGES_DIR, PREPROCESSED_DIR, RESULTS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Model Configuration
# ──────────────────────────────────────────────

# OCR Backbone
TROCR_MODEL_NAME = "microsoft/trocr-large-handwritten"
TROCR_PROCESSOR_NAME = "microsoft/trocr-large-handwritten"

# VLM (optional fallback only)
USE_VLM = True
VLM_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

# LLM Corrector
LLM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# ──────────────────────────────────────────────
# Preprocessing Parameters
# ──────────────────────────────────────────────
PDF_DPI = 300
DENOISE_STRENGTH = 10
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = (8, 8)
SKEW_MAX_ANGLE = 15

# ──────────────────────────────────────────────
# Training Hyperparameters
# ──────────────────────────────────────────────
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
USE_FP16 = True

LEARNING_RATE = 5e-5
NUM_EPOCHS = 3
WARMUP_STEPS = 100
MAX_SEQ_LENGTH = 128
TRAIN_VAL_SPLIT = 0.85

USE_LORA = True
LORA_RANK = 32
LORA_ALPHA = 64

# ──────────────────────────────────────────────
# Inference Parameters
# ──────────────────────────────────────────────
BEAM_SEARCH_WIDTH = 8

VLM_MAX_NEW_TOKENS = 512
LLM_MAX_NEW_TOKENS = 512
QUANTIZE_4BIT = True
CONFIDENCE_THRESHOLD = 0.7
VLM_FALLBACK_THRESHOLD = 0.4

# ──────────────────────────────────────────────
# Dataset file mappings (Handwriting)
# ──────────────────────────────────────────────
HANDWRITING_PAIRS = [
    {
        "pdf": "AHPG_GPAH_1.pdf",
        "transcription": "AHPG_GPAH_1_trans.docx",
        "name": "AHPG-GPAH_1744",
    },
    {
        "pdf": "AHPG_GPAH_AU61.pdf",
        "transcription": "AHPG_GPAH_AU61_trans.docx",
        "name": "AHPG-GPAH_1606",
    },
    {
        "pdf": "ES_AHN.pdf",
        "transcription": "ES_AHN_trans.docx",
        "name": "AHN_Inquisicion_1640",
    },
    {
        "pdf": "PT3279.pdf",
        "transcription": "PT3279_trans.docx",
        "name": "PT3279_1857",
    },
    {
        "pdf": "Pleito_Viana.pdf",
        "transcription": "Pleito_Viana_trans.docx",
        "name": "Pleito_Viana",
    },
]