"""
VLM (Vision-Language Model) integration for direct handwriting interpretation.

This is where the LLM/VLM is CENTRAL to the pipeline — not just post-correction.
The VLM reads handwritten regions directly from images, providing:
1. Region-level text interpretation (what does this handwriting say?)
2. Layout understanding (where is main text vs marginalia?)
3. Contextual transcription (using surrounding context to resolve ambiguity)
"""
import sys
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import VLM_MODEL_NAME, VLM_MAX_NEW_TOKENS, QUANTIZE_4BIT
from utils.helpers import get_device, setup_logger

logger = setup_logger("vlm-reader")


# ──────────────────────────────────────────────
# Prompt templates for different VLM tasks
# ──────────────────────────────────────────────

LAYOUT_ANALYSIS_PROMPT = """You are analyzing a scanned page from a historical manuscript (16th-19th century).
Identify the layout regions:
1. MAIN TEXT: The primary handwritten text block.
2. MARGINALIA: Any side notes, annotations, or marginal text.
3. DECORATIONS: Ornamental elements, stamps, or non-text features.
4. HEADERS/FOOTERS: Page numbers, running titles.

Describe what you see and where the main text block is located.
Focus ONLY on the main text area for transcription."""

TRANSCRIPTION_PROMPT = """You are a specialist in historical handwriting recognition (paleography).
This image shows a page of handwritten text from an early modern manuscript (16th-19th century in Spanish/Latin).

FEW-SHOT EXAMPLES of historical spellings to preserve:
- Original: "vn ombre" → Keep as-is (not "un hombre")
- Original: "dexaron" → Keep as-is (not "dejaron")
- Original: "ſu merced" → Transcribe as "su merced" (long-s becomes regular s)
- Original: "Vuestra Señoria" → Keep as-is (abbreviated "V.S." is common)

CRITICAL HISTORICAL SPELLING RULES:
- 'u' and 'v' are interchangeable: transcribe exactly what is written.
- Long-s ('ſ') looks like 'f': use context to determine which it is (usually 's' in medial position).
- 'ç' (c-cedilla) is historical spelling: PRESERVE it exactly as written.
- IGNORE marginalia, stamps, archivist notes, and page numbers — focus ONLY on the main text.
- If a word is unclear, add [?] after it. Mark fully illegible words as [illegible].
- Preserve line breaks exactly as they appear in the original image.

Transcribe the main handwritten text in this image line by line:"""

CORRECTION_PROMPT_TEMPLATE = """You are a historical text specialist. Below is an OCR-generated transcription 
of a handwritten early modern manuscript. Some words may have been misread by the OCR system.

OCR Transcription:
{ocr_text}

{uncertain_info}

Please review and correct the transcription:
1. Fix obvious OCR misreadings based on common paleographic confusions 
   (e.g., 'n' vs 'u', 'c' vs 'e', long-s 'ſ' read as 'f').
2. PRESERVE historical spellings — do NOT modernize.
3. Use your knowledge of early modern language patterns.
4. Mark still-uncertain words with [?].

Corrected transcription:"""


class VLMReader:
    """
    Vision-Language Model reader for direct handwriting interpretation.

    The VLM serves as the CENTRAL intelligence in the pipeline:
    - It reads handwriting directly (not just correcting OCR output)
    - It understands page layout to filter marginalia
    - It uses historical language knowledge for contextual interpretation
    """

    def __init__(
        self,
        model_name: str = VLM_MODEL_NAME,
        quantize: bool = QUANTIZE_4BIT,
        device: str | torch.device | None = None,
    ):
        self.model_name = model_name
        self.quantize = quantize
        self.device = device or get_device()
        self.model = None
        self.processor = None
        self._loaded = False

        logger.info(f"VLM Reader initialized (model: {model_name}, lazy loading)")

    def load_model(self):
        """
        Lazy-load the VLM model. Called on first use.
        Uses 4-bit quantization to fit in limited VRAM.
        """
        if self._loaded:
            return

        logger.info(f"Loading VLM: {self.model_name} (4-bit={self.quantize})")

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

            load_kwargs = {"torch_dtype": torch.float16}

            if self.quantize and torch.cuda.is_available():
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                load_kwargs["quantization_config"] = quantization_config
                load_kwargs["device_map"] = "auto"
            else:
                load_kwargs["device_map"] = "auto"

            # Dramatically increase max_pixels to allow the VLM to read at a higher resolution.
            # 3136 * 28 * 28 allows for ~2.4 Megapixel image patches, keeping 40-line pages highly legible 
            # while still easily fitting within Colab's 15GB VRAM (since the 3B 4-bit model only uses ~2.5GB).
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, 
                min_pixels=256*28*28, 
                max_pixels=3136*28*28,
            )
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name, **load_kwargs
            )
            self._loaded = True
            logger.info("VLM loaded successfully")

        except Exception as e:
            logger.warning(f"Could not load Qwen2.5-VL: {e}")
            logger.warning("Falling back to a lighter VLM or mock mode")
            self._loaded = False

    def unload_model(self):
        """Free memory by unloading the VLM from RAM and VRAM."""
        if not self._loaded:
            return
        logger.info("Unloading VLM to free memory...")
        import gc
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _generate(
        self, image: Image.Image, prompt: str, max_tokens: int = VLM_MAX_NEW_TOKENS
    ) -> str:
        """Run VLM inference on an image with a text prompt."""
        self.load_model()

        if not self._loaded:
            return "[VLM not available — skipping direct reading]"

        # Build messages in Qwen2.5-VL format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Process inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
            )

        # Decode only the new tokens
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        response = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        return response.strip()

    def analyze_layout(self, page_image: Image.Image) -> str:
        """
        Use VLM to understand page layout and identify text vs marginalia.
        This is where VLM goes beyond traditional OCR.
        """
        if isinstance(page_image, np.ndarray):
            page_image = Image.fromarray(page_image)
        if page_image.mode != "RGB":
            page_image = page_image.convert("RGB")

        return self._generate(page_image, LAYOUT_ANALYSIS_PROMPT, max_tokens=256)

    def transcribe_region(self, region_image: Image.Image) -> str:
        """
        Directly read and transcribe handwritten text from an image region.
        First analyses page layout to identify the main text block,
        then transcribes only that area — filtering margins and stamps.
        """
        if isinstance(region_image, np.ndarray):
            region_image = Image.fromarray(region_image)
        if region_image.mode != "RGB":
            region_image = region_image.convert("RGB")

        # Stage 1: Understand the layout to focus the transcription
        layout_context = self._generate(
            region_image, LAYOUT_ANALYSIS_PROMPT, max_tokens=150
        )

        # Stage 2: Transcribe using the layout as additional context
        focused_prompt = (
            TRANSCRIPTION_PROMPT
            + f"\n\nPage layout note (use to focus on main text only):\n{layout_context}"
        )
        return self._generate(region_image, focused_prompt)

    def correct_with_context(
        self,
        ocr_text: str,
        image: Optional[Image.Image] = None,
        uncertain_words: Optional[list[str]] = None,
    ) -> str:
        """
        Use VLM to correct OCR output using both the image and language context.
        This demonstrates VLM's role beyond simple post-correction:
        it can re-examine the image for uncertain words.
        """
        uncertain_info = ""
        if uncertain_words:
            uncertain_info = (
                f"Words flagged as uncertain by OCR: {', '.join(uncertain_words)}\n"
                "Please pay special attention to these words."
            )

        prompt = CORRECTION_PROMPT_TEMPLATE.format(
            ocr_text=ocr_text, uncertain_info=uncertain_info
        )

        if image is not None:
            # VLM can re-examine the image to verify uncertain words
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            if image.mode != "RGB":
                image = image.convert("RGB")
            return self._generate(image, prompt)
        else:
            # Text-only correction (fallback)
            return self._generate(
                Image.new("RGB", (100, 100), "white"), prompt
            )
