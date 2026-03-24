"""
LLM-based correction and token fusion.
Takes OCR output + VLM output and produces a corrected transcription
with confidence scoring and historical spelling preservation.
"""
import sys
from pathlib import Path

import torch
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import LLM_MODEL_NAME, LLM_MAX_NEW_TOKENS, QUANTIZE_4BIT
from utils.helpers import get_device, setup_logger

logger = setup_logger("llm-corrector")


# ──────────────────────────────────────────────
# Prompt templates for LLM correction tasks
# ──────────────────────────────────────────────

FUSION_PROMPT_TEMPLATE = """You are a specialist in early modern historical texts and paleography.
You have received TWO independent transcriptions of the same handwritten manuscript passage:

=== OCR Engine Transcription (TrOCR) ===
{ocr_text}

=== VLM Direct Reading (Vision Model) ===
{vlm_text}

FEW-SHOT FUSION EXAMPLES (to guide your approach):
- TrOCR: "vn ombre fue al" | VLM: "un hombre fue al" → Fused: "vn ombre fue al" (preserve TrOCR's orthography)
- TrOCR: "fi hiziera" | VLM: "si hiciera" → Fused: "si hiziera" (VLM grammar: 'si' not 'fi'; TrOCR spelling: 'hiziera' not 'hiciera')
- TrOCR: "XXX" | VLM: "Vuestra Señoria" → Fused: "Vuestra Señoria" (VLM wins on hallucinated/missed tokens)

Your fusion rules:
1. The OCR engine (TrOCR) is highly accurate character-by-character, but sometimes loses grammatical flow.
2. The Vision Model (VLM) captures the overall grammatical structure, but frequently hallucinates specific characters.
3. Adopt the grammatical structure and sentence coherence of the VLM.
4. Fiercely enforce the exact orthography and character spellings from TrOCR wherever they differ from VLM.
5. PRESERVE historical spellings (e.g., 'ç', 'v' for 'u', long 's') — do NOT modernize.
6. Mark remaining uncertain words with [?].
7. Output ONLY the fused corrected transcription, line by line.

Fused corrected transcription:"""

CORRECTION_ONLY_PROMPT_TEMPLATE = """You are a specialist in early modern historical texts (Spanish/Latin).
Below is OCR output from a handwritten manuscript. Some words may be misread.

OCR Output:
{ocr_text}

{uncertainty_note}

PALEOGRAPHY INSTRUCTIONS:
1. Fix obvious OCR errors based on the following known 16th-century patterns:
   - 'u' and 'v' are often used interchangeably (e.g., 'vn' vs 'un').
   - 'f' and 's' (long s) look identical in the handwriting. Fix based on contextual grammar.
   - 'ç' (c-cedilla) is a historical spelling that always maps to modern 'z'. Do not change it to 'c'.
2. PRESERVE all historical spellings — do NOT modernize unless fixing an obvious machine hallucination.
3. Use your knowledge of early modern Spanish/Latin language patterns.
4. Mark uncertain words with [?].
5. Output ONLY the corrected text.

Corrected text:"""


class LLMCorrector:
    """
    LLM-based corrector for OCR/VLM token fusion and historical text correction.

    Role in the pipeline:
    - Arbitrates between OCR and VLM transcriptions (token fusion)
    - Corrects common OCR misreadings using language context
    - Preserves historical spellings while fixing machine errors
    - Provides uncertainty annotations
    """

    def __init__(
        self,
        model_name: str = LLM_MODEL_NAME,
        quantize: bool = QUANTIZE_4BIT,
        device: str | torch.device | None = None,
    ):
        self.model_name = model_name
        self.quantize = quantize
        self.device = device or get_device()
        self.model = None
        self.tokenizer = None
        self._loaded = False

        logger.info(f"LLM Corrector initialized (model: {model_name}, lazy loading)")

    def load_model(self):
        """Lazy-load the LLM model with optional 4-bit quantization."""
        if self._loaded:
            return

        logger.info(f"Loading LLM: {self.model_name} (4-bit={self.quantize})")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

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

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, **load_kwargs
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self._loaded = True
            logger.info("LLM loaded successfully")

        except Exception as e:
            logger.warning(f"Could not load LLM: {e}")
            self._loaded = False

    def unload_model(self):
        """Free memory by unloading the LLM from RAM and VRAM."""
        if not self._loaded:
            return
        logger.info("Unloading LLM to free memory...")
        import gc
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _generate(self, prompt: str, max_tokens: int = LLM_MAX_NEW_TOKENS) -> str:
        """Run LLM inference on a text prompt."""
        self.load_model()

        if not self._loaded:
            return "[LLM not available — returning uncorrected text]"

        # Format for instruction-tuned models
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode only new tokens
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        response = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        return response.strip()

    def fuse_transcriptions(self, ocr_text: str, vlm_text: str) -> str:
        """
        Fuse OCR and VLM transcriptions using LLM as arbitrator.

        This is where the LLM adds genuine value:
        - It resolves disagreements between two independent readings
        - It uses language model knowledge to pick the most likely word
        - It can resolve ambiguities neither OCR nor VLM could alone
        """
        prompt = FUSION_PROMPT_TEMPLATE.format(
            ocr_text=ocr_text, vlm_text=vlm_text
        )
        return self._generate(prompt)

    def correct_ocr(
        self,
        ocr_text: str,
        uncertain_words: Optional[list[str]] = None,
    ) -> str:
        """
        Correct OCR output using LLM language knowledge.
        Used when VLM reading is not available (lighter pipeline).
        """
        uncertainty_note = ""
        if uncertain_words:
            uncertainty_note = (
                f"Words flagged as low-confidence: {', '.join(uncertain_words)}\n"
                "Pay special attention to these."
            )

        prompt = CORRECTION_ONLY_PROMPT_TEMPLATE.format(
            ocr_text=ocr_text, uncertainty_note=uncertainty_note
        )
        return self._generate(prompt)

    def batch_correct(
        self,
        page_results: list[dict],
    ) -> list[dict]:
        """
        Correct multiple line results, focusing on low-confidence lines.

        Args:
            page_results: List of dicts with 'text', 'confidence', 'is_uncertain'.

        Returns:
            Updated results with 'corrected_text' field added.
        """
        corrected = []
        for result in page_results:
            if result.get("is_uncertain", False):
                # Correct uncertain lines with LLM
                result["corrected_text"] = self.correct_ocr(result["text"])
                result["correction_applied"] = True
            else:
                result["corrected_text"] = result["text"]
                result["correction_applied"] = False
            corrected.append(result)

        return corrected
