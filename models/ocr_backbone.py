"""
TrOCR-based handwritten text recognition engine.
Wraps microsoft/trocr-large-handwritten for line-level OCR with confidence scoring.
"""
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TROCR_MODEL_NAME, TROCR_PROCESSOR_NAME, CONFIDENCE_THRESHOLD, BEAM_SEARCH_WIDTH
from utils.helpers import get_device, setup_logger

logger = setup_logger("trocr")

# CLIP reranker — loaded lazily only once
_clip_model = None
_clip_processor = None

def _get_clip():
    """Lazily load CLIP for reranking. Cached globally to avoid repeated loads."""
    global _clip_model, _clip_processor
    if _clip_model is None:
        try:
            from transformers import CLIPModel, CLIPProcessor
            logger.info("Loading CLIP for beam reranking...")
            _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            _clip_model.eval()
        except Exception as e:
            logger.warning(f"CLIP not available, skipping reranking: {e}")
    return _clip_model, _clip_processor


class TrOCREngine:
    """
    TrOCR-based OCR engine for handwritten text lines.

    Why TrOCR:
    - Pre-trained specifically on handwritten text (IAM dataset)
    - Encoder-decoder architecture: ViT encoder + GPT-2 decoder
    - Easy to fine-tune on domain-specific handwriting
    - Returns token-level logits for confidence scoring
    """

    def __init__(
        self,
        model_dir: str | Path | None = None,
        device: str | torch.device | None = None,
    ):
        self.device = device or get_device()
        from config import MODELS_DIR
        
        # Auto-detect trained models prioritizing the best checkpoint
        if model_dir is None:
            best_model_path = MODELS_DIR / "trocr-finetuned-best"
            final_model_path = MODELS_DIR / "trocr-finetuned-final"
            if best_model_path.exists():
                model_dir = best_model_path
            elif final_model_path.exists():
                model_dir = final_model_path
                
        if model_dir is not None:
            logger.info(f"Loading finetuned TrOCR from: {model_dir} on {self.device}")
            self.processor = TrOCRProcessor.from_pretrained(model_dir)
            base_model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_NAME)
            
            # Check if it was saved using PEFT/LoRA memory efficient training
            if (Path(model_dir) / "adapter_config.json").exists():
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(base_model, model_dir)
                logger.info("LoRA adapter detected and merged natively.")
            else:
                self.model = VisionEncoderDecoderModel.from_pretrained(model_dir)
        else:
            logger.info(f"Loading Base untrained TrOCR model: {TROCR_MODEL_NAME} on {self.device}")
            self.processor = TrOCRProcessor.from_pretrained(TROCR_PROCESSOR_NAME)
            self.model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_NAME)

        self.model.to(self.device)
        self.model.eval()

        logger.info("TrOCR engine fully active")

    def recognize_line(
        self, line_image: Image.Image | np.ndarray, return_confidence: bool = True
    ) -> dict:
        """
        Recognize text from a single line image.

        Args:
            line_image: Cropped text line image.
            return_confidence: Whether to compute per-token confidence.

        Returns:
            Dict with 'text', 'confidence', 'token_scores'.
        """
        # Convert numpy to PIL if needed — always convert to RGB to handle BGR from OpenCV
        if isinstance(line_image, np.ndarray):
            if len(line_image.shape) == 2:
                # Grayscale: safe to convert directly
                line_image = Image.fromarray(line_image).convert("RGB")
            else:
                # BGR (OpenCV) → RGB before creating PIL image
                import cv2
                line_image = Image.fromarray(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))
        elif line_image.mode != "RGB":
            line_image = line_image.convert("RGB")

        # Process image
        pixel_values = self.processor(
            images=line_image, return_tensors="pt"
        ).pixel_values.to(self.device)

        # Generate with scores for confidence — produce all beams
        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values=pixel_values,
                max_new_tokens=128,
                num_beams=BEAM_SEARCH_WIDTH,
                num_return_sequences=BEAM_SEARCH_WIDTH,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=return_confidence,
            )

        # Decode all beam candidates
        all_texts = self.processor.batch_decode(
            outputs.sequences, skip_special_tokens=True
        )

        # CLIP reranking: pick the candidate that best matches the image semantically
        text = all_texts[0]  # fallback: best beam by log-prob
        clip_model, clip_proc = _get_clip()
        if clip_model is not None and len(all_texts) > 1:
            try:
                with torch.no_grad():
                    clip_inputs = clip_proc(
                        text=all_texts,
                        images=[line_image] * len(all_texts),
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=77,
                    ).to(clip_model.device)
                    sims = clip_model(**clip_inputs).logits_per_image[0]  # (num_beams,)
                    best_idx = sims.argmax().item()
                    text = all_texts[best_idx]
            except Exception:
                pass  # Silent fallback to top beam

        result = {"text": text, "confidence": 1.0, "token_scores": []}

        if return_confidence and outputs.scores:
            # Compute per-token confidence from logits
            token_scores = []
            for score in outputs.scores:
                probs = torch.softmax(score, dim=-1)
                max_prob = probs.max(dim=-1).values[0].item() if probs.dim() > 1 else probs.max(dim=-1).values.item()
                token_scores.append(max_prob)

            result["token_scores"] = token_scores
            result["confidence"] = (
                sum(token_scores) / len(token_scores) if token_scores else 0.0
            )

        return result

    def recognize_lines(
        self, line_images: list[Image.Image | np.ndarray]
    ) -> list[dict]:
        """
        Recognize text from multiple line images.

        Args:
            line_images: List of cropped text line images.

        Returns:
            List of result dicts with 'text', 'confidence', 'token_scores'.
        """
        results = []
        for i, line_img in enumerate(line_images):
            result = self.recognize_line(line_img)
            result["line_index"] = i
            result["is_uncertain"] = result["confidence"] < CONFIDENCE_THRESHOLD
            results.append(result)
        return results

    def recognize_page(
        self,
        line_images: list[np.ndarray],
        join_char: str = "\n",
    ) -> dict:
        """
        Recognize all lines of a page and join into full page text.

        Returns:
            Dict with 'full_text', 'line_results', 'avg_confidence', 'uncertain_lines'.
        """
        line_results = self.recognize_lines(line_images)

        full_text = join_char.join(r["text"] for r in line_results)
        avg_conf = (
            sum(r["confidence"] for r in line_results) / len(line_results)
            if line_results
            else 0.0
        )
        uncertain = [r for r in line_results if r.get("is_uncertain", False)]

        return {
            "full_text": full_text,
            "line_results": line_results,
            "avg_confidence": avg_conf,
            "uncertain_lines": uncertain,
            "num_lines": len(line_results),
        }
