"""
End-to-end inference pipeline for historical handwritten text recognition.
Combines preprocessing → OCR → VLM → LLM correction.
"""
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CONFIDENCE_THRESHOLD
from utils.helpers import setup_logger

logger = setup_logger("pipeline")


class RenaissanceHTRPipeline:
    """
    End-to-end HTR pipeline with three operating modes:

    1. MINIMAL: Preprocessing → TrOCR only (fastest, baseline)
    2. STANDARD: Preprocessing → TrOCR → LLM correction (good balance)
    3. FULL: Preprocessing → TrOCR + VLM dual read → LLM fusion (best quality)
    """

    def __init__(
        self,
        mode: str = "standard",
        load_models: bool = False,
    ):
        """
        Args:
            mode: 'minimal', 'standard', or 'full'.
            load_models: Whether to load models immediately or lazily.
        """
        self.mode = mode.lower()
        self.ocr_engine = None
        self.vlm_reader = None
        self.llm_corrector = None

        if load_models:
            self._load_models()

    def _load_models(self):
        """Load models based on pipeline mode."""
        from models.ocr_backbone import TrOCREngine
        from models.llm_corrector import LLMCorrector

        logger.info(f"Loading models for '{self.mode}' mode...")

        # If not vlm_centric, load TrOCR
        self.ocr_engine = None
        if self.mode != "vlm_centric":
            self.ocr_engine = TrOCREngine()

        if self.mode in ("standard", "full", "vlm_centric"):
            self.llm_corrector = LLMCorrector()

        if self.mode in ("full", "vlm_centric"):
            from models.vlm_reader import VLMReader
            self.vlm_reader = VLMReader()

        logger.info("Models loaded.")

    def unload_all_models(self):
        """Unload all models to free VRAM."""
        import torch, gc
        if self.ocr_engine and hasattr(self.ocr_engine, 'model'):
            del self.ocr_engine.model
            self.ocr_engine.model = None
            
        if getattr(self, 'vlm_reader', None):
            self.vlm_reader.unload_model()
            
        if getattr(self, 'llm_corrector', None):
            self.llm_corrector.unload_model()
            
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def process_batch(self, page_data_list: list[dict]) -> list[dict]:
        """
        Process a batch of pages by stage to avoid VRAM overload.
        """
        from preprocessing.image_processing import preprocess_image, cv2_to_pil
        from preprocessing.text_extraction import get_line_images
        import torch, gc

        if not self.ocr_engine or (self.mode != "vlm_centric" and getattr(self.ocr_engine, 'model', None) is None):
            self._load_models()

        results = [{"page_num": p.get("page_num", i+1), "source": p.get("source", "")} for i, p in enumerate(page_data_list)]

        logger.info("\n--- PHASE 1: OCR ---")
        from tqdm.auto import tqdm
        
        if self.mode == "vlm_centric":
            print(f"\n  [VLM] Starting execution on {len(page_data_list)} pages...")
            for i, p in enumerate(tqdm(page_data_list, desc="VLM Direct Reading")):
                print(f"    -> Reading Page {i+1}/{len(page_data_list)} (this takes ~25s per page)...")
                
                # VLMs perform better on raw RGB images. Avoid aggressive grayscale/denoising meant for TrOCR.
                raw_img = p["image"]
                pil_image = cv2_to_pil(raw_img) if isinstance(raw_img, np.ndarray) else raw_img
                
                raw_text = self.vlm_reader.transcribe_region(pil_image)
                
                results[i]["transcription"] = raw_text
                results[i]["ocr_raw"] = raw_text
                results[i]["method"] = "vlm_direct"
                results[i]["processed_img"] = raw_img
            print("  [VLM] Finished reading all pages!\n")
            self.vlm_reader.unload_model()
        else:
            for i, p in enumerate(page_data_list):
                processed = preprocess_image(p["image"])
                line_images = get_line_images(processed, preprocess=False)
                ocr_result = self.ocr_engine.recognize_page(line_images)
                
                results[i]["transcription"] = ocr_result["full_text"]
                results[i]["ocr_raw"] = ocr_result["full_text"]
                results[i]["confidence"] = ocr_result["avg_confidence"]
                results[i]["method"] = "trocr_only"
                results[i]["line_results"] = ocr_result["line_results"]
                results[i]["processed_img"] = processed
                
            # CRITICAL: Unload TrOCR to free VRAM
            if hasattr(self.ocr_engine, 'model'):
                del self.ocr_engine.model
                self.ocr_engine.model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if self.mode == "full" and getattr(self, 'vlm_reader', None) is not None:
            logger.info("\n--- PHASE 2: VLM ---")
            for i in range(len(page_data_list)):
                processed = results[i]["processed_img"]
                pil_image = cv2_to_pil(processed) if isinstance(processed, np.ndarray) else processed
                results[i]["vlm_raw"] = self.vlm_reader.transcribe_region(pil_image)
            self.vlm_reader.unload_model()

        if self.mode in ("standard", "full", "vlm_centric") and getattr(self, 'llm_corrector', None) is not None:
            logger.info("\n--- PHASE 3: LLM Correction ---")
            for i in range(len(page_data_list)):
                ocr_text = results[i]["ocr_raw"]
                
                if self.mode == "vlm_centric":
                    corrected = self.llm_corrector.correct_ocr(ocr_text)
                    results[i]["method"] = "vlm_llm_corrected"
                elif self.mode == "full" and "vlm_raw" in results[i]:
                    corrected = self.llm_corrector.fuse_transcriptions(ocr_text, results[i]["vlm_raw"])
                    results[i]["method"] = "trocr_vlm_llm_fusion"
                else:
                    uncertain_words = []
                    for lr in results[i].get("line_results", []):
                        if lr.get("is_uncertain", False):
                            uncertain_words.extend(lr["text"].split()[:3])
                    corrected = self.llm_corrector.correct_ocr(ocr_text, uncertain_words=uncertain_words)
                    results[i]["method"] = "trocr_llm_corrected"
                    
                results[i]["transcription"] = corrected
            self.llm_corrector.unload_model()

        for r in results:
            if "processed_img" in r:
                del r["processed_img"]

        return results

    def process_page(
        self,
        page_image: Image.Image | np.ndarray,
        return_intermediate: bool = False,
    ) -> dict:
        """
        Process a single page image through the full pipeline.

        Args:
            page_image: Full page scan image.
            return_intermediate: Whether to return intermediate results.

        Returns:
            Dict with 'transcription', 'confidence', 'method', and optionally
            'ocr_raw', 'vlm_raw', 'line_results'.
        """
        from preprocessing.image_processing import preprocess_image, cv2_to_pil
        from preprocessing.text_extraction import get_line_images

        if self.ocr_engine is None:
            self._load_models()

        # Step 1: Preprocess
        logger.info("Step 1: Preprocessing...")
        processed = preprocess_image(page_image)

        # Step 2: Extract text lines
        logger.info("Step 2: Extracting text lines...")
        line_images = get_line_images(processed, preprocess=False)
        logger.info(f"  Found {len(line_images)} text lines")

        # Step 3: OCR with TrOCR
        logger.info("Step 3: Running TrOCR OCR...")
        ocr_result = self.ocr_engine.recognize_page(line_images)
        ocr_text = ocr_result["full_text"]

        result = {
            "transcription": ocr_text,
            "confidence": ocr_result["avg_confidence"],
            "method": "trocr_only",
            "num_lines": ocr_result["num_lines"],
        }

        if return_intermediate:
            result["ocr_raw"] = ocr_text
            result["line_results"] = ocr_result["line_results"]

        # Step 4: VLM direct reading (FULL mode only)
        vlm_text = None
        if self.mode == "full" and self.vlm_reader is not None:
            logger.info("Step 4: VLM direct reading...")
            pil_image = cv2_to_pil(processed) if isinstance(processed, np.ndarray) else processed
            vlm_text = self.vlm_reader.transcribe_region(pil_image)
            if return_intermediate:
                result["vlm_raw"] = vlm_text

        # Step 5: LLM correction / fusion
        if self.mode in ("standard", "full") and self.llm_corrector is not None:
            logger.info("Step 5: LLM correction...")

            # Identify uncertain words
            uncertain_words = []
            for lr in ocr_result.get("line_results", []):
                if lr.get("is_uncertain", False):
                    words = lr["text"].split()
                    uncertain_words.extend(words[:3])  # First few words of uncertain lines

            if self.mode == "full" and vlm_text:
                # Fuse OCR + VLM transcriptions
                corrected = self.llm_corrector.fuse_transcriptions(ocr_text, vlm_text)
                result["method"] = "trocr_vlm_llm_fusion"
            else:
                # Correct OCR only
                corrected = self.llm_corrector.correct_ocr(
                    ocr_text, uncertain_words=uncertain_words
                )
                result["method"] = "trocr_llm_corrected"

            result["transcription"] = corrected

        return result

    def process_document(
        self,
        pdf_path: str | Path,
        max_pages: Optional[int] = None,
        dpi: int = 200,
    ) -> list[dict]:
        """
        Process an entire PDF document sequentially by model to avoid OOM crashes.
        
        Args:
            pdf_path: Path to PDF file.
            max_pages: Limit number of pages to process.
            dpi: DPI for PDF conversion.

        Returns:
            List of page results.
        """
        from preprocessing.pdf_to_images import pdf_to_images
        from preprocessing.image_processing import preprocess_image, cv2_to_pil
        from preprocessing.text_extraction import get_line_images

        logger.info(f"Processing document: {pdf_path}")
        images = pdf_to_images(pdf_path, dpi=dpi, save=False)
        if max_pages:
            images = images[:max_pages]

        results = [{"page_num": i + 1} for i in range(len(images))]
        
        # ─── PHASE 1: Preprocessing & OCR ───
        logger.info("\n--- PHASE 1: OCR ---")
        self._load_models()  # Ensures TrOCR is ready
        
        for i, img in enumerate(images):
            processed = preprocess_image(img)
            line_images = get_line_images(processed, preprocess=False)
            ocr_result = self.ocr_engine.recognize_page(line_images)
            
            results[i]["transcription"] = ocr_result["full_text"]
            results[i]["ocr_raw"] = ocr_result["full_text"]
            results[i]["confidence"] = ocr_result["avg_confidence"]
            results[i]["method"] = "trocr_only"
            results[i]["line_results"] = ocr_result["line_results"]
            results[i]["processed_img"] = processed  # Temporarily store for VLM
            
        # Free TrOCR memory (optional but safe)
        import torch, gc
        if hasattr(self.ocr_engine, 'model'):
            del self.ocr_engine.model
            self.ocr_engine.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ─── PHASE 2: VLM ───
        if self.mode == "full" and self.vlm_reader is not None:
            logger.info("\n--- PHASE 2: VLM ---")
            for i in range(len(images)):
                raw_img = images[i]
                pil_image = cv2_to_pil(raw_img) if isinstance(raw_img, np.ndarray) else raw_img
                results[i]["vlm_raw"] = self.vlm_reader.transcribe_region(pil_image)
            
            # CRITICAL: Unload VLM to free 6GB+ memory before loading LLM
            self.vlm_reader.unload_model()

        # ─── PHASE 3: LLM Fusion/Correction ───
        if self.mode in ("standard", "full") and self.llm_corrector is not None:
            logger.info("\n--- PHASE 3: LLM Correction ---")
            for i in range(len(images)):
                ocr_text = results[i]["ocr_raw"]
                
                uncertain_words = []
                for lr in results[i].get("line_results", []):
                    if lr.get("is_uncertain", False):
                        uncertain_words.extend(lr["text"].split()[:3])
                
                if self.mode == "full" and "vlm_raw" in results[i]:
                    corrected = self.llm_corrector.fuse_transcriptions(
                        ocr_text, results[i]["vlm_raw"]
                    )
                    results[i]["method"] = "trocr_vlm_llm_fusion"
                else:
                    corrected = self.llm_corrector.correct_ocr(
                        ocr_text, uncertain_words=uncertain_words
                    )
                    results[i]["method"] = "trocr_llm_corrected"
                    
                results[i]["transcription"] = corrected
                
            # CRITICAL: Unload LLM
            self.llm_corrector.unload_model()

        # Cleanup temp images
        for r in results:
            if "processed_img" in r:
                del r["processed_img"]

        logger.info(f"\nDocument complete: {len(results)} pages processed")
        return results

    def get_full_transcription(self, page_results: list[dict]) -> str:
        """Combine page results into a single document transcription."""
        pages = []
        for r in page_results:
            header = f"--- Page {r.get('page_num', '?')} ---"
            pages.append(f"{header}\n{r['transcription']}")
        return "\n\n".join(pages)
