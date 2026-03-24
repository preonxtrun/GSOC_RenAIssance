"""
TrOCR fine-tuning script for historical handwritten text.
Supports both full fine-tuning and LoRA (memory-efficient) training.
"""
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    TROCR_MODEL_NAME,
    TROCR_PROCESSOR_NAME,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    WARMUP_STEPS,
    USE_LORA,
    LORA_RANK,
    LORA_ALPHA,
    MODELS_DIR,
    GRADIENT_ACCUMULATION_STEPS,
    USE_FP16,
)
from transformers import get_cosine_schedule_with_warmup
from utils.helpers import get_device, setup_logger
from training.dataset import HTRDataset
from training.data_alignment import align_transcriptions, get_train_val_test_split
from evaluation.metrics import compute_cer

logger = setup_logger("training")


def setup_model(model_name: str = TROCR_MODEL_NAME, use_lora: bool = USE_LORA):
    """
    Load TrOCR model and optionally apply LoRA for memory-efficient fine-tuning.
    """
    from transformers import VisionEncoderDecoderModel, TrOCRProcessor, VisionEncoderDecoderConfig

    logger.info(f"Loading model: {model_name}")
    
    # Fix for PEFT save_pretrained crashing on composite VisionEncoderDecoder models.
    # PEFT tries to both GET and SET vocab_size on the config object.
    # We expose the decoder's vocab_size via a property with a no-op setter.
    if not hasattr(VisionEncoderDecoderConfig, "vocab_size"):
        def _vocab_size_getter(self):
            return self.decoder.vocab_size
        def _vocab_size_setter(self, value):
            pass  # intentional no-op: value lives in decoder config
        VisionEncoderDecoderConfig.vocab_size = property(_vocab_size_getter, _vocab_size_setter)

    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    processor = TrOCRProcessor.from_pretrained(TROCR_PROCESSOR_NAME)

    # Set special tokens
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    if use_lora:
        logger.info(f"Applying LoRA (rank={LORA_RANK}, alpha={LORA_ALPHA})")
        try:
            from peft import LoraConfig, get_peft_model, TaskType

            lora_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                r=LORA_RANK,
                lora_alpha=LORA_ALPHA,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj"],  # Attention layers
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        except ImportError:
            logger.warning("peft not installed, falling back to full fine-tuning")

    return model, processor


def train(
    max_pages_per_doc: int = 3,
    dpi: int = 200,
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    save_dir: Path = MODELS_DIR,
):
    """
    Full training loop for TrOCR fine-tuning.

    Args:
        max_pages_per_doc: Limit pages per document (for quick testing).
        dpi: DPI for PDF conversion.
        num_epochs: Number of training epochs.
        batch_size: Batch size for training.
        learning_rate: Learning rate.
        save_dir: Directory to save fine-tuned model.
    """
    device = get_device()

    # Step 1: Align data
    logger.info("Step 1: Aligning transcriptions with page images...")
    aligned_data = align_transcriptions(
        max_pages_per_doc=max_pages_per_doc, dpi=dpi
    )

    train_data, val_data, _ = get_train_val_test_split(aligned_data)
    logger.info(f"Train: {len(train_data)} pages, Val: {len(val_data)} pages")

    if not train_data:
        logger.error("No training data found!")
        return

    # Step 2: Create datasets
    logger.info("Step 2: Creating datasets...")
    train_dataset = HTRDataset(train_data, is_train=True)
    val_dataset = HTRDataset(val_data, is_train=False) if val_data else None

    logger.info(f"Train samples (lines): {len(train_dataset)}")
    if val_dataset:
        logger.info(f"Val samples (lines): {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = (
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        if val_dataset
        else None
    )

    # Step 3: Setup model
    logger.info("Step 3: Setting up model...")
    model, processor = setup_model()
    model.to(device)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)
    
    total_steps = max(1, num_epochs * len(train_loader) // GRADIENT_ACCUMULATION_STEPS)
    warmup_ratio_steps = int(0.1 * total_steps)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_ratio_steps, 
        num_training_steps=total_steps
    )
    
    scaler = torch.amp.GradScaler('cuda', enabled=USE_FP16 and device.type == "cuda")

    # Step 4: Training loop
    logger.info("Step 4: Starting training...")
    best_val_cer = float("inf")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast('cuda', enabled=USE_FP16 and device.type == "cuda"):
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        logger.info(f"Epoch {epoch + 1}/{num_epochs} — Loss: {avg_loss:.4f}")

        # Validation
        if val_loader:
            model.eval()
            all_preds = []
            all_refs = []

            with torch.no_grad():
                for batch in val_loader:
                    pixel_values = batch["pixel_values"].to(device)
                    labels = batch["labels"].clone()
                    labels[labels == -100] = processor.tokenizer.pad_token_id

                    generated = model.generate(pixel_values=pixel_values, max_new_tokens=128)
                    preds = processor.batch_decode(generated, skip_special_tokens=True)
                    refs = processor.tokenizer.batch_decode(
                        labels, skip_special_tokens=True
                    )

                    all_preds.extend(preds)
                    all_refs.extend(refs)

            val_cer = compute_cer(all_preds, all_refs)
            logger.info(f"  Val CER: {val_cer:.4f}")

            # Save best model
            if val_cer < best_val_cer:
                best_val_cer = val_cer
                save_path = save_dir / "trocr-finetuned-best"
                model.save_pretrained(str(save_path))
                processor.save_pretrained(str(save_path))
                logger.info(f"  Saved best model (CER={val_cer:.4f}) -> {save_path}")

    # Save final model
    final_path = save_dir / "trocr-finetuned-final"
    model.save_pretrained(str(final_path))
    processor.save_pretrained(str(final_path))
    logger.info(f"Training complete. Final model -> {final_path}")

    return model, processor


if __name__ == "__main__":
    train(max_pages_per_doc=2, num_epochs=3)  # Quick test run
