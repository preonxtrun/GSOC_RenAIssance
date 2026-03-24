"""
PDF to Image conversion.
Converts multi-page scanned PDFs into individual PIL images at high DPI.
"""
import sys
from pathlib import Path
from typing import Optional

from PIL import Image
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PDF_DPI, IMAGES_DIR


def pdf_to_images(
    pdf_path: str | Path,
    output_dir: Optional[str | Path] = None,
    dpi: int = PDF_DPI,
    save: bool = True,
) -> list[Image.Image]:
    """
    Convert a PDF file to a list of PIL Images.

    Args:
        pdf_path: Path to the PDF file.
        output_dir: Directory to save images. Defaults to config IMAGES_DIR.
        dpi: Resolution for conversion. Higher = better OCR but more memory.
        save: Whether to save images to disk.

    Returns:
        List of PIL Image objects, one per page.
    """
    import fitz  # PyMuPDF

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    output_dir = Path(output_dir) if output_dir else IMAGES_DIR / pdf_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting {pdf_path.name} at {dpi} DPI...")

    images = []
    
    # Open PDF
    doc = fitz.open(str(pdf_path))
    
    # Calculate zoom factor for target DPI (default fitz DPI is 72)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    for i in tqdm(range(len(doc)), desc="Converting pages"):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        # Convert fitz pixmap to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # CRITICAL RAM FIX: Cap the max dimension to 2000px to avoid 10GB+ System RAM consumption
        # Historical PDFs often have weird built-in dimensions that blow up when multiplied by DPI scaling
        max_dim = 2000
        if img.width > max_dim or img.height > max_dim:
            ratio = min(max_dim / img.width, max_dim / img.height)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            
        images.append(img)
        
        if save:
            img_path = output_dir / f"page_{i + 1:04d}.png"
            img.save(str(img_path), "PNG")

    doc.close()

    if save:
        for i, img in enumerate(tqdm(images, desc="Saving pages")):
            img_path = output_dir / f"page_{i + 1:04d}.png"
            img.save(str(img_path), "PNG")

        print(f"  -> {len(images)} pages extracted to {output_dir}")
    return images


def batch_convert_pdfs(
    pdf_dir: str | Path,
    output_dir: Optional[str | Path] = None,
    dpi: int = PDF_DPI,
) -> dict[str, list[Image.Image]]:
    """
    Convert all PDFs in a directory to images.

    Returns:
        Dict mapping PDF stem name → list of page images.
    """
    pdf_dir = Path(pdf_dir)
    results = {}

    pdf_files = list(pdf_dir.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files in {pdf_dir}")

    for pdf_path in pdf_files:
        out = Path(output_dir) / pdf_path.stem if output_dir else None
        images = pdf_to_images(pdf_path, output_dir=out, dpi=dpi)
        results[pdf_path.stem] = images

    return results


if __name__ == "__main__":
    # Quick test: convert first handwriting PDF
    from config import HANDWRITING_PDFS_DIR, HANDWRITING_PAIRS

    pair = HANDWRITING_PAIRS[0]
    pdf_path = HANDWRITING_PDFS_DIR / pair["pdf"]
    images = pdf_to_images(pdf_path, dpi=200)  # Lower DPI for quick test
    print(f"Test: got {len(images)} images from {pair['name']}")
