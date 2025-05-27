from pathlib import Path
from pdf2image import convert_from_path

pdf_path = Path('uploads//20250325_Flanary_Sync.pdf')
output_dir = './test_ocr'

# Create output directory if it doesn't exist
import os
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Try to convert the PDF to images
try:
    print(f"Converting {pdf_path}...")
    images = convert_from_path(
        pdf_path, 
        dpi=300,
        output_folder=output_dir,
        fmt='jpg',
        grayscale=True
    )
    print(f"Successfully converted to {len(images)} images")
    print(f"Images saved to {output_dir}")
except Exception as e:
    print(f"Error: {e}")
