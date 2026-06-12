import os
import cv2
import numpy as np
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import re

def preprocess_image(image_pil):
    """
    Convert PIL image to OpenCV format, grayscale, threshold, and deskew.
    """
    # Convert PIL to OpenCV format
    open_cv_image = np.array(image_pil)
    # Convert RGB to BGR
    if len(open_cv_image.shape) == 3 and open_cv_image.shape[2] == 3:
        img = open_cv_image[:, :, ::-1].copy()
    else:
        img = open_cv_image.copy()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # Simple deskew
    coords = np.column_stack(np.where(gray > 0))
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        if abs(angle) > 0.5 and abs(angle) < 45:
            (h, w) = gray.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return Image.fromarray(gray)

def clean_ocr_text(text):
    """
    Advanced cleaning to preserve structural context.
    Propagates parent section numbers to sub-items.
    Example: 
    6. Insurance
       A. Liability
    becomes:
    6. Insurance
    [Section 6] A. Liability
    """
    if not text:
        return ""
        
    lines = text.splitlines()
    cleaned_lines = []
    current_section = None
    
    for line in lines:
        original_line = line.strip()
        if not original_line:
            continue
            
        # Detect primary section (e.g., "6.", "Section 6", "6.0")
        section_match = re.match(r'^(\d+[\.\)]|Section\s+\d+)', original_line, re.IGNORECASE)
        if section_match:
            current_section = section_match.group(1).strip()
            # Clean up the section string for propagation
            current_section = re.sub(r'[\.\)]', '', current_section)
        
        # Detect sub-item (e.g., "A.", "a)", "(b)")
        sub_item_match = re.match(r'^([A-Za-z][\.\)])', original_line)
        if sub_item_match and current_section:
            # Prepend context
            cleaned_lines.append(f"[Section {current_section}] {original_line}")
        else:
            cleaned_lines.append(original_line)

    # Join lines with spaces but keep paragraph-like separation for chunking
    text = "\n".join(cleaned_lines)
    
    # Standard cleanup
    text = re.sub(r' +', ' ', text)
    
    return text.strip()

def extract_text_from_image(image_pil):
    """
    Extracts text from a PIL Image using Tesseract OCR.
    Handles orientation detection and advanced cleaning.
    """
    try:
        osd = pytesseract.image_to_osd(image_pil)
        angle = int(re.search(r'(?<=Rotate: )\d+', osd).group(0))
        if angle != 0:
            image_pil = image_pil.rotate(-angle, expand=True)
    except Exception:
        pass

    processed_img = preprocess_image(image_pil)
    custom_config = r'--oem 3 --psm 3'
    text = pytesseract.image_to_string(processed_img, config=custom_config)
    return clean_ocr_text(text)

def process_file(file_path):
    """
    Processes a file (PDF or Image), returning a list of dicts.
    """
    ext = os.path.splitext(file_path)[1].lower()
    pages_data = []

    if ext in ['.pdf']:
        try:
            images = convert_from_path(file_path)
            for i, img in enumerate(images):
                text = extract_text_from_image(img)
                if text:
                    pages_data.append({'page': i + 1, 'text': text})
        except Exception as e:
            print(f"Error processing PDF {file_path}: {e}")
            
    elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
        try:
            img = Image.open(file_path)
            text = extract_text_from_image(img)
            if text:
                pages_data.append({'page': 1, 'text': text})
        except Exception as e:
            print(f"Error processing Image {file_path}: {e}")
            
    elif ext in ['.txt', '.csv', '.md']:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                if text.strip():
                    pages_data.append({'page': 1, 'text': text.strip()})
        except Exception as e:
            print(f"Error processing text file {file_path}: {e}")

    return pages_data
