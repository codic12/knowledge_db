import os
import cv2
import numpy as np
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import re
import csv
import io

def preprocess_image(image_pil):
    open_cv_image = np.array(image_pil)
    if len(open_cv_image.shape) == 3 and open_cv_image.shape[2] == 3:
        img = open_cv_image[:, :, ::-1].copy()
    else:
        img = open_cv_image.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    coords = np.column_stack(np.where(gray > 0))
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        if abs(angle) > 0.5 and abs(angle) < 45:
            (h, w) = gray.shape[:2]
            center = (w//2, h//2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return Image.fromarray(gray)

def parse_csv_to_structured_text(csv_text):
    f = io.StringIO(csv_text.strip())
    reader = csv.reader(f)
    try:
        rows = list(reader)
    except Exception as e:
        print(f"CSV Parsing Error: {e}")
        return csv_text
        
    if not rows:
        return ""
        
    headers = [h.strip() for h in rows[0]]
    structured_lines = []
    for r_idx, row in enumerate(rows[1:]):
        row_parts = []
        for c_idx, val in enumerate(row):
            header = headers[c_idx] if c_idx < len(headers) else f"Col{c_idx+1}"
            row_parts.append(f"{header}: {val.strip()}")
        structured_lines.append(f"[Table Row] " + " | ".join(row_parts))
    return "\n".join(structured_lines)

def parse_text_tables(text):
    """
    Detects plain-text/OCR tables and formats them row-by-row, prepending column headers to prevent context loss.
    """
    if not text:
        return ""
        
    lines = text.splitlines()
    processed_lines = []
    in_table = False
    table_lines = []
    
    def flush_table(t_lines):
        if not t_lines:
            return []
            
        parsed_rows = []
        for line in t_lines:
            stripped = line.strip()
            # Split by '|' if present, else '\t', else 3 or more spaces
            if '|' in stripped:
                parts = [p.strip() for p in stripped.split('|') if p.strip()]
            elif '\t' in stripped:
                parts = [p.strip() for p in stripped.split('\t') if p.strip()]
            else:
                parts = [p.strip() for p in re.split(r'\s{3,}', stripped) if p.strip()]
            if parts:
                parsed_rows.append(parts)
        
        if not parsed_rows or len(parsed_rows) < 2:
            return t_lines
            
        col_counts = [len(r) for r in parsed_rows]
        # Check column count consistency: must be >= 2 and most common column count is at least 60% frequency
        from collections import Counter
        most_common_col_count, count_freq = Counter(col_counts).most_common(1)[0]
        
        if most_common_col_count < 2 or (count_freq / len(parsed_rows)) < 0.6:
            return t_lines
            
        headers = parsed_rows[0]
        formatted_lines = []
        
        # Keep the header representation
        formatted_lines.append(f"[Table Header] " + " | ".join(headers))
        
        for r_idx, row in enumerate(parsed_rows[1:]):
            row_str_parts = []
            for c_idx in range(max(len(headers), len(row))):
                header = headers[c_idx] if c_idx < len(headers) else f"Col{c_idx+1}"
                val = row[c_idx] if c_idx < len(row) else ""
                row_str_parts.append(f"{header}: {val}")
            formatted_lines.append(f"[Table Row] " + " | ".join(row_str_parts))
        return formatted_lines

    for line in lines:
        stripped = line.strip()
        is_table_line = False
        if '|' in stripped:
            is_table_line = True
        elif '\t' in stripped:
            is_table_line = True
        else:
            parts = [p.strip() for p in re.split(r'\s{3,}', stripped) if p.strip()]
            if len(parts) >= 2:
                is_table_line = True
        
        if is_table_line:
            if not in_table:
                in_table = True
            table_lines.append(line)
        else:
            if in_table:
                processed_lines.extend(flush_table(table_lines))
                table_lines = []
                in_table = False
            processed_lines.append(line)
            
    if in_table:
        processed_lines.extend(flush_table(table_lines))
        
    return "\n".join(processed_lines)

def clean_ocr_text(text):
    """
    Advanced cleaning to preserve structural context.
    Propagates parent section numbers to sub-items.
    Also formats embedded tables so headers match individual row cells.
    """
    if not text:
        return ""
        
    # Detect and structure plain text tables first
    text = parse_text_tables(text)
        
    lines = text.splitlines()
    cleaned_lines = []
    current_section = None
    
    for line in lines:
        original_line = line.strip()
        if not original_line:
            continue
            
        # If it is an optimized table row, bypass section prefixing
        if original_line.startswith("[Table Row]") or original_line.startswith("[Table Header]"):
            cleaned_lines.append(original_line)
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
                    if ext == '.csv':
                        text = parse_csv_to_structured_text(text)
                    else:
                        text = clean_ocr_text(text)
                    pages_data.append({'page': 1, 'text': text})
        except Exception as e:
            print(f"Error processing text file {file_path}: {e}")

    return pages_data
