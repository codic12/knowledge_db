from ocr_engine import process_file
import json

file_path = "mock_s3/SampleContract-Shuttle.pdf"
pages = process_file(file_path)

for page in pages:
    print(f"--- Page {page['page']} ---")
    print(page['text'])
    print("\n" + "="*50 + "\n")
