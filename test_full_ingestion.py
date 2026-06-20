import os
from vectorless_engine import vectorless_index

def ingest_all():
    target_dir = "mock_s3"
    print(f"Starting full ingestion of {target_dir}...")
    
    count = 0
    for root, dirs, files in os.walk(target_dir):
        for f in files:
            if f.startswith('.'): continue
            
            file_path = os.path.join(root, f)
            print(f"\n[{count+1}] Ingesting: {file_path}")
            vectorless_index.add_document(file_path)
            count += 1
            
    print(f"\nSuccessfully processed {count} files.")

if __name__ == "__main__":
    ingest_all()
