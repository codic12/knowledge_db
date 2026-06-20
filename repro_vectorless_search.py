import os
import json
from vectorless_engine import vectorless_index
from llm_client import LLM_PROVIDER, MODEL_NAME

def repro_empty_indices():
    print(f"Testing Vectorless Search with {LLM_PROVIDER} / {MODEL_NAME}")
    
    # Mock some data if index is empty
    if not vectorless_index.index:
        print("Index is empty, adding mock data...")
        vectorless_index.index = {
            "who_sitrep_1.pdf": {
                "pages": {
                    "1": {"summary": "Situation report for December 2019 regarding initial cases.", "text": "In December 2019, several cases were reported..."},
                    "2": {"summary": "Data tables for early January 2020.", "text": "By January 2020, the numbers grew..."}
                }
            }
        }
    
    query = "what happened in december 2019 according to situation report"
    print(f"Query: {query}")
    
    results = vectorless_index.search(query, top_n=5)
    print(f"\nFinal Results Count: {len(results)}")
    for res in results:
        print(f" - {res['filepath']}, Page {res['page']}")

if __name__ == "__main__":
    repro_empty_indices()
