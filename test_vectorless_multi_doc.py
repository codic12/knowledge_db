import os
import sys
from vectorless_engine import vectorless_index
from llm_client import generate_answer

def run_test():
    # 1. Setup sample files
    doc1 = "mock_s3/sample_contract.txt"
    doc2 = "mock_s3/folder1/SampleContract-Shuttle.pdf" # This might not exist, checking directory structure
    
    # Let's check what we have
    print("Files available in mock_s3:")
    for root, dirs, files in os.walk("mock_s3"):
        for f in files:
            print(f"  {os.path.join(root, f)}")

    target_files = [
        "mock_s3/sample_contract.txt",
        "mock_s3/SampleContract-Shuttle.pdf"
    ]

    # 2. Ingest documents into the custom vectorless index
    print("\n--- INGESTION PHASE ---")
    for f in target_files:
        if os.path.exists(f):
            vectorless_index.add_document(f)
        else:
            print(f"Warning: {f} not found.")

    # 3. Perform a multi-document query
    # We want a question that might be answered by either or both
    query = "What are the insurance requirements and who are the parties involved in these contracts?"
    
    print(f"\n--- RETRIEVAL PHASE (Query: {query}) ---")
    top_contexts = vectorless_index.search(query, top_n=5)
    
    if not top_contexts:
        print("No contexts retrieved.")
        return

    print(f"Retrieved {len(top_contexts)} relevant pages/sections.")
    for i, ctx in enumerate(top_contexts):
        print(f"[{i+1}] {os.path.basename(ctx['filepath'])} (Page {ctx['page']})")

    # 4. Generate final answer
    print("\n--- GENERATION PHASE ---")
    answer = generate_answer(query, top_contexts)
    print("\nFINAL ANSWER:")
    print(answer)

if __name__ == "__main__":
    from llm_client import LLM_PROVIDER, config
    
    # Ensure environment variables are set for the current provider
    api_key_env = config["api_key_env"]
    api_key = os.environ.get(api_key_env)
    
    # Also check GEMINI_API_KEY if provider is gemini
    if LLM_PROVIDER == "gemini" and not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        print(f"Error: {api_key_env} is not set for provider '{LLM_PROVIDER}'.")
        # If gemini, we also allow GEMINI_API_KEY
        if LLM_PROVIDER == "gemini":
             print("Alternatively, set GEMINI_API_KEY.")
        sys.exit(1)
        
    run_test()
