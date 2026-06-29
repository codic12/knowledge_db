import os
import sys
from vectorless_engine import vectorless_index
from llm_client import generate_answer

def run_test():
    doc1 = "mock_s3/companies_table.csv"
    doc2 = "mock_s3/financials_table.csv"

    print("\n--- INGESTION PHASE (Relational Tables) ---")
    for f in [doc1, doc2]:
        if os.path.exists(f):
            print(f"Indexing {f} (force=True)...")
            vectorless_index.add_document(f, force=True)
        else:
            print(f"Error: {f} not found.")
            sys.exit(1)

    # Let's inspect what's inside the pages database to check our table formatting
    print("\n--- DATABASE INSPECTION ---")
    vectorless_index.cursor.execute("SELECT filepath, text FROM pages JOIN documents ON pages.doc_id = documents.id WHERE filepath IN (?, ?)", (doc1, doc2))
    for filepath, text in vectorless_index.cursor.fetchall():
        print(f"\nDocument: {os.path.basename(filepath)}")
        print(text)

    # 3. Perform a relational multi-document query
    query = "What is the revenue and headcount of Acme Space Logistics?"
    
    print(f"\n--- RETRIEVAL PHASE (Query: {query}) ---")
    top_contexts = vectorless_index.search(query, top_n=5)
    
    if not top_contexts:
        print("No contexts retrieved.")
        return

    print(f"\nRetrieved {len(top_contexts)} relevant pages/sections:")
    for i, ctx in enumerate(top_contexts):
        print(f"[{i+1}] {os.path.basename(ctx['filepath'])} (Page {ctx['page']})")
        print(f"Content:\n{ctx['text']}\n")

    # 4. Generate final answer
    print("\n--- GENERATION PHASE ---")
    answer = generate_answer(query, top_contexts)
    print("\nFINAL ANSWER:")
    print(answer)

if __name__ == "__main__":
    from llm_client import LLM_PROVIDER, config
    
    api_key_env = config["api_key_env"]
    api_key = os.environ.get(api_key_env)
    
    if LLM_PROVIDER == "gemini" and not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        print(f"Error: {api_key_env} is not set for provider '{LLM_PROVIDER}'.")
        sys.exit(1)
        
    run_test()
