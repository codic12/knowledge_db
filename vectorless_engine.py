import os
import json
from ocr_engine import process_file
from llm_client import safe_chat_completion, MODEL_NAME

DATA_DIR = "data"
VECTORLESS_INDEX_PATH = os.path.join(DATA_DIR, "vectorless_index.json")

class LocalVectorlessIndex:
    """
    A 'Vectorless' RAG implementation that uses LLM reasoning over page summaries
    instead of vector similarity.
    """
    def __init__(self):
        self.index = {} # {filepath: { "summary": "...", "pages": { page_num: { "summary": "...", "text": "..." } } } }
        self.load_index()

    def load_index(self):
        if os.path.exists(VECTORLESS_INDEX_PATH):
            try:
                with open(VECTORLESS_INDEX_PATH, 'r') as f:
                    self.index = json.load(f)
            except Exception as e:
                print(f"Error loading vectorless index: {e}")
                self.index = {}

    def save_index(self):
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        with open(VECTORLESS_INDEX_PATH, 'w') as f:
            json.dump(self.index, f, indent=2)

    def add_document(self, file_path: str):
        """
        Ingests a document: Extracts text, then uses LLM to generate summaries for each page.
        """
        print(f"Vectorless Ingestion: Processing {file_path}...")
        pages_data = process_file(file_path)
        
        doc_data = {
            "full_path": file_path,
            "pages": {}
        }

        for p in pages_data:
            page_num = p['page']
            text = p['text']
            
            # Generate a very concise summary for this page to use in the 'reasoning' step
            summary = self._generate_page_summary(text)
            doc_data["pages"][str(page_num)] = {
                "summary": summary,
                "text": text
            }
            print(f"  - Summarized Page {page_num}")

        self.index[file_path] = doc_data
        self.save_index()
        print(f"Successfully indexed {file_path} with {len(pages_data)} pages.")

    def _generate_page_summary(self, text: str) -> str:
        """Uses the LLM to create a 1-sentence summary of a page."""
        system_message = "You are a document analyzer."
        user_prompt = (
            "Summarize the following document page text in EXACTLY one concise sentence. "
            "Focus on the main topic, section headings, and key entities mentioned.\n\n"
            f"TEXT:\n{text[:2000]}\n\nSUMMARY:"
        )
        try:
            response = safe_chat_completion(
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=150
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Summary Error: {e}")
            return "Summary unavailable."

    def search(self, query: str, top_n: int = 5):
        """
        Reasoning-based retrieval:
        1. Show all page summaries to the LLM.
        2. Ask the LLM which pages are relevant.
        3. Return the full text of those pages.
        """
        if not self.index:
            print("VECTORLESS RAG: Index is empty.")
            return []

        # Step 1: Prepare the 'Global Index' (summaries of all pages in all docs)
        global_index_view = ""
        page_map = [] # To map the LLM's choice back to our data
        
        idx = 1
        for file_path, doc in self.index.items():
            filename = os.path.basename(file_path)
            for page_num, data in doc["pages"].items():
                global_index_view += f"[{idx}] Doc: {filename}, Page: {page_num} - Summary: {data['summary']}\n"
                page_map.append({
                    "filepath": file_path,
                    "page": int(page_num),
                    "text": data["text"]
                })
                idx += 1

        print("\n--- VECTORLESS RAG: GLOBAL INDEX VIEW ---")
        print(global_index_view)
        print("------------------------------------------\n")

        # Step 2: Ask the LLM to identify relevant page indices
        reasoning_prompt = (
            "You are a retrieval assistant. Given the following index of document pages, "
            "identify the indices [N] of the pages that MOST LIKELY contain the answer to the user's question.\n"
            "Return ONLY a comma-separated list of indices, e.g., '1, 4, 12'. "
            "If none are relevant, return 'None'.\n\n"
            f"INDEX:\n{global_index_view}\n"
            f"QUESTION: {query}\n"
            "RELEVANT INDICES:"
        )

        print("--- VECTORLESS RAG: REASONING PROMPT ---")
        print(reasoning_prompt)
        print("----------------------------------------\n")

        try:
            response = safe_chat_completion(
                messages=[{"role": "user", "content": reasoning_prompt}],
                temperature=0.0,
                max_tokens=50
            )
            choice_text = response.choices[0].message.content.strip()
            print(f"VECTORLESS RAG: LLM Choice -> {choice_text}")
            
            if choice_text.lower() == "none":
                return []

            # Parse indices
            indices = []
            for part in choice_text.split(','):
                try:
                    val = int(part.strip().strip('[]'))
                    if 1 <= val <= len(page_map):
                        indices.append(val - 1)
                except:
                    continue
            
            # Step 3: Fetch full text for the chosen pages
            results = []
            print(f"VECTORLESS RAG: Accessing full text for {len(indices[:top_n])} pages.")
            for i in indices[:top_n]:
                res = page_map[i]
                print(f"  - Accessing: {os.path.basename(res['filepath'])}, Page {res['page']}")
                results.append(res)
            
            return results

        except Exception as e:
            print(f"Reasoning Error: {e}")
            return []

# Global instance
vectorless_index = LocalVectorlessIndex()
