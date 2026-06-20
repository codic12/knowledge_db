import os
import sqlite3
import numpy as np
import io
from ocr_engine import process_file
from llm_client import safe_chat_completion, generate_embeddings, MODEL_NAME

DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "vectorless_index.db")

def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

# Register the adapter/converter for numpy arrays
sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)

class LocalVectorlessIndex:
    """
    A RAG implementation that uses:
    1. Vector search over document chunks in SQLite to filter relevant documents (Stage 1).
    2. LLM reasoning over page summaries of filtered documents to select pages (Stage 2).
    """
    def __init__(self):
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        self.conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._create_tables()

    def _create_tables(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filepath TEXT UNIQUE,
                mtime REAL,
                summary TEXT
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS pages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id INTEGER,
                page_num INTEGER,
                summary TEXT,
                text TEXT,
                FOREIGN KEY(doc_id) REFERENCES documents(id) ON DELETE CASCADE
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                page_id INTEGER,
                text TEXT,
                embedding array,
                FOREIGN KEY(page_id) REFERENCES pages(id) ON DELETE CASCADE
            )
        ''')
        # Indexes for faster lookup
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_pages_doc_id ON pages(doc_id)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunks_page_id ON chunks(page_id)')
        self.conn.commit()

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
        """Simple character-based chunking with overlap."""
        if not text:
            return []
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    def add_document(self, file_path: str, force: bool = False):
        """
        Ingests a document: Extracts text, generates page summaries, 
        and computes chunk embeddings for vector search.
        Uses SQLite for storage.
        """
        if not os.path.exists(file_path):
            print(f"Vectorless Ingestion: File not found {file_path}")
            return

        mtime = os.path.getmtime(file_path)
        
        self.cursor.execute('SELECT id, mtime, summary FROM documents WHERE filepath = ?', (file_path,))
        doc_row = self.cursor.fetchone()
        
        if not force and doc_row and doc_row[1] == mtime:
            doc_id = doc_row[0]
            # Check if all chunks have embeddings
            self.cursor.execute('''
                SELECT COUNT(*) FROM chunks 
                JOIN pages ON chunks.page_id = pages.id 
                WHERE pages.doc_id = ?
            ''', (doc_id,))
            chunk_count = self.cursor.fetchone()[0]
            if chunk_count > 0:
                print(f"Vectorless Ingestion: Skipping {file_path} (already indexed).")
                return

        # If file changed or force or missing chunks, we might need to re-process.
        # But wait, the user wants us to be efficient.
        # If the document entry exists and mtime is the same, we can reuse pages and summaries.
        can_reuse_pages = not force and doc_row and doc_row[1] == mtime
        
        if can_reuse_pages:
            print(f"Vectorless Ingestion: Updating embeddings for {file_path}...")
            doc_id = doc_row[0]
            # We'll just delete existing chunks and regenerate
            self.cursor.execute('DELETE FROM chunks WHERE page_id IN (SELECT id FROM pages WHERE doc_id = ?)', (doc_id,))
        else:
            print(f"Vectorless Ingestion: Full processing for {file_path}...")
            if doc_row:
                self.cursor.execute('DELETE FROM documents WHERE id = ?', (doc_row[0],))
            
            self.cursor.execute('INSERT INTO documents (filepath, mtime) VALUES (?, ?)', (file_path, mtime))
            doc_id = self.cursor.lastrowid
            
            pages_data = process_file(file_path)
            temp_pages = {} # To use for document summary
            
            for p in pages_data:
                page_num = p['page']
                text = p['text']
                summary = self._generate_page_summary(text)
                self.cursor.execute('''
                    INSERT INTO pages (doc_id, page_num, summary, text)
                    VALUES (?, ?, ?, ?)
                ''', (doc_id, page_num, summary, text))
                temp_pages[str(page_num)] = {"summary": summary}
                print(f"  - Summarized Page {page_num}")
            
            doc_summary = self._generate_document_summary(temp_pages)
            self.cursor.execute('UPDATE documents SET summary = ? WHERE id = ?', (doc_summary, doc_id))

        # Now handle chunks and embeddings
        self.cursor.execute('SELECT id, text FROM pages WHERE doc_id = ?', (doc_id,))
        page_rows = self.cursor.fetchall()
        
        all_chunks_to_embed = []
        chunk_metadata = [] # (page_id, chunk_text)
        
        for page_id, page_text in page_rows:
            chunks = self._chunk_text(page_text)
            for chunk_text in chunks:
                all_chunks_to_embed.append(chunk_text)
                chunk_metadata.append((page_id, chunk_text))

        if all_chunks_to_embed:
            print(f"  - Generating local embeddings for {len(all_chunks_to_embed)} chunks...")
            embeddings = generate_embeddings(all_chunks_to_embed)
            for (page_id, chunk_text), emb in zip(chunk_metadata, embeddings):
                self.cursor.execute('''
                    INSERT INTO chunks (page_id, text, embedding)
                    VALUES (?, ?, ?)
                ''', (page_id, chunk_text, np.array(emb)))

        self.conn.commit()
        print(f"Successfully indexed {file_path}.")

    def _generate_document_summary(self, pages: dict) -> str:
        """Uses the LLM to compile page summaries into an overall document summary."""
        page_summaries = []
        for page_num in sorted(pages.keys(), key=lambda k: int(k)):
            page_summaries.append(f"Page {page_num}: {pages[page_num]['summary']}")
        combined = "\n".join(page_summaries)
        system_message = "You are a document analyzer."
        user_prompt = (
            "The following are individual page summaries from a single document. "
            "Compile them into ONE concise paragraph that captures the overall purpose, "
            "key topics, and essential details of the entire document.\n\n"
            f"PAGE SUMMARIES:\n{combined[:4000]}\n\nOVERALL DOCUMENT SUMMARY:"
        )
        try:
            response_text = safe_chat_completion(
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=300
            )
            return response_text
        except Exception as e:
            print(f"Document Summary Error: {e}")
            return "Summary unavailable."

    def _generate_page_summary(self, text: str) -> str:
        """Uses the LLM to create a 1-sentence summary of a page."""
        system_message = "You are a document analyzer."
        user_prompt = (
            "Summarize the following document page text in EXACTLY one concise sentence. "
            "Focus on the main topic, section headings, and key entities mentioned.\n\n"
            f"TEXT:\n{text[:2000]}\n\nSUMMARY:"
        )
        try:
            response_text = safe_chat_completion(
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=150
            )
            return response_text
        except Exception as e:
            print(f"Summary Error: {e}")
            return "Summary unavailable."

    def search(self, query: str, top_n: int = 5, top_k_chunks: int = 40, doc_threshold: int = 5):
        """
        Two-stage retrieval using SQLite:
        Stage 1: Vector search over all chunks in SQLite, aggregate to DocScore to filter documents.
        Stage 2: Show page summaries ONLY for the filtered documents to select specific pages.
        """
        self.cursor.execute('SELECT COUNT(*) FROM documents')
        if self.cursor.fetchone()[0] == 0:
            print("VECTORLESS RAG: Database is empty.")
            return []

        # --- Stage 1: Vector Search (Document Filtering) ---
        print(f"VECTORLESS RAG: Stage 1 - Vector Search for '{query}'")
        query_emb = generate_embeddings([query], is_query=True)[0]
        query_vec = np.array(query_emb)

        # Retrieve ALL chunks and embeddings for scoring
        # For very large datasets, we'd use a vector extension or FAISS, 
        # but for this scale, SQLite + numpy is fine.
        self.cursor.execute('''
            SELECT documents.filepath, chunks.embedding 
            FROM chunks
            JOIN pages ON chunks.page_id = pages.id
            JOIN documents ON pages.doc_id = documents.id
        ''')
        rows = self.cursor.fetchall()
        
        chunk_scores = []
        for filepath, emb in rows:
            chunk_vec = np.array(emb)
            score = np.dot(query_vec, chunk_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(chunk_vec))
            chunk_scores.append((filepath, score))

        # Sort by score and take top K
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        top_chunks = chunk_scores[:top_k_chunks]

        # Aggregate to DocScore: 1/sqrt(N+1) * sum(ChunkScores)
        doc_stats = {} # {file_path: [scores]}
        for file_path, score in top_chunks:
            if file_path not in doc_stats:
                doc_stats[file_path] = []
            doc_stats[file_path].append(score)

        doc_scores = []
        for file_path, scores in doc_stats.items():
            N = len(scores)
            doc_score = (1.0 / np.sqrt(N + 1)) * sum(scores)
            doc_scores.append((file_path, doc_score))

        doc_scores.sort(key=lambda x: x[1], reverse=True)
        filtered_doc_paths = [path for path, score in doc_scores[:doc_threshold]]
        
        print(f"VECTORLESS RAG: Stage 1 Results -> Filtered to {len(filtered_doc_paths)} documents:")
        for path, score in doc_scores[:doc_threshold]:
            print(f"  - {os.path.basename(path)} (Score: {score:.4f})")

        if not filtered_doc_paths:
            return []

        # --- Stage 2: Page Selection (LLM Reasoning) ---
        page_summaries_view = ""
        page_map = []
        idx = 1
        
        for file_path in filtered_doc_paths:
            filename = os.path.basename(file_path)
            self.cursor.execute('''
                SELECT pages.page_num, pages.summary, pages.text 
                FROM pages
                JOIN documents ON pages.doc_id = documents.id
                WHERE documents.filepath = ?
                ORDER BY pages.page_num
            ''', (file_path,))
            for p_num, p_summary, p_text in self.cursor.fetchall():
                page_summaries_view += f"[{idx}] Doc: {filename}, Page: {p_num} - Summary: {p_summary}\n"
                page_map.append({
                    "filepath": file_path,
                    "page": p_num,
                    "text": p_text
                })
                idx += 1

        print("\n--- VECTORLESS RAG: STAGE 2 (PAGE SUMMARIES) ---")
        print(page_summaries_view)
        print("----------------------------------------------\n")

        page_selection_prompt = (
            "You are an expert retrieval assistant. We have identified relevant documents using semantic search. "
            "Now, identify the specific pages that contain the answer to the user's question.\n\n"
            "INSTRUCTIONS:\n"
            "1. Return ONLY a comma-separated list of page indices, e.g., '1, 4, 10'.\n"
            "2. Prioritize the most relevant pages. Return at most 5 indices.\n"
            "3. If none are relevant, return 'None'.\n\n"
            f"PAGE SUMMARIES:\n{page_summaries_view}\n"
            f"QUESTION: {query}\n"
            "RELEVANT PAGE INDICES:"
        )

        try:
            page_choice_text = safe_chat_completion(
                messages=[{"role": "user", "content": page_selection_prompt}],
                temperature=0.0,
                max_tokens=100
            )
            print(f"VECTORLESS RAG: Stage 2 Choice -> {page_choice_text}")

            if page_choice_text.lower() == "none":
                return []

            selected_page_indices = []
            for part in page_choice_text.split(','):
                try:
                    val = int(part.strip().strip('[]'))
                    if 1 <= val <= len(page_map):
                        selected_page_indices.append(val - 1)
                except:
                    continue
            
            # Step 3: Fetch full text for the chosen pages
            results = []
            print(f"VECTORLESS RAG: Accessing full text for {len(selected_page_indices[:top_n])} pages.")
            for i in selected_page_indices[:top_n]:
                res = page_map[i]
                print(f"  - Accessing: {os.path.basename(res['filepath'])}, Page {res['page']}")
                results.append(res)
            
            return results

        except Exception as e:
            print(f"Reasoning Error: {e}")
            return []

# Global instance
vectorless_index = LocalVectorlessIndex()
