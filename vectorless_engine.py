import os
import sqlite3
import numpy as np
import io
import re
import math
from collections import Counter
from ocr_engine import process_file
from llm_client import safe_chat_completion, generate_embeddings, MODEL_NAME

def tokenize(text):
    if not text:
        return []
    # Preserve dot-separated or hyphenated section numbers, codes, and IDs (e.g. 4.18.4.6, ABC-123)
    return re.findall(r'\b\w+(?:[\.-]\w+)*\b', text.lower())

def compute_bm25_scores(pages, query_text, k1=1.5, b=0.75):
    query_terms = tokenize(query_text)
    if not query_terms:
        return {p['id']: 0.0 for p in pages}
    
    # Prepare corpus
    corpus = []
    page_ids = []
    for p in pages:
        corpus.append(tokenize(p['text']))
        page_ids.append(p['id'])
    
    N = len(corpus)
    if N == 0:
        return {}
    
    # Calculate document lengths and average length
    doc_lens = [len(doc) for doc in corpus]
    avg_doc_len = sum(doc_lens) / N if N > 0 else 1.0
    
    # Calculate term frequencies for each document
    dfs = Counter()
    tfs = []
    for doc in corpus:
        doc_tfs = Counter(doc)
        tfs.append(doc_tfs)
        for term in doc_tfs:
            dfs[term] += 1
            
    # Calculate IDF for each query term
    idfs = {}
    for term in query_terms:
        df = dfs[term]
        # BM25 IDF formula
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
        idfs[term] = max(0.0001, idf)
        
    # Calculate scores
    scores = {}
    for idx, p_id in enumerate(page_ids):
        doc_tf = tfs[idx]
        doc_len = doc_lens[idx]
        score = 0.0
        for term in query_terms:
            tf = doc_tf[term]
            idf = idfs[term]
            denom = tf + k1 * (1.0 - b + b * doc_len / avg_doc_len)
            if denom > 0:
                score += idf * (tf * (k1 + 1.0)) / denom
        scores[p_id] = score
        
    return scores

def LevenshteinDistance(s, t):
    if s == t: return 0
    if len(s) == 0: return len(t)
    if len(t) == 0: return len(s)
    
    v0 = [0] * (len(t) + 1)
    v1 = [0] * (len(t) + 1)
    
    for i in range(len(v0)):
        v0[i] = i
        
    for i in range(len(s)):
        v1[0] = i + 1
        for j in range(len(t)):
            cost = 0 if s[i] == t[j] else 1
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        v0 = v1[:]
        
    return v0[len(t)]

def fuzzy_match_score(page_text, query_text):
    query_words = [w for w in tokenize(query_text) if len(w) > 2]
    page_words = set(tokenize(page_text))
    
    if not query_words or not page_words:
        return 0.0
        
    matches = 0
    for q_w in query_words:
        # Check exact match
        if q_w in page_words:
            matches += 1
            continue
        # Check fuzzy match against page words of similar length
        q_len = len(q_w)
        for p_w in page_words:
            if abs(len(p_w) - q_len) <= 2:
                dist = LevenshteinDistance(q_w, p_w)
                max_len = max(len(q_w), len(p_w))
                sim = 1.0 - (dist / max_len)
                if sim >= 0.8:
                    matches += 1
                    break
                    
    return matches / len(query_words)

def normalize_scores(score_dict):
    if not score_dict:
        return {}
    vals = list(score_dict.values())
    min_v = min(vals)
    max_v = max(vals)
    diff = max_v - min_v
    if diff == 0:
        return {k: 1.0 for k in score_dict}
    return {k: (v - min_v) / diff for k, v in score_dict.items()}

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
    return np.load(out, allow_pickle=True)

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
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY,
                title TEXT,
                updated_at REAL,
                messages TEXT
            )
        ''')
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

    def _extract_linking_keys(self, query: str, candidate_pages: list) -> list[str]:
        if not candidate_pages:
            return []
        
        # Compile text snippet of candidates
        snippet = ""
        for idx, p in enumerate(candidate_pages[:3]): # inspect top 3 to keep prompt small and fast
            snippet += f"Candidate {idx+1} ({p['filename']}, Page {p['page_num']}):\n{p['text'][:1500]}\n\n"
            
        prompt = (
            "You are a database and document linking assistant.\n"
            f"The user's query is: \"{query}\"\n"
            "We have retrieved the following candidate pages:\n"
            f"{snippet}\n"
            "Identify if there are any specific entity IDs, foreign keys, or codes (e.g., a Company ID, client code, transaction code, etc.) mentioned in the retrieved pages that might link these entities to details in other documents to answer the query.\n"
            "Rules:\n"
            "1. Extract ONLY the literal values of the keys/IDs (e.g. '101', 'COMP-45').\n"
            "2. Do NOT extract common words, generic names, or the query text itself.\n"
            "3. Return the keys as a comma-separated list. If none, return 'None'.\n"
            "Keys:"
        )
        try:
            response = safe_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=50
            )
            if response.lower().strip() == "none" or not response.strip():
                return []
            keys = [k.strip().strip("'\"[]") for k in response.split(",") if k.strip()]
            return [k for k in keys if k.lower() != "none" and len(k) > 0]
        except Exception as e:
            print(f"Error in _extract_linking_keys: {e}")
            return []

    def _search_pages_by_key(self, key: str, exclude_paths: set) -> list:
        query_param = f"%{key}%"
        self.cursor.execute('''
            SELECT pages.page_num, pages.summary, pages.text, documents.filepath
            FROM pages
            JOIN documents ON pages.doc_id = documents.id
            WHERE pages.text LIKE ?
        ''', (query_param,))
        
        matched_pages = []
        for page_num, summary, text, filepath in self.cursor.fetchall():
            if filepath in exclude_paths:
                continue
            
            pattern = r'\b' + re.escape(key) + r'\b'
            if not key[0].isalnum() or not key[-1].isalnum():
                pattern = re.escape(key)
                
            if re.search(pattern, text, re.IGNORECASE):
                matched_pages.append({
                    "page_num": page_num,
                    "summary": summary,
                    "text": text,
                    "filepath": filepath,
                    "filename": os.path.basename(filepath)
                })
        return matched_pages

    def search(self, query: str, top_n: int = 5, top_k_chunks: int = 40, doc_threshold: int = 5, top_k_hybrid: int = 10):
        """
        Integrated Global Hybrid Search and Relational Query Expansion:
        Stage 1: Retrieve global candidate pages using vector similarity (top chunks) and exact keyword matches.
        Stage 2: Perform global hybrid vector + BM25 + fuzzy search to score and rank candidate pages.
        Stage 3: Scan top candidate pages for linking keys/IDs and perform secondary relational searches.
        Stage 4: Show summaries of all candidate pages to LLM to select the final pages.
        """
        self.cursor.execute('SELECT COUNT(*) FROM documents')
        if self.cursor.fetchone()[0] == 0:
            print("VECTORLESS RAG: Database is empty.")
            return []

        print(f"VECTORLESS RAG: Stage 1 - Overall Hybrid Search for '{query}'")
        
        # 1. Vector Search Candidates
        query_emb = generate_embeddings([query], is_query=True)[0]
        query_vec = np.array(query_emb)

        self.cursor.execute('''
            SELECT chunks.page_id, chunks.embedding, pages.text, pages.summary, pages.page_num, documents.filepath
            FROM chunks
            JOIN pages ON chunks.page_id = pages.id
            JOIN documents ON pages.doc_id = documents.id
        ''')
        all_chunks = self.cursor.fetchall()
        
        chunk_scores = []
        for p_id, emb, p_text, p_summary, p_num, filepath in all_chunks:
            chunk_vec = np.array(emb)
            score = np.dot(query_vec, chunk_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(chunk_vec))
            chunk_scores.append((p_id, score, p_text, p_summary, p_num, filepath))
            
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        top_vector_chunks = chunk_scores[:top_k_chunks]
        
        candidate_pages_map = {}
        for p_id, score, p_text, p_summary, p_num, filepath in top_vector_chunks:
            if p_id not in candidate_pages_map:
                candidate_pages_map[p_id] = {
                    "id": p_id,
                    "page_num": p_num,
                    "summary": p_summary,
                    "text": p_text,
                    "filepath": filepath,
                    "filename": os.path.basename(filepath),
                    "vector_score": score
                }
            else:
                candidate_pages_map[p_id]["vector_score"] = max(candidate_pages_map[p_id]["vector_score"], score)

        # 2. Keyword Match Candidates (including symbols/alphanumeric terms)
        STOP_WORDS = {"what", "is", "the", "and", "of", "in", "to", "a", "for", "with", "on", "at", "by", "an", "this", "that", "these", "those", "are", "who", "whom", "whose"}
        
        # Tokenize query but also split on spaces to keep words with symbols intact (e.g. ID#)
        alphanumeric_keywords = [w for w in tokenize(query) if w not in STOP_WORDS]
        raw_keywords = [w for w in query.split() if w.lower() not in STOP_WORDS]
        
        keywords_to_search = set(alphanumeric_keywords + raw_keywords)
        
        for keyword in keywords_to_search:
            if len(keyword) < 2:
                continue
            like_query = f"%{keyword}%"
            self.cursor.execute('''
                SELECT pages.id, pages.page_num, pages.summary, pages.text, documents.filepath
                FROM pages
                JOIN documents ON pages.doc_id = documents.id
                WHERE pages.text LIKE ?
            ''', (like_query,))
            for p_id, p_num, p_summary, p_text, filepath in self.cursor.fetchall():
                if p_id not in candidate_pages_map:
                    # Calculate vector score
                    self.cursor.execute('SELECT embedding FROM chunks WHERE page_id = ?', (p_id,))
                    chunk_embs = [r[0] for r in self.cursor.fetchall()]
                    max_sim = 0.0
                    if chunk_embs:
                        max_sim = max([np.dot(query_vec, emb) / (np.linalg.norm(query_vec) * np.linalg.norm(emb)) for emb in chunk_embs])
                    candidate_pages_map[p_id] = {
                        "id": p_id,
                        "page_num": p_num,
                        "summary": p_summary,
                        "text": p_text,
                        "filepath": filepath,
                        "filename": os.path.basename(filepath),
                        "vector_score": max_sim
                    }

        # --- Stage 2: Global Hybrid Search Scoring ---
        pool_pages = list(candidate_pages_map.values())
        if not pool_pages:
            print("VECTORLESS RAG: No global candidates matched.")
            return []

        page_bm25_scores = compute_bm25_scores(pool_pages, query)
        page_fuzzy_scores = {p['id']: fuzzy_match_score(p['text'], query) for p in pool_pages}
        page_vector_scores = {p['id']: p['vector_score'] for p in pool_pages}
        
        norm_vector = normalize_scores(page_vector_scores)
        norm_bm25 = normalize_scores(page_bm25_scores)
        norm_fuzzy = normalize_scores(page_fuzzy_scores)
        
        pool_hybrid_scores = {}
        for p in pool_pages:
            p_id = p['id']
            v = norm_vector.get(p_id, 0.0)
            b = norm_bm25.get(p_id, 0.0)
            f = norm_fuzzy.get(p_id, 0.0)
            pool_hybrid_scores[p_id] = 0.4 * v + 0.4 * b + 0.2 * f

        # Sort candidate pool pages by hybrid score
        pool_pages.sort(key=lambda p: pool_hybrid_scores[p['id']], reverse=True)
        
        candidate_pages = pool_pages[:top_k_hybrid]
        print(f"VECTORLESS RAG: Stage 2 Overall Hybrid Search -> Top {len(candidate_pages)} candidates:")
        for idx, p in enumerate(candidate_pages):
            print(f"  - [{idx+1}] {p['filename']} Page {p['page_num']} (Score: {pool_hybrid_scores[p['id']]:.4f})")

        # --- Stage 3: Relational Query Expansion (Multi-Document Linking) ---
        linking_keys = self._extract_linking_keys(query, candidate_pages)
        relational_pages = []
        if linking_keys:
            print(f"VECTORLESS RAG: Stage 3 -> Extracted linking keys: {linking_keys}")
            exclude_paths = {p['filepath'] for p in candidate_pages}
            for key in linking_keys:
                matched = self._search_pages_by_key(key, exclude_paths)
                for m_page in matched:
                    if not any(x['filepath'] == m_page['filepath'] and x['page_num'] == m_page['page_num'] for x in candidate_pages + relational_pages):
                        relational_pages.append(m_page)
                        print(f"  - Relational Link Found: {os.path.basename(m_page['filepath'])}, Page {m_page['page_num']} via key '{key}'")

        # Merge candidate and relational pages
        merged_pages = candidate_pages + relational_pages

        # --- Stage 4: LLM Page Selection ---
        page_summaries_view = ""
        page_map = []
        for idx, p in enumerate(merged_pages):
            matched_map = {}
            for kw in keywords_to_search:
                if len(kw) >= 2:
                    if kw.lower() in p['text'].lower():
                        lower_kw = kw.lower()
                        if lower_kw not in matched_map or kw[0].isupper():
                            matched_map[lower_kw] = kw
            matched_terms = sorted(matched_map.values())
            matched_str = ""
            if matched_terms:
                matched_str = f" | Matched Terms: {', '.join(matched_terms)}"
            page_summaries_view += f"[{idx+1}] Doc: {p['filename']}, Page: {p['page_num']} - Summary: {p['summary']}{matched_str}\n"
            page_map.append({
                "filepath": p['filepath'],
                "page": p['page_num'],
                "text": p['text']
            })

        print("\n--- VECTORLESS RAG: STAGE 4 (PAGE SUMMARIES FOR LLM) ---")
        print(page_summaries_view)
        print("----------------------------------------------\n")

        page_selection_prompt = (
            "You are an expert retrieval assistant. We have identified relevant documents using semantic and lexical search. "
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
            print(f"VECTORLESS RAG: Stage 4 Choice -> {page_choice_text}")

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
            
            # Fetch full text for the chosen pages
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

    def get_all_chats(self) -> list[dict]:
        self.cursor.execute('SELECT id, title, updated_at FROM chats ORDER BY updated_at DESC')
        rows = self.cursor.fetchall()
        return [{"id": r[0], "title": r[1], "updatedAt": r[2]} for r in rows]

    def get_chat_messages(self, chat_id: str) -> list[dict]:
        self.cursor.execute('SELECT messages FROM chats WHERE id = ?', (chat_id,))
        row = self.cursor.fetchone()
        if row and row[0]:
            import json
            try:
                return json.loads(row[0])
            except Exception as e:
                print(f"Error parsing chat messages for {chat_id}: {e}")
                return []
        return []

    def save_chat(self, chat_id: str, title: str, updated_at: float, messages: list[dict]):
        import json
        messages_json = json.dumps(messages)
        self.cursor.execute('''
            INSERT INTO chats (id, title, updated_at, messages)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                updated_at = excluded.updated_at,
                messages = excluded.messages
        ''', (chat_id, title, updated_at, messages_json))
        self.conn.commit()

    def delete_chat(self, chat_id: str):
        self.cursor.execute('DELETE FROM chats WHERE id = ?', (chat_id,))
        self.conn.commit()

# Global instance
vectorless_index = LocalVectorlessIndex()
