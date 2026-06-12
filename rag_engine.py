import os
import json
import pickle
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Robust imports for varying LangChain versions
try:
    from langchain_community.retrievers import BM25Retriever
except ImportError:
    from langchain_classic.retrievers import BM25Retriever

try:
    from langchain.retrievers import EnsembleRetriever
except (ImportError, ModuleNotFoundError):
    try:
        from langchain_classic.retrievers import EnsembleRetriever
    except ImportError:
        raise ImportError("Could not find EnsembleRetriever in langchain or langchain_classic.")

DATA_DIR = "data"
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index")
BM25_PATH = os.path.join(DATA_DIR, "bm25_retriever.pkl")

class HybridDocumentIndex:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.vector_store = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        self.load_index()

    def _update_ensemble(self):
        if self.vector_store and self.bm25_retriever:
            faiss_retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[self.bm25_retriever, faiss_retriever],
                weights=[0.5, 0.5]
            )
        elif self.vector_store:
            self.ensemble_retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})

    def add_documents(self, new_docs):
        """
        new_docs: list of dicts [{'filepath': str, 'page': int, 'text': str}]
        """
        langchain_docs = []
        for doc in new_docs:
            # Split the page text into smaller overlapping chunks
            chunks = self.text_splitter.split_text(doc['text'])
            for chunk in chunks:
                lc_doc = Document(
                    page_content=chunk,
                    metadata={
                        "filepath": doc['filepath'],
                        "page": doc['page']
                    }
                )
                langchain_docs.append(lc_doc)

        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(langchain_docs, self.embeddings)
        else:
            self.vector_store.add_documents(langchain_docs)
        
        all_lc_docs = self._get_all_documents()
        self.bm25_retriever = BM25Retriever.from_documents(all_lc_docs)
        self.bm25_retriever.k = 5
        
        self._update_ensemble()
        self.save_index()

    def _get_all_documents(self):
        if not self.vector_store:
            return []
        return list(self.vector_store.docstore._dict.values())

    def save_index(self):
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        if self.vector_store:
            self.vector_store.save_local(FAISS_INDEX_PATH)
        if self.bm25_retriever:
            with open(BM25_PATH, 'wb') as f:
                pickle.dump(self.bm25_retriever, f)

    def load_index(self):
        if os.path.exists(os.path.join(FAISS_INDEX_PATH, "index.faiss")):
            try:
                self.vector_store = FAISS.load_local(
                    FAISS_INDEX_PATH, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"Error loading FAISS index: {e}")
                self.vector_store = None
        
        if os.path.exists(BM25_PATH):
            try:
                with open(BM25_PATH, 'rb') as f:
                    self.bm25_retriever = pickle.load(f)
            except:
                self.bm25_retriever = None
        
        if self.vector_store and not self.bm25_retriever:
            all_docs = self._get_all_documents()
            if all_docs:
                self.bm25_retriever = BM25Retriever.from_documents(all_docs)
                self.bm25_retriever.k = 5

        self._update_ensemble()

    def search(self, query, top_n=5):
        if self.ensemble_retriever is None:
            return []
            
        results = self.ensemble_retriever.invoke(query)
        
        formatted_results = []
        for doc in results[:top_n]:
            formatted_results.append({
                'text': doc.page_content,
                'filepath': doc.metadata['filepath'],
                'page': doc.metadata['page']
            })
            
        return formatted_results

# Global singleton instance
doc_index = HybridDocumentIndex()
