from baseclass import BaseRag,VectorStore,BenchmarkRag
import requests
import numpy as np

class OllamaRag(BaseRag):
    def __init__(self, model_name: str, chunk_size: int, top_k: int):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.top_k = top_k

    def query(self, query: str) -> str:
        return "This is a test query"

    def get_embedding(self,text):
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={
                "model": "nomic-embed-text",
                "prompt": text
            }
        )
        response.raise_for_status()
        return np.array(response.json()["embedding"], dtype=np.float32)

    def retrieve_chunks(self, text: str) -> list[str]:
        return []

    def enhance_query(self, query: str) -> str:
        return query
    
    def summarizer(self, chunks: list[str], query: str) -> str:
        return "This is a test summarizer"
    

class FaissVectorStore(VectorStore):
    def __init__(self, docs_path: str,faiss_path: str):
        self.docs_path = docs_path
        self.faiss_path = faiss_path

    def connect(self, text: str) -> None:
        return None
    
    def add_documents(self, documents: list[str]) -> None:
        return None
    
    def update_index(self, docs_path: str, cache_path: str) -> list[str]:
        return super().update_index(docs_path, cache_path)

    def delete_documents(self, documents: list[str]) -> None:
        return super().delete_documents(documents)

    def chunk_text(self, query: str) -> list[str]:
        return super().chunk_text(query)
    
    

class FaissOllamaBenchmark(BenchmarkRag):

    def evaluate_chunking_strategy(self, query: str) -> float:
        return 0.0
    
    def evaluate_embedding_model(self, query: str) -> float:
        return 0.0
    
    def evaluate_retrievals(self, query: str) -> float:
        return 0.0
    
    def evaluate_reranker(self, query: str) -> float:
        return 0.0
    
    def evaluate_document_parsing(self, query: str) -> float:
        return 0.0
    
    def get_results(self) -> dict:
        return {}
    