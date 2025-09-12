from abc import ABC, abstractmethod
from typing_extensions import Optional
import numpy as np

"""------------------------------------------------------------------------------------------------"""

class BaseRag(ABC):
    """
        Base class for all RAG pipelines.
    """
    @abstractmethod
    def query(self, text: str) -> str:
        pass

    @abstractmethod
    def get_embeddings(self, text: str) -> np.ndarray:
        pass

    @abstractmethod
    def retrieve_chunks(self, text: str) -> list[str]:
        pass
    
    @abstractmethod
    def enhance_query(self, query: str) -> str:
        pass

    @abstractmethod
    def summarizer(self, chunks: list[str], query: str) -> str:
        pass

"""------------------------------------------------------------------------------------------------"""

class VectorStore(ABC):
    @abstractmethod
    def connect(self, text: str) -> None:
        pass

    @abstractmethod
    def add_documents(self, documents: list[str]) -> None:
        pass

    @abstractmethod
    def update_index(self, query: str) -> list[str]: #That auto caching and chunking logic here
        pass

    @abstractmethod
    def delete_documents(self, documents: list[str]) -> None:
        pass

    @abstractmethod
    def chunk_text(self, query: str) -> list[str]:
        pass

"""------------------------------------------------------------------------------------------------"""

class BenchmarkRag(ABC):
    @abstractmethod
    def evaluate_chunking_strategy(self, query: str) -> float:
        pass

    @abstractmethod
    def evaluate_embedding_model(self, query: str) -> float:
        pass

    @abstractmethod
    def evaluate_retrievals(self, query: str) -> float:
        pass

    @abstractmethod
    def evaluate_reranker(self, query: str) -> float:
        pass

    @abstractmethod
    def evaluate_document_parsing(self, query: str) -> float:
        pass
    
    @abstractmethod
    def get_results(self) -> dict:
        pass
    
"""------------------------------------------------------------------------------------------------"""