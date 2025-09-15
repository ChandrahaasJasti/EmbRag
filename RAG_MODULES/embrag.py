from baseclass import BaseRag, VectorStore
from llm import LLM

class EmbRag(BaseRag):
    def __init__(self,vector_store: VectorStore):
        self.vectorDB = vector_store
        self.llm_obj = LLM()


    def retrieve_memory(self, query: str) -> list[str]:
        enhanced_query = self.enhance_query(query)
        chunks = self.vectorDB.retrieve_chunks(enhanced_query)
        return str(self.summarizer(chunks, query))

    def enhance_query(self, query: str) -> str:
        pass

    def summarizer(self, chunks: list[str], query: str) -> str:
        pass
    
    