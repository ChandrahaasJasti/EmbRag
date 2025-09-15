from baseclass import VectorStore
from engines import NomicEngine
from pinecone import Pinecone


class PineconeVectorStore(VectorStore):
    def __init__(self, index_name: str, engine: NomicEngine):
        self.index_name = index_name
        self.pc = Pinecone(api_key="********-****-****-****-************")
        self.index = self.pc.Index(self.index_name)

    def connect(self, text: str) -> None:
        pass

    def add_documents(self, documents: list[str]) -> None:
        pass

    def update_index(self, docs_path: str, cache_path: str) -> list[str]:
        pass

    def delete_documents(self, documents: list[str]) -> None:
        pass

    def retrieve_chunks(self, query: str) -> list[str]:
        pass