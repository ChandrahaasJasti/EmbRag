from baseclass import VectorStore

class PineconeVectorStore(VectorStore):
    def connect(self):
        pass

    def add_documents(self, documents: list[str]) -> None:
        return super().add_documents(documents)

    def delete_documents(self, documents: list[str]) -> None:
        return super().delete_documents(documents)

    def chunk_text(self, query: str) -> list[str]:
        return super().chunk_text(query)

    def retrieve_chunks(self):
        pass