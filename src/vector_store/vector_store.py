import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class VectorStore:
    """Persistent Chroma vector store for SGBank Withdrawal Policy"""

    def __init__(
        self,
        persist_directory: str = "./vectordb",
        collection_name: str = "sgbank_withdrawal_policy",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        os.makedirs(persist_directory, exist_ok=True)

        # Explicit persistent client (important)
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Embedding model
        self.embedding_model = SentenceTransformer(embedding_model)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"domain": "sgbank_withdrawal_policy"}
        )

    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:

        if not documents:
            return

        if ids is None:
            # Avoid collisions if re-ingesting
            existing_count = self.collection.count()
            ids = [f"doc_{existing_count + i}" for i in range(len(documents))]

        if metadatas is None:
            metadatas = [{"source": "SGBank Withdrawal Policy"} for _ in documents]

        embeddings = self.embedding_model.encode(
            documents,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).tolist()

        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )

    def search(self, query: str, n_results: int = 3) -> Dict[str, Any]:

        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).tolist()

        return self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )

    def get_collection_count(self) -> int:
        return self.collection.count()

    def clear_collection(self) -> None:
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"domain": "sgbank_withdrawal_policy"}
        )
