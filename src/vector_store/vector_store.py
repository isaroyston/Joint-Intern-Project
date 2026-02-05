"""
Vector Database Store Module
Provides functionality for storing and retrieving embeddings using ChromaDB
"""

import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class VectorStore:
    """Vector database store using ChromaDB for efficient similarity search"""
    
    def __init__(self, 
                 persist_directory: str = "./data/vector_db",
                 collection_name: str = "chatbot_knowledge",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the vector store
        
        Args:
            persist_directory: Directory to persist the database
            collection_name: Name of the collection
            embedding_model: Sentence transformer model name
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Chatbot knowledge base"}
        )
    
    def add_documents(self, 
                      documents: List[str], 
                      metadatas: Optional[List[Dict[str, Any]]] = None,
                      ids: Optional[List[str]] = None) -> None:
        """
        Add documents to the vector store
        
        Args:
            documents: List of text documents to add
            metadatas: Optional metadata for each document
            ids: Optional IDs for each document
        """
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
        
        if metadatas is None:
            metadatas = [{"source": "manual"} for _ in documents]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(documents).tolist()
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    def search(self, 
               query: str, 
               n_results: int = 3) -> Dict[str, Any]:
        """
        Search for similar documents
        
        Args:
            query: Query text
            n_results: Number of results to return
            
        Returns:
            Dictionary containing documents, distances, and metadatas
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        # Search
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        return results
    
    def delete_collection(self) -> None:
        """Delete the collection"""
        self.client.delete_collection(name=self.collection_name)
    
    def get_collection_count(self) -> int:
        """Get the number of documents in the collection"""
        return self.collection.count()
    
    def clear_collection(self) -> None:
        """Clear all documents from the collection"""
        # Delete and recreate collection
        self.delete_collection()
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Chatbot knowledge base"}
        )
