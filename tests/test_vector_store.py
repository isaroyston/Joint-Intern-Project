"""
Unit tests for Vector Store
"""

import unittest
import tempfile
import shutil
import os
from src.vector_store import VectorStore


class TestVectorStore(unittest.TestCase):
    """Test cases for VectorStore class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.vector_store = VectorStore(
            persist_directory=self.test_dir,
            collection_name="test_collection"
        )
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test vector store initialization"""
        self.assertIsNotNone(self.vector_store)
        self.assertEqual(self.vector_store.collection_name, "test_collection")
    
    def test_add_documents(self):
        """Test adding documents to vector store"""
        documents = ["Test document 1", "Test document 2"]
        self.vector_store.add_documents(documents)
        count = self.vector_store.get_collection_count()
        self.assertEqual(count, 2)
    
    def test_search(self):
        """Test searching for similar documents"""
        documents = [
            "Python is a programming language",
            "Machine learning is fun",
            "Databases store data"
        ]
        self.vector_store.add_documents(documents)
        
        results = self.vector_store.search("What is Python?", n_results=1)
        self.assertIsNotNone(results)
        self.assertIn('documents', results)
        self.assertTrue(len(results['documents'][0]) > 0)
    
    def test_clear_collection(self):
        """Test clearing the collection"""
        documents = ["Test document"]
        self.vector_store.add_documents(documents)
        self.assertEqual(self.vector_store.get_collection_count(), 1)
        
        self.vector_store.clear_collection()
        self.assertEqual(self.vector_store.get_collection_count(), 0)


if __name__ == '__main__':
    unittest.main()
