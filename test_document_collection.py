"""
Test script for DocumentCollection class
"""

import os
import logging
import unittest
from app_rag import DocumentCollection
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TestDocumentCollection(unittest.TestCase):
    """Tests for the DocumentCollection class"""
    
    def setUp(self):
        """Set up test documents"""
        self.doc1 = Document(page_content="Test document 1", metadata={"source": "test.pdf", "page": 1})
        self.doc2 = Document(page_content="Test document 2", metadata={"source": "test.pdf", "page": 2})
        self.doc3 = Document(page_content="Test document 3", metadata={"source": "other.pdf", "page": 1})
    
    def test_init_empty(self):
        """Test initializing an empty collection"""
        collection = DocumentCollection()
        self.assertEqual(len(collection), 0)
        self.assertEqual(collection.sql_count, 0)
        self.assertEqual(collection.vector_count, 0)
        self.assertEqual(collection.fallback_count, 0)
        self.assertEqual(collection.parent_count, 0)
    
    def test_init_with_docs(self):
        """Test initializing with documents"""
        docs = [self.doc1, self.doc2]
        collection = DocumentCollection(docs)
        self.assertEqual(len(collection), 2)
        self.assertEqual(collection[0].page_content, "Test document 1")
        self.assertEqual(collection[1].page_content, "Test document 2")
    
    def test_append(self):
        """Test appending documents"""
        collection = DocumentCollection()
        collection.append(self.doc1)
        self.assertEqual(len(collection), 1)
        self.assertEqual(collection[0].page_content, "Test document 1")
        
        collection.append(self.doc2)
        self.assertEqual(len(collection), 2)
        self.assertEqual(collection[1].page_content, "Test document 2")
    
    def test_extend(self):
        """Test extending with multiple documents"""
        collection = DocumentCollection([self.doc1])
        collection.extend([self.doc2, self.doc3])
        self.assertEqual(len(collection), 3)
        self.assertEqual(collection[2].page_content, "Test document 3")
    
    def test_iteration(self):
        """Test iterating through the collection"""
        collection = DocumentCollection([self.doc1, self.doc2, self.doc3])
        docs = [doc for doc in collection]
        self.assertEqual(len(docs), 3)
        self.assertEqual(docs[0].page_content, "Test document 1")
        self.assertEqual(docs[1].page_content, "Test document 2")
        self.assertEqual(docs[2].page_content, "Test document 3")
    
    def test_slicing(self):
        """Test slicing the collection"""
        collection = DocumentCollection([self.doc1, self.doc2, self.doc3])
        collection.sql_count = 2
        collection.vector_count = 1
        
        # Get slice
        subset = collection[1:3]
        self.assertEqual(len(subset), 2)
        self.assertEqual(subset[0].page_content, "Test document 2")
        self.assertEqual(subset[1].page_content, "Test document 3")
        
        # Verify metrics are preserved
        self.assertEqual(subset.sql_count, 2)
        self.assertEqual(subset.vector_count, 1)
    
    def test_copy(self):
        """Test copying the collection"""
        collection = DocumentCollection([self.doc1, self.doc2])
        collection.sql_count = 1
        collection.vector_count = 1
        
        copy = collection.copy()
        self.assertEqual(len(copy), 2)
        self.assertEqual(copy.sql_count, 1)
        self.assertEqual(copy.vector_count, 1)
        
        # Modify the copy and verify the original is unchanged
        copy.append(self.doc3)
        copy.sql_count = 2
        self.assertEqual(len(collection), 2)
        self.assertEqual(collection.sql_count, 1)
        self.assertEqual(len(copy), 3)
        self.assertEqual(copy.sql_count, 2)
    
    def test_metrics(self):
        """Test setting and getting metrics"""
        collection = DocumentCollection([self.doc1, self.doc2])
        collection.sql_count = 5
        collection.vector_count = 10
        collection.fallback_count = 2
        collection.parent_count = 3
        
        self.assertEqual(collection.sql_count, 5)
        self.assertEqual(collection.vector_count, 10)
        self.assertEqual(collection.fallback_count, 2)
        self.assertEqual(collection.parent_count, 3)

if __name__ == "__main__":
    unittest.main() 