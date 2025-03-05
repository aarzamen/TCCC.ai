"""
Test suite for the Document Library module.

This module contains tests for:
- Document Library initialization
- Document addition and retrieval
- Document chunking
- Vector embedding
- Semantic search
- Caching mechanism
"""

import os
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import faiss

from tccc.document_library import DocumentLibrary, DocumentProcessor, CacheManager
from tccc.utils.config import Config


class TestDocumentLibrary(unittest.TestCase):
    """Test case for the Document Library module."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directories for testing
        self.temp_dir = tempfile.mkdtemp()
        self.documents_dir = os.path.join(self.temp_dir, "documents")
        self.index_dir = os.path.join(self.temp_dir, "index")
        self.cache_dir = os.path.join(self.temp_dir, "cache")
        self.embeddings_dir = os.path.join(self.temp_dir, "embeddings")
        
        os.makedirs(self.documents_dir, exist_ok=True)
        os.makedirs(self.index_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.embeddings_dir, exist_ok=True)
        
        # Create a sample document file
        self.sample_doc_path = os.path.join(self.documents_dir, "sample.txt")
        with open(self.sample_doc_path, "w") as f:
            f.write("This is a sample document for testing the Document Library.")
        
        # Create test configuration
        self.config = {
            "storage": {
                "type": "file",
                "base_dir": self.documents_dir,
                "index_path": self.index_dir,
                "cache_dir": self.cache_dir,
                "retention_days": 0,
                "max_size_mb": 10,
                "compression_level": 0
            },
            "embedding": {
                "model_name": "all-MiniLM-L12-v2",
                "cache_dir": self.embeddings_dir,
                "dimension": 384,
                "max_seq_length": 512,
                "use_gpu": False,
                "batch_size": 8,
                "normalize": True
            },
            "search": {
                "provider": "faiss",
                "semantic_search": True,
                "embedding_model": "all-MiniLM-L12-v2",
                "min_similarity": 0.7,
                "default_results": 5,
                "max_results": 20,
                "fuzzy_matching": True,
                "cache_timeout": 3600
            },
            "indexing": {
                "auto_index": True,
                "update_frequency": 300,
                "max_document_size_kb": 1024,
                "chunk_size": 100,
                "chunk_overlap": 20,
                "document_types": [
                    {"text": [".txt", ".md"]},
                    {"pdf": [".pdf"]},
                    {"docx": [".docx"]},
                    {"html": [".html", ".htm"]}
                ]
            }
        }
    
    def tearDown(self):
        """Clean up after test."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    @patch("sentence_transformers.SentenceTransformer")
    @patch("faiss.IndexFlatIP")
    def test_initialization(self, mock_faiss, mock_transformer):
        """Test DocumentLibrary initialization."""
        # Mock FAISS index and transformer
        mock_index = MagicMock()
        mock_faiss.return_value = mock_index
        
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model
        
        # Initialize DocumentLibrary
        doc_lib = DocumentLibrary()
        result = doc_lib.initialize(self.config)
        
        # Check initialization
        self.assertTrue(result)
        self.assertTrue(doc_lib.initialized)
        self.assertEqual(doc_lib.config, self.config)
        self.assertEqual(doc_lib.next_doc_id, 1)
        self.assertEqual(len(doc_lib.documents), 0)
        self.assertEqual(len(doc_lib.chunks), 0)
        
        # Check that model was loaded
        mock_transformer.assert_called_once_with(
            self.config["embedding"]["model_name"],
            cache_folder=self.config["embedding"]["cache_dir"]
        )
        
        # Check cache manager and document processor initialization
        self.assertIsNotNone(doc_lib.cache_manager)
        self.assertIsNotNone(doc_lib.document_processor)
    
    @patch("sentence_transformers.SentenceTransformer")
    @patch("faiss.IndexFlatIP")
    def test_add_document(self, mock_faiss, mock_transformer):
        """Test adding a document to the library."""
        # Mock FAISS index and transformer
        mock_index = MagicMock()
        mock_faiss.return_value = mock_index
        
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(1, 384)  # Mock embeddings
        mock_transformer.return_value = mock_model
        
        # Initialize DocumentLibrary
        doc_lib = DocumentLibrary()
        doc_lib.initialize(self.config)
        
        # Mock document processor
        doc_lib.document_processor = MagicMock()
        doc_lib.document_processor.process_document.return_value = {
            "success": True,
            "text": "This is a sample document for testing the Document Library.",
            "metadata": {
                "content_type": "text/plain",
                "file_size": 100
            }
        }
        
        # Add document
        document_data = {
            "file_path": self.sample_doc_path,
            "metadata": {
                "category": "Test",
                "source": "Unit Test"
            }
        }
        
        doc_id = doc_lib.add_document(document_data)
        
        # Check that document was added
        self.assertIsNotNone(doc_id)
        self.assertEqual(doc_lib.next_doc_id, 2)
        self.assertEqual(len(doc_lib.documents), 1)
        self.assertTrue(doc_id in doc_lib.documents)
        
        # Check document metadata
        doc_metadata = doc_lib.documents[doc_id]
        self.assertEqual(doc_metadata["file_path"], self.sample_doc_path)
        self.assertEqual(doc_metadata["metadata"]["category"], "Test")
        self.assertEqual(doc_metadata["metadata"]["source"], "Unit Test")
        self.assertEqual(doc_metadata["metadata"]["file_name"], "sample.txt")
        
        # Verify document processor was called
        doc_lib.document_processor.process_document.assert_called_once_with(self.sample_doc_path)
    
    @patch("sentence_transformers.SentenceTransformer")
    @patch("faiss.IndexFlatIP")
    def test_query(self, mock_faiss, mock_transformer):
        """Test querying the document library."""
        # Mock FAISS index
        mock_index = MagicMock()
        mock_index.ntotal = 1  # Pretend we have 1 vector
        mock_index.search.return_value = (
            np.array([[0.95]]),  # Score
            np.array([[0]])      # Index
        )
        mock_faiss.return_value = mock_index
        
        # Mock transformer
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(384)  # Mock query embedding
        mock_transformer.return_value = mock_model
        
        # Initialize DocumentLibrary
        doc_lib = DocumentLibrary()
        doc_lib.initialize(self.config)
        
        # Mock cache manager
        doc_lib.cache_manager = MagicMock()
        doc_lib.cache_manager.generate_key.return_value = "test_key"
        doc_lib.cache_manager.get.return_value = None  # No cache hit
        
        # Set up a fake document and chunk
        doc_lib.documents = {
            "1": {
                "id": "1",
                "metadata": {"category": "Test"},
                "file_path": self.sample_doc_path
            }
        }
        
        doc_lib.chunks = {
            "1_0": {
                "chunk_id": "1_0",
                "doc_id": "1",
                "text": "This is a sample chunk.",
                "start_char": 0,
                "end_char": 100
            }
        }
        
        # Run query
        query_result = doc_lib.query("test query", 1)
        
        # Check that query worked
        self.assertEqual(query_result["query"], "test query")
        self.assertEqual(query_result["total_results"], 1)
        self.assertEqual(len(query_result["results"]), 1)
        
        # Check result content
        result = query_result["results"][0]
        self.assertEqual(result["document_id"], "1")
        self.assertEqual(result["text"], "This is a sample chunk.")
        self.assertEqual(result["metadata"]["category"], "Test")
        self.assertAlmostEqual(result["score"], 0.95)
        
        # Verify cache was checked and result was cached
        doc_lib.cache_manager.generate_key.assert_called_once()
        doc_lib.cache_manager.get.assert_called_once_with("test_key")
        doc_lib.cache_manager.set.assert_called_once()
    
    @patch("sentence_transformers.SentenceTransformer")
    @patch("faiss.IndexFlatIP")
    def test_cache_hit(self, mock_faiss, mock_transformer):
        """Test cache hit when querying."""
        # Mock FAISS and transformer
        mock_index = MagicMock()
        mock_faiss.return_value = mock_index
        
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model
        
        # Initialize DocumentLibrary
        doc_lib = DocumentLibrary()
        doc_lib.initialize(self.config)
        
        # Mock cache manager with a cache hit
        cache_result = {
            "query": "test query",
            "results": [
                {
                    "document_id": "1",
                    "text": "This is a cached result.",
                    "score": 0.95,
                    "metadata": {"category": "Test"}
                }
            ],
            "total_results": 1,
            "processing_time": 0.1,
            "cache_hit": True
        }
        
        doc_lib.cache_manager = MagicMock()
        doc_lib.cache_manager.generate_key.return_value = "test_key"
        doc_lib.cache_manager.get.return_value = cache_result
        
        # Run query
        query_result = doc_lib.query("test query", 1)
        
        # Check that we got the cached result
        self.assertEqual(query_result, cache_result)
        self.assertTrue(query_result["cache_hit"])
        
        # Verify cache was checked but not updated
        doc_lib.cache_manager.generate_key.assert_called_once()
        doc_lib.cache_manager.get.assert_called_once_with("test_key")
        doc_lib.cache_manager.set.assert_not_called()
        
        # Verify model was not used
        mock_model.encode.assert_not_called()
        mock_index.search.assert_not_called()
    
    @patch("sentence_transformers.SentenceTransformer")
    @patch("faiss.IndexFlatIP")
    def test_status(self, mock_faiss, mock_transformer):
        """Test getting status information."""
        # Mock FAISS and transformer
        mock_index = MagicMock()
        mock_index.ntotal = 5  # Pretend we have 5 vectors
        mock_faiss.return_value = mock_index
        
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model
        
        # Initialize DocumentLibrary
        doc_lib = DocumentLibrary()
        doc_lib.initialize(self.config)
        
        # Add some fake documents and chunks
        doc_lib.documents = {
            "1": {"id": "1"},
            "2": {"id": "2"}
        }
        
        doc_lib.chunks = {
            "1_0": {"chunk_id": "1_0", "doc_id": "1"},
            "1_1": {"chunk_id": "1_1", "doc_id": "1"},
            "2_0": {"chunk_id": "2_0", "doc_id": "2"}
        }
        
        # Mock cache manager stats
        doc_lib.cache_manager = MagicMock()
        doc_lib.cache_manager.get_stats.return_value = {
            "memory_entries": 2,
            "disk_entries": 5,
            "disk_size_mb": 0.1
        }
        
        # Get status
        status = doc_lib.get_status()
        
        # Check status
        self.assertEqual(status["status"], "initialized")
        self.assertEqual(status["documents"]["count"], 2)
        self.assertEqual(status["documents"]["chunks"], 3)
        self.assertEqual(status["index"]["vectors"], 5)
        self.assertEqual(status["index"]["dimension"], 384)
        self.assertEqual(status["model"]["name"], "all-MiniLM-L12-v2")
        self.assertEqual(status["cache"]["memory_entries"], 2)
        self.assertEqual(status["cache"]["disk_entries"], 5)


class TestDocumentProcessor(unittest.TestCase):
    """Test case for the DocumentProcessor class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test configuration
        self.config = {
            "indexing": {
                "max_document_size_kb": 1024,
                "document_types": [
                    {"text": [".txt", ".md"]},
                    {"pdf": [".pdf"]},
                    {"docx": [".docx"]},
                    {"html": [".html", ".htm"]}
                ]
            }
        }
        
        # Create sample documents
        self.text_file = os.path.join(self.temp_dir, "sample.txt")
        with open(self.text_file, "w") as f:
            f.write("This is a sample text document.")
    
    def tearDown(self):
        """Clean up after test."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test DocumentProcessor initialization."""
        processor = DocumentProcessor(self.config)
        
        # Check supported extensions
        self.assertIn("txt", processor.supported_extensions)
        self.assertIn("md", processor.supported_extensions)
        self.assertIn("pdf", processor.supported_extensions)
        self.assertIn("docx", processor.supported_extensions)
        self.assertIn("html", processor.supported_extensions)
        
        # Check format handlers
        self.assertIn("text", processor.format_handlers)
        self.assertIn("pdf", processor.format_handlers)
        self.assertIn("docx", processor.format_handlers)
        self.assertIn("html", processor.format_handlers)
    
    def test_process_text_file(self):
        """Test processing a text file."""
        processor = DocumentProcessor(self.config)
        
        # Process text file
        result = processor.process_document(self.text_file)
        
        # Check result
        self.assertTrue(result["success"])
        self.assertEqual(result["text"], "This is a sample text document.")
        self.assertEqual(result["metadata"]["file_name"], "sample.txt")
        self.assertEqual(result["metadata"]["content_type"], "text/plain")
        self.assertGreater(result["metadata"]["file_size"], 0)
    
    def test_unsupported_format(self):
        """Test handling of unsupported format."""
        processor = DocumentProcessor(self.config)
        
        # Create unsupported file
        unsupported_file = os.path.join(self.temp_dir, "sample.xyz")
        with open(unsupported_file, "w") as f:
            f.write("This is an unsupported file format.")
        
        # Try to process it
        result = processor.process_document(unsupported_file)
        
        # Check result (should fall back to text format)
        self.assertTrue(result["success"])
        self.assertEqual(result["text"], "This is an unsupported file format.")
    
    def test_nonexistent_file(self):
        """Test handling of nonexistent file."""
        processor = DocumentProcessor(self.config)
        
        # Try to process nonexistent file
        result = processor.process_document(os.path.join(self.temp_dir, "nonexistent.txt"))
        
        # Check result
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        self.assertIn("File not found", result["error"])
    
    def test_clean_text(self):
        """Test text cleaning functionality."""
        processor = DocumentProcessor(self.config)
        
        # Test with extra whitespace and newlines
        dirty_text = "  This  has   extra \n\n\n  whitespace  \n  and newlines.  "
        clean_text = processor._clean_text(dirty_text)
        
        # Check cleaned text
        self.assertEqual(clean_text, "This has extra\n\nwhitespace and newlines.")


class TestCacheManager(unittest.TestCase):
    """Test case for the CacheManager class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = os.path.join(self.temp_dir, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Create test configuration
        self.config = {
            "storage": {
                "cache_dir": self.cache_dir,
                "max_size_mb": 10
            },
            "search": {
                "cache_timeout": 3600
            }
        }
    
    def tearDown(self):
        """Clean up after test."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test CacheManager initialization."""
        cache_manager = CacheManager(self.config)
        
        # Check initialization
        self.assertEqual(cache_manager.cache_dir, self.cache_dir)
        self.assertEqual(cache_manager.cache_timeout, 3600)
        self.assertEqual(len(cache_manager.memory_cache), 0)
    
    def test_set_and_get(self):
        """Test setting and getting cache entries."""
        cache_manager = CacheManager(self.config)
        
        # Create test result
        test_result = {
            "query": "test query",
            "results": [{"text": "test result"}],
            "total_results": 1
        }
        
        # Set cache entry
        cache_manager.set("test_key", test_result)
        
        # Check that it was added to memory cache
        self.assertIn("test_key", cache_manager.memory_cache)
        
        # Check that it was written to disk
        cache_file = os.path.join(self.cache_dir, "test_key.json")
        self.assertTrue(os.path.exists(cache_file))
        
        # Get cache entry
        cached_result = cache_manager.get("test_key")
        
        # Check cached result
        self.assertEqual(cached_result["query"], "test query")
        self.assertEqual(cached_result["results"][0]["text"], "test result")
        self.assertEqual(cached_result["total_results"], 1)
        self.assertTrue(cached_result["cache_hit"])
    
    def test_invalidate(self):
        """Test invalidating cache entries."""
        cache_manager = CacheManager(self.config)
        
        # Create test result
        test_result = {
            "query": "test query",
            "results": [{"text": "test result"}],
            "total_results": 1
        }
        
        # Set cache entry
        cache_manager.set("test_key", test_result)
        
        # Invalidate cache entry
        result = cache_manager.invalidate("test_key")
        
        # Check that it was invalidated
        self.assertTrue(result)
        self.assertNotIn("test_key", cache_manager.memory_cache)
        
        cache_file = os.path.join(self.cache_dir, "test_key.json")
        self.assertFalse(os.path.exists(cache_file))
        
        # Try to get invalidated entry
        cached_result = cache_manager.get("test_key")
        
        # Should return None
        self.assertIsNone(cached_result)
    
    def test_clear(self):
        """Test clearing all cache entries."""
        cache_manager = CacheManager(self.config)
        
        # Create multiple test results
        for i in range(3):
            cache_manager.set(f"key_{i}", {"query": f"query_{i}"})
        
        # Verify entries were added
        self.assertEqual(len(cache_manager.memory_cache), 3)
        self.assertEqual(len(os.listdir(self.cache_dir)), 3)
        
        # Clear cache
        cache_manager.clear()
        
        # Verify all entries were cleared
        self.assertEqual(len(cache_manager.memory_cache), 0)
        self.assertEqual(len([f for f in os.listdir(self.cache_dir) if f.endswith('.json')]), 0)
    
    def test_generate_key(self):
        """Test generating cache keys."""
        # Generate keys for different queries and parameters
        key1 = CacheManager.generate_key("query1")
        key2 = CacheManager.generate_key("query2")
        key3 = CacheManager.generate_key("query1", {"n_results": 5})
        key4 = CacheManager.generate_key("query1", {"n_results": 10})
        
        # Keys should be different
        self.assertNotEqual(key1, key2)
        self.assertNotEqual(key1, key3)
        self.assertNotEqual(key3, key4)
        
        # Same query with same params should give same key
        key5 = CacheManager.generate_key("query1")
        key6 = CacheManager.generate_key("query1", {"n_results": 5})
        
        self.assertEqual(key1, key5)
        self.assertEqual(key3, key6)
        
        # Check parameter count suffix
        self.assertTrue(key1.endswith("_0"))
        self.assertTrue(key3.endswith("_1"))


if __name__ == "__main__":
    unittest.main()