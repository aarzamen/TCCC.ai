"""
Vector Store for the TCCC.ai Document Library.

This module provides a vector database implementation for storing and retrieving
document embeddings, optimized for semantic search with FAISS.
"""

import os
import json
import time
import logging
import pickle
import tempfile
import threading
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

# Import conditionally to handle missing dependencies
try:
    import faiss
    import torch
    from sentence_transformers import SentenceTransformer
    MOCK_MODE = False
except ImportError:
    MOCK_MODE = True
    
from tccc.utils.logging import get_logger

logger = get_logger(__name__)

class MockEmbeddingModel:
    """Mock embedding model for testing."""
    
    def __init__(self, dimension=384):
        self.dimension = dimension
        
    def encode(self, texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True):
        """Generate fake embeddings."""
        if isinstance(texts, str):
            texts = [texts]
        return np.random.rand(len(texts), self.dimension).astype(np.float32)

class MockFaissIndex:
    """Mock FAISS index for testing."""
    
    def __init__(self, dimension=384):
        self.dimension = dimension
        self.vectors = []
        self.ntotal = 0
        
    def add(self, embeddings):
        """Add vectors to index."""
        self.vectors.extend(embeddings)
        self.ntotal = len(self.vectors)
        
    def search(self, query_vectors, k):
        """Search for similar vectors."""
        # Return random scores and indices
        num_queries = len(query_vectors)
        scores = np.random.rand(num_queries, min(k, max(1, self.ntotal))).astype(np.float32)
        indices = np.random.randint(0, max(1, self.ntotal), (num_queries, min(k, max(1, self.ntotal))))
        return scores, indices

class VectorStore:
    """
    Vector Store implementation using FAISS.
    
    This class manages document embeddings for semantic search, providing:
    - Efficient vector storage with FAISS
    - Document-to-vector mapping
    - Batch indexing capabilities
    - Optimized similarity search
    - Support for incremental updates
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Vector Store.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.index = None
        self.index_path = os.path.join(config["storage"]["index_path"], "document_index.faiss")
        self.mapping_path = os.path.join(config["storage"]["index_path"], "chunk_mapping.pkl")
        self.chunk_to_id = {}
        self.id_to_metadata = {}
        self.embedding_model = None
        self.dimension = config["embedding"]["dimension"]
        self.model_name = config["embedding"]["model_name"]
        self.use_gpu = config["embedding"]["use_gpu"]
        self.batch_size = config["embedding"]["batch_size"]
        self.normalize = config["embedding"]["normalize"]
        self.lock = threading.Lock()
        self.initialized = False
        self.mock_mode = MOCK_MODE
    
    def initialize(self) -> bool:
        """
        Initialize the Vector Store.
        
        Returns:
            Success status
        """
        try:
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            
            # Check if we're using mock mode
            if self.mock_mode:
                logger.warning("Using mock Vector Store due to missing dependencies")
                self.embedding_model = MockEmbeddingModel(dimension=self.dimension)
                self.index = MockFaissIndex(dimension=self.dimension)
                
                # Initialize with some fake data
                for i in range(10):
                    doc_id = f"doc_{i}"
                    self.chunk_to_id[i] = doc_id
                    self.id_to_metadata[doc_id] = {
                        "title": f"Mock Document {i}",
                        "source": "mock-source"
                    }
                
                # Add some vectors to the mock index
                mock_embeddings = np.random.rand(10, self.dimension).astype(np.float32)
                self.index.add(mock_embeddings)
                
                self.initialized = True
                logger.info(f"Mock Vector Store initialized with {self.index.ntotal} vectors")
                return True
            
            # Real initialization with actual dependencies
            # Load embedding model
            logger.info(f"Loading embedding model: {self.model_name}")
            self.embedding_model = SentenceTransformer(
                self.model_name,
                cache_folder=self.config["embedding"]["cache_dir"],
                device="cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"
            )
            
            # Load or create index
            if os.path.exists(self.index_path) and os.path.exists(self.mapping_path):
                logger.info("Loading existing FAISS index")
                self.index = faiss.read_index(self.index_path)
                with open(self.mapping_path, 'rb') as f:
                    self.chunk_to_id = pickle.load(f)
            else:
                logger.info("Creating new FAISS index")
                # Use inner product index for normalized vectors
                if self.normalize:
                    self.index = faiss.IndexFlatIP(self.dimension)
                else:
                    # Use L2 distance for non-normalized vectors
                    self.index = faiss.IndexFlatL2(self.dimension)
            
            # Load id_to_metadata if available
            metadata_path = os.path.join(os.path.dirname(self.index_path), "id_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.id_to_metadata = json.load(f)
            
            self.initialized = True
            logger.info(f"Vector Store initialized with {self.index.ntotal} vectors")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Vector Store: {str(e)}")
            return False
    
    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            NumPy array of embeddings
        """
        try:
            # Generate embeddings
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=self.normalize
            )
            
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise
    
    def add(self, 
            texts: List[str], 
            ids: List[str], 
            metadata: Optional[List[Dict[str, Any]]] = None) -> bool:
        """
        Add vectors to the index.
        
        Args:
            texts: List of text strings
            ids: List of document IDs
            metadata: Optional list of metadata dictionaries
            
        Returns:
            Success status
        """
        if not self.initialized:
            logger.error("Vector Store not initialized")
            return False
        
        if len(texts) != len(ids):
            logger.error(f"Number of texts ({len(texts)}) does not match number of IDs ({len(ids)})")
            return False
        
        if metadata is not None and len(metadata) != len(texts):
            logger.error(f"Number of metadata entries ({len(metadata)}) does not match number of texts ({len(texts)})")
            return False
        
        try:
            # Generate embeddings
            logger.debug(f"Generating embeddings for {len(texts)} texts")
            embeddings = self._embed_texts(texts)
            
            # Add to index
            with self.lock:
                # Get current index size
                start_index = self.index.ntotal
                
                # Add embeddings to index
                self.index.add(np.array(embeddings).astype('float32'))
                
                # Update mapping
                for i, doc_id in enumerate(ids):
                    # Map from vector index to document ID
                    vector_idx = start_index + i
                    self.chunk_to_id[vector_idx] = doc_id
                    
                    # Store metadata if provided
                    if metadata is not None:
                        self.id_to_metadata[doc_id] = metadata[i]
                
                # Save index and mapping
                self._save()
            
            logger.info(f"Added {len(texts)} vectors to index")
            return True
        except Exception as e:
            logger.error(f"Failed to add vectors to index: {str(e)}")
            return False
    
    def search(self, 
               query: str, 
               k: int = 5, 
               threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            query: Query text
            k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of result dictionaries
        """
        if not self.initialized:
            logger.error("Vector Store not initialized")
            return []
        
        if self.index.ntotal == 0:
            logger.warning("Index is empty")
            return []
        
        try:
            # Generate embedding for query
            query_embedding = self._embed_texts([query])[0]
            
            # Search index
            k = min(k, self.index.ntotal)
            scores, indices = self.index.search(
                np.array([query_embedding]).astype('float32'), 
                k
            )
            
            # Process results
            results = []
            for i, idx in enumerate(indices[0]):
                score = float(scores[0][i])
                
                # Skip results below threshold
                if threshold > 0 and score < threshold:
                    continue
                
                # Get document ID from mapping
                doc_id = self.chunk_to_id.get(int(idx))
                if doc_id is None:
                    continue
                
                # Get metadata if available
                metadata = self.id_to_metadata.get(doc_id, {})
                
                # Add to results
                results.append({
                    "id": doc_id,
                    "score": score,
                    "metadata": metadata
                })
            
            return results
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []
    
    def batch_search(self, 
                     queries: List[str], 
                     k: int = 5, 
                     threshold: float = 0.0) -> List[List[Dict[str, Any]]]:
        """
        Search for multiple queries in batch.
        
        Args:
            queries: List of query texts
            k: Number of results per query
            threshold: Minimum similarity threshold
            
        Returns:
            List of result lists
        """
        if not self.initialized:
            logger.error("Vector Store not initialized")
            return [[] for _ in queries]
        
        if self.index.ntotal == 0:
            logger.warning("Index is empty")
            return [[] for _ in queries]
        
        try:
            # Generate embeddings for queries
            query_embeddings = self._embed_texts(queries)
            
            # Search index
            k = min(k, self.index.ntotal)
            scores, indices = self.index.search(
                query_embeddings.astype('float32'), 
                k
            )
            
            # Process results
            all_results = []
            for q_idx in range(len(queries)):
                results = []
                for i, idx in enumerate(indices[q_idx]):
                    score = float(scores[q_idx][i])
                    
                    # Skip results below threshold
                    if threshold > 0 and score < threshold:
                        continue
                    
                    # Get document ID from mapping
                    doc_id = self.chunk_to_id.get(int(idx))
                    if doc_id is None:
                        continue
                    
                    # Get metadata if available
                    metadata = self.id_to_metadata.get(doc_id, {})
                    
                    # Add to results
                    results.append({
                        "id": doc_id,
                        "score": score,
                        "metadata": metadata
                    })
                
                all_results.append(results)
            
            return all_results
        except Exception as e:
            logger.error(f"Batch search failed: {str(e)}")
            return [[] for _ in queries]
    
    def delete(self, ids: List[str]) -> bool:
        """
        Delete vectors by ID.
        
        Args:
            ids: List of document IDs to delete
            
        Returns:
            Success status
        """
        if not self.initialized:
            logger.error("Vector Store not initialized")
            return False
        
        try:
            with self.lock:
                # Find indices to remove
                indices_to_remove = []
                for idx, doc_id in self.chunk_to_id.items():
                    if doc_id in ids:
                        indices_to_remove.append(idx)
                
                if not indices_to_remove:
                    logger.warning(f"No vectors found for IDs: {ids}")
                    return True
                
                # We need to rebuild the index without these vectors
                # as FAISS doesn't support direct removal
                
                # Create a mask of vectors to keep
                keep_mask = np.ones(self.index.ntotal, dtype=bool)
                for idx in indices_to_remove:
                    keep_mask[idx] = False
                
                # Extract vectors to keep
                all_vectors = faiss.extract_index_vectors(self.index)[1]
                kept_vectors = all_vectors[keep_mask]
                
                # Create new index
                if self.normalize:
                    new_index = faiss.IndexFlatIP(self.dimension)
                else:
                    new_index = faiss.IndexFlatL2(self.dimension)
                
                # Add kept vectors
                if len(kept_vectors) > 0:
                    new_index.add(kept_vectors)
                
                # Update mapping
                new_chunk_to_id = {}
                idx_counter = 0
                for old_idx, doc_id in self.chunk_to_id.items():
                    if old_idx not in indices_to_remove:
                        new_chunk_to_id[idx_counter] = doc_id
                        idx_counter += 1
                
                # Remove metadata
                for doc_id in ids:
                    if doc_id in self.id_to_metadata:
                        del self.id_to_metadata[doc_id]
                
                # Update instance variables
                self.index = new_index
                self.chunk_to_id = new_chunk_to_id
                
                # Save updated index and mapping
                self._save()
                
                logger.info(f"Deleted {len(indices_to_remove)} vectors for {len(ids)} document IDs")
                return True
        except Exception as e:
            logger.error(f"Failed to delete vectors: {str(e)}")
            return False
    
    def _save(self) -> bool:
        """
        Save index and mappings to disk.
        
        Returns:
            Success status
        """
        try:
            # Save index using a temporary file to avoid corruption
            with tempfile.NamedTemporaryFile(delete=False) as temp_index:
                temp_index_path = temp_index.name
                faiss.write_index(self.index, temp_index_path)
                os.replace(temp_index_path, self.index_path)
            
            # Save mapping
            with tempfile.NamedTemporaryFile(delete=False) as temp_mapping:
                temp_mapping_path = temp_mapping.name
                with open(temp_mapping_path, 'wb') as f:
                    pickle.dump(self.chunk_to_id, f)
                os.replace(temp_mapping_path, self.mapping_path)
            
            # Save metadata
            metadata_path = os.path.join(os.path.dirname(self.index_path), "id_metadata.json")
            with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') as temp_metadata:
                temp_metadata_path = temp_metadata.name
                json.dump(self.id_to_metadata, temp_metadata)
                os.replace(temp_metadata_path, metadata_path)
            
            return True
        except Exception as e:
            logger.error(f"Failed to save index: {str(e)}")
            return False
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding as NumPy array
        """
        if not self.initialized:
            logger.error("Vector Store not initialized")
            raise RuntimeError("Vector Store not initialized")
        
        try:
            return self._embed_texts([text])[0]
        except Exception as e:
            logger.error(f"Failed to get embedding: {str(e)}")
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get status of the Vector Store.
        
        Returns:
            Status dictionary
        """
        if not self.initialized:
            return {
                "status": "not_initialized"
            }
        
        return {
            "status": "initialized",
            "vectors": self.index.ntotal if self.index else 0,
            "dimension": self.dimension,
            "model": self.model_name,
            "normalize": self.normalize,
            "gpu_enabled": self.use_gpu and torch.cuda.is_available()
        }