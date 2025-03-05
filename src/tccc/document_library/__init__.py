"""
Document Library module for TCCC.ai system.

This module provides document storage, retrieval, and semantic search capabilities
using Nexa AI's all-MiniLM-L12-v2 embedding model.
"""

import os
import json
import time
import logging
import hashlib
import pickle
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class DocumentLibrary:
    """
    Document Library implementation using Nexa AI's all-MiniLM-L12-v2 embedding model.
    
    This implementation provides:
    - Document storage and retrieval
    - Semantic search using FAISS vector database
    - Document chunking and indexing
    - Query result caching
    """
    
    def __init__(self):
        """Initialize the document library."""
        self.initialized = False
        self.documents = {}
        self.chunks = {}
        self.next_doc_id = 1
        self.config = None
        self.model = None
        self.index = None
        self.query_cache = {}
        self.chunk_to_doc = {}
    
    def initialize(self, config):
        """Initialize the document library with configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self.config = config
            
            # Create required directories
            os.makedirs(config["storage"]["base_dir"], exist_ok=True)
            os.makedirs(config["storage"]["index_path"], exist_ok=True)
            os.makedirs(config["storage"]["cache_dir"], exist_ok=True)
            os.makedirs(config["embedding"]["cache_dir"], exist_ok=True)
            
            # Load embedding model
            logger.info(f"Loading embedding model: {config['embedding']['model_name']}")
            self.model = SentenceTransformer(
                config["embedding"]["model_name"],
                cache_folder=config["embedding"]["cache_dir"]
            )
            
            # Initialize FAISS index
            self._initialize_index()
            
            # Load existing documents if available
            self._load_documents()
            
            self.initialized = True
            logger.info("Document Library initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize DocumentLibrary: {str(e)}")
            return False
    
    def _initialize_index(self):
        """Initialize the FAISS vector index."""
        try:
            index_path = os.path.join(self.config["storage"]["index_path"], "document_index.faiss")
            mapping_path = os.path.join(self.config["storage"]["index_path"], "chunk_mapping.pkl")
            
            # Check if we have an existing index
            if os.path.exists(index_path) and os.path.exists(mapping_path):
                logger.info("Loading existing FAISS index")
                self.index = faiss.read_index(index_path)
                with open(mapping_path, 'rb') as f:
                    self.chunk_to_doc = pickle.load(f)
            else:
                # Create a new index
                logger.info("Creating new FAISS index")
                dimension = self.config["embedding"]["dimension"]
                self.index = faiss.IndexFlatIP(dimension)  # Inner product index (for normalized vectors)
                
        except Exception as e:
            logger.error(f"Failed to initialize index: {str(e)}")
            raise
    
    def _load_documents(self):
        """Load existing documents from storage."""
        try:
            docs_path = os.path.join(self.config["storage"]["index_path"], "documents.json")
            chunks_path = os.path.join(self.config["storage"]["index_path"], "chunks.json")
            
            if os.path.exists(docs_path):
                with open(docs_path, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
                    # Find the highest document ID
                    if self.documents:
                        doc_ids = [int(doc_id) for doc_id in self.documents.keys()]
                        self.next_doc_id = max(doc_ids) + 1
                    logger.info(f"Loaded {len(self.documents)} documents from storage")
            
            if os.path.exists(chunks_path):
                with open(chunks_path, 'r', encoding='utf-8') as f:
                    self.chunks = json.load(f)
                    logger.info(f"Loaded {len(self.chunks)} document chunks from storage")
            
        except Exception as e:
            logger.error(f"Failed to load documents: {str(e)}")
            # Create empty collections if loading fails
            self.documents = {}
            self.chunks = {}
    
    def _save_documents(self):
        """Save documents to storage."""
        try:
            docs_path = os.path.join(self.config["storage"]["index_path"], "documents.json")
            chunks_path = os.path.join(self.config["storage"]["index_path"], "chunks.json")
            
            with open(docs_path, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, indent=2)
            
            with open(chunks_path, 'w', encoding='utf-8') as f:
                json.dump(self.chunks, f, indent=2)
            
            # Save FAISS index
            index_path = os.path.join(self.config["storage"]["index_path"], "document_index.faiss")
            mapping_path = os.path.join(self.config["storage"]["index_path"], "chunk_mapping.pkl")
            
            faiss.write_index(self.index, index_path)
            with open(mapping_path, 'wb') as f:
                pickle.dump(self.chunk_to_doc, f)
                
            logger.info("Document Library state saved")
            
        except Exception as e:
            logger.error(f"Failed to save documents: {str(e)}")
    
    def _chunk_document(self, doc_id: str, file_path: str) -> List[Dict[str, Any]]:
        """Split document into chunks for embedding.
        
        Args:
            doc_id: Document ID
            file_path: Path to the document file
            
        Returns:
            List of document chunks
        """
        try:
            # Read document content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Get chunk size and overlap from config
            chunk_size = self.config["indexing"]["chunk_size"]
            chunk_overlap = self.config["indexing"]["chunk_overlap"]
            
            # Split into chunks with overlap
            chunks = []
            for i in range(0, len(content), chunk_size - chunk_overlap):
                chunk_content = content[i:i + chunk_size]
                if len(chunk_content.strip()) > 0:
                    chunk_id = f"{doc_id}_{len(chunks)}"
                    chunks.append({
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "text": chunk_content,
                        "start_char": i,
                        "end_char": min(i + chunk_size, len(content)),
                    })
                    self.chunk_to_doc[chunk_id] = doc_id
            
            logger.debug(f"Created {len(chunks)} chunks for document {doc_id}")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to chunk document {doc_id}: {str(e)}")
            return []
    
    def _index_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        """Embed and index document chunks.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            True if indexing was successful
        """
        try:
            if not chunks:
                return True
            
            # Extract text from chunks
            texts = [chunk["text"] for chunk in chunks]
            
            # Embed text chunks
            embeddings = self.model.encode(
                texts,
                batch_size=self.config["embedding"]["batch_size"],
                show_progress_bar=False,
                normalize_embeddings=self.config["embedding"]["normalize"]
            )
            
            # Add embeddings to FAISS index
            chunk_ids = [chunk["chunk_id"] for chunk in chunks]
            self.index.add(np.array(embeddings).astype('float32'))
            
            # Store chunks in memory and on disk
            for i, chunk in enumerate(chunks):
                self.chunks[chunk["chunk_id"]] = chunk
            
            logger.debug(f"Indexed {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to index chunks: {str(e)}")
            return False
    
    def add_document(self, document_data):
        """Add a document to the library.
        
        Args:
            document_data: Document metadata and content
            
        Returns:
            Document ID if successful, None otherwise
        """
        try:
            if not self.initialized:
                logger.error("Document Library not initialized")
                return None
            
            file_path = document_data.get("file_path")
            if not file_path or not os.path.exists(file_path):
                logger.error(f"Invalid file path: {file_path}")
                return None
            
            # Create document entry
            doc_id = str(self.next_doc_id)
            self.next_doc_id += 1
            
            self.documents[doc_id] = {
                "id": doc_id,
                "metadata": document_data.get("metadata", {}),
                "file_path": file_path,
                "added_at": datetime.now().isoformat()
            }
            
            # Chunk and index document
            chunks = self._chunk_document(doc_id, file_path)
            success = self._index_chunks(chunks)
            
            # Add file name to metadata for display
            file_name = os.path.basename(file_path)
            if "metadata" in self.documents[doc_id]:
                self.documents[doc_id]["metadata"]["file_name"] = file_name
            
            # Save document library state
            self._save_documents()
            
            if success:
                logger.info(f"Document added with ID: {doc_id}")
                return doc_id
            else:
                logger.error(f"Failed to index document: {doc_id}")
                # Remove document if indexing failed
                del self.documents[doc_id]
                return None
                
        except Exception as e:
            logger.error(f"Failed to add document: {str(e)}")
            return None
    
    def query(self, query_text, n_results=3):
        """Query the document library for relevant documents.
        
        Args:
            query_text: Query text
            n_results: Number of results to return
            
        Returns:
            Dictionary with query results
        """
        try:
            if not self.initialized:
                logger.error("Document Library not initialized")
                return {"error": "Document Library not initialized"}
            
            # Check cache
            cache_key = self._generate_cache_key(query_text, n_results)
            cache_result = self._check_cache(cache_key)
            if cache_result:
                return cache_result
            
            start_time = time.time()
            
            # Embed query
            query_embedding = self.model.encode(
                query_text,
                normalize_embeddings=self.config["embedding"]["normalize"]
            )
            
            # Query FAISS index
            if self.index.ntotal == 0:
                logger.warning("Index is empty, no results available")
                return {
                    "query": query_text,
                    "results": [],
                    "total_results": 0,
                    "processing_time": time.time() - start_time,
                    "cache_hit": False
                }
            
            k = min(n_results * 3, self.index.ntotal)  # Get more results than needed to filter by unique documents
            scores, indices = self.index.search(
                np.array([query_embedding]).astype('float32'), 
                k
            )
            
            # Process results
            results = []
            seen_docs = set()
            
            for i in range(len(indices[0])):
                idx = indices[0][i]
                score = float(scores[0][i])
                
                if idx >= len(self.chunks):
                    continue
                
                # Get chunk and document info
                chunk_id = list(self.chunks.keys())[idx]
                chunk = self.chunks[chunk_id]
                doc_id = chunk["doc_id"]
                
                # Skip documents we've already seen
                if doc_id in seen_docs:
                    continue
                
                seen_docs.add(doc_id)
                
                # Get document metadata
                doc = self.documents.get(doc_id, {})
                metadata = doc.get("metadata", {})
                
                results.append({
                    "document_id": doc_id,
                    "text": chunk["text"],
                    "score": score,
                    "metadata": metadata
                })
                
                # Stop once we have enough unique document results
                if len(results) >= n_results:
                    break
            
            # Build response
            response = {
                "query": query_text,
                "results": results,
                "total_results": len(results),
                "processing_time": time.time() - start_time,
                "cache_hit": False
            }
            
            # Cache result
            self._cache_result(cache_key, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            return {
                "query": query_text,
                "error": str(e),
                "results": []
            }
    
    def _generate_cache_key(self, query_text, n_results):
        """Generate a cache key for a query.
        
        Args:
            query_text: Query text
            n_results: Number of results requested
            
        Returns:
            Cache key string
        """
        query_hash = hashlib.md5(query_text.encode()).hexdigest()
        return f"{query_hash}_{n_results}"
    
    def _check_cache(self, cache_key):
        """Check if a query result is in cache.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cached result or None
        """
        if cache_key in self.query_cache:
            cached = self.query_cache[cache_key]
            cache_age = time.time() - cached["timestamp"]
            
            # Check if cache is still valid
            if cache_age < self.config["search"]["cache_timeout"]:
                result = cached["result"]
                result["cache_hit"] = True
                return result
            
            # Remove expired cache entry
            del self.query_cache[cache_key]
        
        # Try to load from disk cache
        cache_file = os.path.join(
            self.config["storage"]["cache_dir"], 
            f"{cache_key}.json"
        )
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached = json.load(f)
                
                cache_age = time.time() - cached["timestamp"]
                if cache_age < self.config["search"]["cache_timeout"]:
                    result = cached["result"]
                    result["cache_hit"] = True
                    
                    # Add to memory cache
                    self.query_cache[cache_key] = {
                        "result": result,
                        "timestamp": cached["timestamp"]
                    }
                    
                    return result
                else:
                    # Remove expired cache file
                    os.remove(cache_file)
            except Exception as e:
                logger.error(f"Failed to read cache file: {str(e)}")
        
        return None
    
    def _cache_result(self, cache_key, result):
        """Cache a query result.
        
        Args:
            cache_key: Cache key
            result: Query result
        """
        try:
            # Add to memory cache
            self.query_cache[cache_key] = {
                "result": result,
                "timestamp": time.time()
            }
            
            # Write to disk cache
            cache_file = os.path.join(
                self.config["storage"]["cache_dir"], 
                f"{cache_key}.json"
            )
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "result": result,
                    "timestamp": time.time()
                }, f)
                
        except Exception as e:
            logger.error(f"Failed to cache result: {str(e)}")
    
    def get_document_metadata(self, doc_id):
        """Get metadata for a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document metadata or None
        """
        try:
            if not self.initialized:
                logger.error("Document Library not initialized")
                return None
            
            if doc_id not in self.documents:
                logger.error(f"Document not found: {doc_id}")
                return None
            
            return self.documents[doc_id]
            
        except Exception as e:
            logger.error(f"Failed to get document metadata: {str(e)}")
            return None
    
    def get_status(self):
        """Get the status of the document library.
        
        Returns:
            Dictionary with status information
        """
        try:
            if not self.initialized:
                return {
                    "status": "not_initialized"
                }
            
            vector_count = 0
            if self.index:
                vector_count = self.index.ntotal
                
            cache_info = {
                "memory_entries": len(self.query_cache),
                "disk_entries": len(os.listdir(self.config["storage"]["cache_dir"]))
            }
            
            return {
                "status": "initialized",
                "documents": {
                    "count": len(self.documents),
                    "chunks": len(self.chunks)
                },
                "index": {
                    "vectors": vector_count,
                    "dimension": self.config["embedding"]["dimension"]
                },
                "model": {
                    "name": self.config["embedding"]["model_name"]
                },
                "cache": cache_info
            }
            
        except Exception as e:
            logger.error(f"Failed to get status: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }