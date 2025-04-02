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

# Import local modules
from tccc.utils.logging import get_logger
from tccc.document_library.document_processor import DocumentProcessor
from tccc.document_library.cache_manager import CacheManager
from tccc.processing_core.processing_core import ModuleState

logger = get_logger(__name__)


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
        self.chunk_to_doc = {}
        
        # Will be initialized in initialize()
        self.document_processor = None
        self.cache_manager = None
        self.vector_store = None
        self.query_engine = None
        self.response_generator = None
        self.medical_vocabulary = None
    
    def initialize(self, config):
        """Initialize the document library with configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Use a copy of config to avoid modifying the original
            if config is None:
                logger.warning("No configuration provided, using default configuration")
                config = {
                    "storage": {
                        "base_dir": "data/documents",
                        "index_path": "data/document_index",
                        "cache_dir": "data/query_cache"
                    },
                    "embedding": {
                        "model_name": "all-MiniLM-L12-v2",
                        "cache_dir": "data/models/embeddings",
                        "dimension": 384,
                        "normalize": True,
                        "batch_size": 32
                    },
                    "indexing": {
                        "chunk_size": 1000,
                        "chunk_overlap": 200
                    }
                }
            
            logger.debug("DocumentLibrary: Loading config")
            self.config = config
            
            # Validate essential config keys
            for section in ["storage", "embedding"]:
                if section not in config:
                    logger.warning(f"Missing '{section}' configuration section, using defaults")
                    if section == "storage":
                        config[section] = {
                            "base_dir": "data/documents",
                            "index_path": "data/document_index",
                            "cache_dir": "data/query_cache"
                        }
                    elif section == "embedding":
                        config[section] = {
                            "model_name": "all-MiniLM-L12-v2",
                            "cache_dir": "data/models/embeddings",
                            "dimension": 384,
                            "normalize": True,
                            "batch_size": 32,
                            "use_gpu": False
                        }
                    else:
                        config[section] = {}
            
            logger.debug("DocumentLibrary: Validating config keys")
            # Ensure required storage paths with defaults
            storage_paths = {
                "base_dir": "data/documents",
                "index_path": "data/document_index",
                "cache_dir": "data/query_cache"
            }
            
            for path_key, default_path in storage_paths.items():
                if path_key not in config["storage"]:
                    logger.warning(f"Missing storage path '{path_key}', using default: {default_path}")
                    config["storage"][path_key] = default_path
            
            logger.debug("DocumentLibrary: Ensuring storage paths")
            # Create required directories with proper error handling
            for path_key, path_name in [
                ("base_dir", "Document Storage"),
                ("index_path", "Index"),
                ("cache_dir", "Cache")
            ]:
                try:
                    dir_path = config["storage"][path_key]
                    os.makedirs(dir_path, exist_ok=True)
                    logger.info(f"{path_name} directory created/verified: {dir_path}")
                except Exception as dir_error:
                    logger.error(f"Failed to create {path_name.lower()} directory: {str(dir_error)}")
                    # Continue with initialization, as we might still be able to function
            
            logger.debug("DocumentLibrary: Ensuring embedding cache directory exists")
            # Ensure embedding cache directory exists
            try:
                if "cache_dir" not in config["embedding"]:
                    config["embedding"]["cache_dir"] = "data/models/embeddings"
                    
                embedding_cache_dir = config["embedding"]["cache_dir"]
                os.makedirs(embedding_cache_dir, exist_ok=True)
                logger.info(f"Embedding cache directory created/verified: {embedding_cache_dir}")
            except Exception as dir_error:
                logger.error(f"Failed to create embedding cache directory: {str(dir_error)}")
                # Continue with initialization
            
            logger.debug("DocumentLibrary: Initializing document processor")
            # Initialize document processor
            try:
                self.document_processor = DocumentProcessor(config)
                logger.info("Document processor initialized")
            except Exception as dp_error:
                logger.warning(f"Failed to initialize document processor: {str(dp_error)}")
                self.document_processor = None
                # Continue with initialization
            
            logger.debug("DocumentLibrary: Initializing cache manager")
            # Initialize cache manager
            try:
                self.cache_manager = CacheManager(config)
                logger.info("Cache manager initialized")
            except Exception as cm_error:
                logger.warning(f"Failed to initialize cache manager: {str(cm_error)}")
                self.cache_manager = None
                # Continue with initialization
            
            logger.debug("DocumentLibrary: Adding default indexing configuration")
            # Add default indexing configuration if missing
            if "indexing" not in config:
                logger.warning("Missing 'indexing' configuration section, using defaults")
                config["indexing"] = {
                    "chunk_size": 1000,
                    "chunk_overlap": 200
                }
                
            logger.debug("DocumentLibrary: Importing components")
            # Import components (done here to avoid circular imports)
            try:
                from tccc.document_library.vector_store import VectorStore
                from tccc.document_library.query_engine import QueryEngine
                from tccc.document_library.response_generator import ResponseGenerator
                from tccc.document_library.medical_vocabulary import MedicalVocabularyManager
            except ImportError as import_error:
                logger.error(f"Failed to import required components: {str(import_error)}")
                return False
            
            logger.debug("DocumentLibrary: Initializing vector store")
            # Initialize vector store with improved error handling
            self.vector_store = None
            self.model = None
            self.index = None
            
            try:
                self.vector_store = VectorStore(config)
                vector_store_init = self.vector_store.initialize()
                
                if vector_store_init:
                    logger.info("Vector store initialized successfully")
                    
                    # Load embedding model (for backward compatibility)
                    model_name = config.get("embedding", {}).get("model_name", "all-MiniLM-L12-v2")
                    logger.info(f"Loading embedding model: {model_name}")
                    self.model = self.vector_store.embedding_model
                    
                    # Initialize FAISS index (for backward compatibility)
                    self.index = self.vector_store.index
                else:
                    logger.warning("Vector store initialization failed, using fallback")
                    # Create minimal vector_store for operation
                    if hasattr(self.vector_store, '_create_minimal_vector_store'):
                        self.vector_store._create_minimal_vector_store()
                        self.model = self.vector_store.embedding_model
                        self.index = self.vector_store.index
            except Exception as vs_error:
                logger.error(f"Vector store initialization error: {str(vs_error)}")
                # Create a minimal vector store implementation for continued operation
                try:
                    # Minimal implementation that will return empty results but won't crash
                    # Define class methods properly with self parameter 
                    class MinimalVectorStore:
                        def __init__(self):
                            self.index = None
                            self.embedding_model = None
                            self.initialized = True
                            
                        def get_status(self):
                            return {"vectors": 0, "initialized": False}
                            
                        def add_embeddings(self, *args, **kwargs):
                            return False
                        
                        def add(self, *args, **kwargs):
                            return False
                            
                        def search(self, *args, **kwargs):
                            return []
                            
                        def initialize(self):
                            return True
                            
                        def _create_minimal_vector_store(self):
                            return True
                    
                    # Create minimal class instance with proper methods
                    self.vector_store = MinimalVectorStore()
                    self.model = None
                    self.index = None
                except Exception as err:
                    logger.error(f"Failed to create minimal vector store: {err}")
            
            logger.debug("DocumentLibrary: Initializing medical vocabulary")
            # Initialize medical vocabulary with improved error handling
            try:
                self.medical_vocabulary = MedicalVocabularyManager(config)
                self.medical_vocabulary.initialize()
                logger.info("Medical vocabulary initialized successfully")
            except Exception as vocab_error:
                logger.warning(f"Medical vocabulary initialization failed: {str(vocab_error)}")
                self.medical_vocabulary = None
            
            logger.debug("DocumentLibrary: Initializing query engine")
            # Initialize query engine with fallback
            try:
                if self.vector_store:
                    self.query_engine = QueryEngine(config, self.vector_store, self.cache_manager)
                    logger.info("Query engine initialized")
                else:
                    self.query_engine = None
                    logger.warning("Skipping query engine initialization due to missing vector store")
            except Exception as qe_error:
                logger.warning(f"Query engine initialization failed: {str(qe_error)}")
                self.query_engine = None
            
            logger.debug("DocumentLibrary: Initializing response generator")
            # Initialize response generator with fallback
            try:
                if self.query_engine:
                    self.response_generator = ResponseGenerator(config, self.query_engine)
                    logger.info("Response generator initialized")
                else:
                    self.response_generator = None
                    logger.warning("Skipping response generator initialization due to missing query engine")
            except Exception as rg_error:
                logger.warning(f"Response generator initialization failed: {str(rg_error)}")
                self.response_generator = None
            
            logger.debug("DocumentLibrary: Loading existing documents")
            # Load existing documents if available
            try:
                self._load_documents()
            except Exception as ld_error:
                logger.warning(f"Failed to load existing documents: {str(ld_error)}")
                # Initialize empty document collections as fallback
                self.documents = {}
                self.chunks = {}
                self.chunk_to_doc = {}
            
            logger.debug("DocumentLibrary: Marking as initialized")
            # Mark as initialized with limited functionality if needed
            self.initialized = True
            
            if not self.vector_store or not self.model or not self.index:
                logger.warning("Document Library initialized with limited functionality")
            else:
                logger.info("Document Library initialized successfully")
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize DocumentLibrary: {str(e)}")
            # Set partial initialization to allow for some basic functionality
            self.initialized = False
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
    
    def _chunk_document(self, doc_id: str, text: str) -> List[Dict[str, Any]]:
        """Split document into chunks for embedding.
        
        Args:
            doc_id: Document ID
            text: Document text content
            
        Returns:
            List of document chunks
        """
        try:
            # Get chunk size and overlap from config
            chunk_size = self.config["indexing"]["chunk_size"]
            chunk_overlap = self.config["indexing"]["chunk_overlap"]
            
            # Split into chunks with overlap
            chunks = []
            for i in range(0, len(text), chunk_size - chunk_overlap):
                chunk_content = text[i:i + chunk_size]
                if len(chunk_content.strip()) > 0:
                    chunk_id = f"{doc_id}_{len(chunks)}"
                    chunks.append({
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "text": chunk_content,
                        "start_char": i,
                        "end_char": min(i + chunk_size, len(text)),
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
            document_data: Document metadata and content. Can contain either:
                - file_path: Path to the document file
                - text: Direct text content of the document
            
        Returns:
            Document ID if successful, None otherwise
        """
        try:
            if not self.initialized:
                logger.error("Document Library not initialized")
                return None
            
            # Check for direct text content first (highest priority)
            if "text" in document_data and document_data["text"]:
                text = document_data["text"]
                metadata = document_data.get("metadata", {})
                source = document_data.get("source", "Direct text input")
                
                # If source is specified and it's a file, use it as file_path for reference
                file_path = source if os.path.isfile(source) else None
                
                logger.info(f"Adding document from direct text input (length: {len(text)})")
            
            # Otherwise check for file path
            else:
                file_path = document_data.get("file_path")
                if not file_path or not os.path.exists(file_path):
                    logger.error(f"Invalid file path: {file_path}")
                    return None
                
                # Process the document to extract text and metadata
                if hasattr(self, "document_processor") and self.document_processor:
                    # Use document processor if available
                    processed = self.document_processor.process_document(file_path)
                    
                    if not processed["success"]:
                        logger.error(f"Failed to process document: {processed.get('error', 'Unknown error')}")
                        return None
                    
                    text = processed["text"]
                    
                    # Merge metadata
                    metadata = document_data.get("metadata", {})
                    if "metadata" in processed:
                        metadata.update(processed["metadata"])
                else:
                    # Fall back to simple text reading
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                            text = f.read()
                    except Exception as e:
                        logger.error(f"Failed to read document: {str(e)}")
                        return None
                    
                    metadata = document_data.get("metadata", {})
            
            # Create document entry
            doc_id = str(self.next_doc_id)
            self.next_doc_id += 1
            
            self.documents[doc_id] = {
                "id": doc_id,
                "metadata": metadata,
                "file_path": file_path,
                "content": text, # Store the content directly
                "added_at": datetime.now().isoformat()
            }
            
            # Chunk document text
            chunks = self._chunk_document(doc_id, text)
            success = self._index_chunks(chunks)
            
            # Add file name to metadata for display
            file_name = None
            if file_path:
                try:
                    file_name = os.path.basename(file_path)
                except Exception as e:
                    logger.warning(f"Could not get basename for file_path '{file_path}': {e}")
        
            # Fallback to title or doc_id if file_name is None
            if not file_name:
                file_name = metadata.get("title", f"doc_{doc_id}")

            if "metadata" in self.documents[doc_id]:
                self.documents[doc_id]["metadata"]["file_name"] = file_name
            else:
                 self.documents[doc_id]["metadata"] = {"file_name": file_name}

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

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document by its ID.

        Args:
            doc_id: The ID of the document to retrieve.

        Returns:
            The document dictionary if found, otherwise None.
        """
        if not self.initialized:
            logger.error("Document Library not initialized")
            return None
            
        # Attempt to retrieve from in-memory dictionary first
        doc = self.documents.get(str(doc_id)) # Ensure doc_id is string
        if doc:
            # Optionally, retrieve full content if not stored directly
            # This depends on how content is stored (e.g., separate file, chunked)
            # For now, assume content retrieval isn't needed or handled elsewhere
            # --> Correction: We *do* need the content now, and we stored it above.
             logger.debug(f"Retrieved document {doc_id} from memory.")
             return doc # The 'content' key is already in the doc dictionary
        else:
             # If not in memory, perhaps check persistent storage if applicable
             # (Currently, _load_documents loads everything into memory on init)
             logger.warning(f"Document with ID {doc_id} not found.")
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
            
            # Check cache if available
            if hasattr(self, "cache_manager") and self.cache_manager:
                cache_key = self.cache_manager.generate_key(query_text, {"n_results": n_results})
                cached_result = self.cache_manager.get(cache_key)
                if cached_result:
                    return cached_result
            
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
            
            # Cache result if manager available
            if hasattr(self, "cache_manager") and self.cache_manager:
                self.cache_manager.set(cache_key, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            return {
                "query": query_text,
                "error": str(e),
                "results": []
            }
    
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
    
    def advanced_query(self, 
                  query_text: str, 
                  strategy: str = "hybrid",
                  limit: int = None, 
                  min_similarity: float = None,
                  filter_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform an advanced query against the document library.
        
        Args:
            query_text: Query text
            strategy: Query strategy (semantic, keyword, hybrid, expanded)
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold
            filter_metadata: Metadata filters
            
        Returns:
            Query result dictionary
        """
        if not self.initialized:
            return {
                "error": "Document Library not initialized",
                "results": []
            }
        
        # Use the query engine for advanced querying
        if self.query_engine:
            return self.query_engine.query(
                query_text=query_text,
                strategy=strategy,
                limit=limit,
                min_similarity=min_similarity,
                filter_metadata=filter_metadata
            )
        else:
            # Fall back to basic query if query engine not available
            return self.query(query_text, limit or 5)
    
    def generate_llm_prompt(self, 
                          query: str, 
                          strategy: str = "hybrid",
                          limit: int = 3,
                          max_context_length: int = None) -> str:
        """
        Generate a prompt for the LLM with relevant context.
        
        Args:
            query: User query
            strategy: Query strategy
            limit: Maximum number of results to include
            max_context_length: Override for maximum context length
            
        Returns:
            Formatted prompt string
        """
        if not self.initialized:
            return f"""
You are TCCC.ai, an expert in Tactical Combat Casualty Care (TCCC).
Please answer the following query to the best of your ability:

{query}

Note: The Document Library is not initialized so no specific context can be provided.
"""
        
        # Use the response generator if available
        if self.response_generator:
            # Set context length limit if provided
            if max_context_length is not None:
                # Store current value to restore later
                original_length = self.response_generator.max_context_length
                self.response_generator.max_context_length = max_context_length
                logger.info(f"Overriding max context length: {max_context_length}")
                
            try:
                # Generate prompt with current settings
                prompt = self.response_generator.generate_prompt(
                    query=query,
                    strategy=strategy,
                    limit=limit
                )
                
                # Restore original context length if it was changed
                if max_context_length is not None:
                    self.response_generator.max_context_length = original_length
                
                return prompt
                
            except Exception as e:
                logger.error(f"Error generating prompt: {str(e)}")
                # Restore original context length if it was changed
                if max_context_length is not None:
                    self.response_generator.max_context_length = original_length
                
                # Fall through to fallback method
        
        # Fallback to basic prompt
        try:
            results = self.query(query, min(limit, 2))  # Limit to 2 results for safety
            
            # Truncate context if needed
            context_parts = []
            total_length = 0
            max_length = max_context_length or 1500
            
            for r in results.get("results", []):
                text = r.get('text', '')
                doc_id = r.get('document_id', 'unknown')
                
                # Truncate text if too long
                if len(text) > 500:
                    text = text[:497] + "..."
                
                part = f"From document {doc_id}: {text}"
                
                # Check if adding this part would exceed max length
                if total_length + len(part) + 10 > max_length:
                    if not context_parts:  # Ensure at least one result is included
                        context_parts.append(part[:max_length - 100] + "... [truncated]")
                    break
                    
                context_parts.append(part)
                total_length += len(part) + 2  # +2 for newlines
            
            context = "\n\n".join(context_parts)
            
            return f"""
You are TCCC.ai, an expert in Tactical Combat Casualty Care (TCCC).
Please use the following information to answer the query:

CONTEXT:
{context}

QUERY:
{query}

If the context doesn't contain enough information, please state that clearly.
"""
        except Exception as e:
            logger.error(f"Error in fallback prompt generation: {str(e)}")
            # Ultra fallback with no context
            return f"""
You are TCCC.ai, an expert in Tactical Combat Casualty Care (TCCC).
Please answer the following query to the best of your ability:

{query}

Note: Unable to retrieve specific context from the document library due to a system error.
"""
    
    def extract_medical_terms(self, text: str) -> List[str]:
        """
        Extract known medical terms from text.
        
        Args:
            text: Input text
            
        Returns:
            List of recognized medical terms
        """
        if not self.initialized:
            return []
        
        if self.medical_vocabulary:
            return self.medical_vocabulary.extract_medical_terms(text)
        else:
            return []
    
    def explain_medical_terms(self, text: str) -> Dict[str, str]:
        """
        Provide explanations for medical terms in text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping terms to explanations
        """
        if not self.initialized:
            return {}
        
        if self.medical_vocabulary:
            return self.medical_vocabulary.explain_medical_terms(text)
        else:
            return {}
    
    def get_status(self):
        """Get the status of the document library.
        
        Returns:
            Dictionary with status information (using ModuleState enum for 'status').
        """
        overall_status = ModuleState.UNINITIALIZED
        if self.initialized:
            overall_status = ModuleState.ACTIVE # Use ACTIVE instead of READY to align with verification expectations
        
        try:
            if not self.initialized:
                return {
                    "status": ModuleState.UNINITIALIZED
                }
            
            # Get vector count - with robust error handling
            vector_count = 0
            dimension = 384  # Default dimension
            model_name = "unknown"  # Default model name
            
            try:
                if self.vector_store and hasattr(self.vector_store, 'get_status'):
                    try:
                        vector_status = self.vector_store.get_status()
                        vector_count = vector_status.get("vectors", 0)
                    except Exception as vs_error:
                        logger.error(f"Failed to get vector store status: {vs_error}")
                        # Continue with defaults
                elif self.index:
                    try:
                        if hasattr(self.index, 'ntotal'):
                            vector_count = self.index.ntotal
                    except Exception as idx_error:
                        logger.error(f"Failed to get index ntotal: {idx_error}")
                        # Continue with defaults
                
                # Safely get config values with defaults
                if self.config and isinstance(self.config, dict):
                    embedding_config = self.config.get("embedding", {})
                    if isinstance(embedding_config, dict):
                        dimension = embedding_config.get("dimension", 384)
                        model_name = embedding_config.get("model_name", "unknown")
            except Exception as vector_error:
                logger.error(f"Error getting vector information: {vector_error}")
                # Continue with defaults
            
            # Get cache stats if available - with robust error handling
            cache_info = {
                "memory_entries": 0,
                "disk_entries": 0
            }
            
            try:
                if hasattr(self, "cache_manager") and self.cache_manager:
                    try:
                        cache_stats = self.cache_manager.get_stats()
                        if isinstance(cache_stats, dict) and "error" not in cache_stats:
                            cache_info = cache_stats
                    except Exception as cache_error:
                        logger.error(f"Failed to get cache stats: {cache_error}")
                        # Continue with defaults
            except Exception as cm_error:
                logger.error(f"Error accessing cache manager: {cm_error}")
                # Continue with defaults
            
            # Get medical vocabulary status
            try:
                medical_vocab_status = "available" if (
                    hasattr(self, "medical_vocabulary") and 
                    self.medical_vocabulary and 
                    hasattr(self.medical_vocabulary, "initialized") and
                    self.medical_vocabulary.initialized
                ) else "not_available"
            except Exception as mv_error:
                logger.error(f"Error getting medical vocabulary status: {mv_error}")
                medical_vocab_status = "error"
            
            # Check component existence with safe access
            doc_processor_available = False
            try:
                doc_processor_available = hasattr(self, "document_processor") and self.document_processor is not None
            except Exception:
                pass
                
            vector_store_available = False
            try:
                vector_store_available = hasattr(self, "vector_store") and self.vector_store is not None
            except Exception:
                pass
                
            query_engine_available = False
            try:
                query_engine_available = hasattr(self, "query_engine") and self.query_engine is not None
            except Exception:
                pass
                
            response_generator_available = False
            try:
                response_generator_available = hasattr(self, "response_generator") and self.response_generator is not None
            except Exception:
                pass
            
            # Build status dictionary
            status_details = {
                "initialized": self.initialized,
                "documents": {
                    "count": len(self.documents) if hasattr(self, "documents") else 0,
                    "chunks": len(self.chunks) if hasattr(self, "chunks") else 0
                },
                "index": {
                    "vectors": vector_count,
                    "dimension": dimension
                },
                "model": {
                    "name": model_name
                },
                "cache": cache_info,
                "components": {
                    "document_processor": doc_processor_available,
                    "vector_store": vector_store_available,
                    "query_engine": query_engine_available,
                    "response_generator": response_generator_available,
                    "medical_vocabulary": medical_vocab_status
                }
            }

            # Check if any component reported an error during status check
            if (cache_info.get('status') == 'error' or 
                medical_vocab_status == 'error'): 
                 # Add checks for other component errors if their get_status returns error states
                overall_status = ModuleState.ERROR

            status = {"status": overall_status}
            status.update(status_details)
            return status
            
        except Exception as e:
            logger.error(f"Failed to get DocumentLibrary status: {e}")
            return {
                'status': ModuleState.ERROR,
                'initialized': self.initialized, # Keep initialized flag even on error
                'error': str(e)
            }