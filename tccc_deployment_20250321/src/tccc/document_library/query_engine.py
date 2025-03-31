"""
Query Engine for the TCCC.ai Document Library.

This module provides advanced query processing for the Document Library,
including query expansion, multi-strategy retrieval, and result ranking.
"""

import re
import time
import hashlib
import logging
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Union, Set

import numpy as np

from tccc.utils.logging import get_logger
from tccc.document_library.vector_store import VectorStore
from tccc.document_library.cache_manager import CacheManager
from tccc.document_library.medical_vocabulary import MedicalVocabularyManager

logger = get_logger(__name__)

class QueryStrategy(Enum):
    """Query strategy enumeration."""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    EXPANDED = "expanded"

class QueryEngine:
    """
    Query Engine for Document Library.
    
    This class provides:
    - Advanced query processing
    - Multiple retrieval strategies
    - Query expansion with medical terminology
    - Hybrid search combining semantic and keyword approaches
    - Result deduplication and ranking
    """
    
    def __init__(self, config: Dict[str, Any], vector_store: VectorStore, cache_manager: Optional[CacheManager] = None):
        """
        Initialize the Query Engine.
        
        Args:
            config: Configuration dictionary
            vector_store: Vector Store instance
            cache_manager: Optional Cache Manager instance
        """
        self.config = config
        self.vector_store = vector_store
        self.cache_manager = cache_manager
        self.min_similarity = config["search"].get("min_similarity", 0.7)
        self.default_limit = config["search"].get("default_results", 5)
        self.max_limit = config["search"].get("max_results", 20)
        self.fuzzy_matching = config["search"].get("fuzzy_matching", True)
        
        # Load medical vocabulary if available
        try:
            self.medical_vocabulary = MedicalVocabularyManager(config)
            self.medical_vocabulary.initialize()
            self.has_medical_vocabulary = True
        except:
            logger.warning("Medical vocabulary not available")
            self.has_medical_vocabulary = False
            self.medical_vocabulary = None
    
    def query(self, 
              query_text: str, 
              strategy: Union[str, QueryStrategy] = QueryStrategy.HYBRID,
              limit: Optional[int] = None, 
              min_similarity: Optional[float] = None,
              filter_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a query against the document library.
        
        Args:
            query_text: Query text
            strategy: Query strategy
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold
            filter_metadata: Metadata filters
            
        Returns:
            Query result dictionary
        """
        start_time = time.time()
        
        # Convert string strategy to enum
        if isinstance(strategy, str):
            try:
                strategy = QueryStrategy(strategy.lower())
            except:
                logger.warning(f"Invalid query strategy: {strategy}, using HYBRID")
                strategy = QueryStrategy.HYBRID
        
        # Validate parameters
        limit = min(self.max_limit, limit or self.default_limit)
        min_similarity = min_similarity or self.min_similarity
        
        # Generate cache key if cache manager available
        cache_key = None
        if self.cache_manager:
            cache_params = {
                "strategy": strategy.value,
                "limit": limit,
                "min_similarity": min_similarity
            }
            if filter_metadata:
                cache_params["filter"] = str(sorted(filter_metadata.items()))
            
            cache_key = self.cache_manager.generate_key(query_text, cache_params)
            cached_result = self.cache_manager.get(cache_key)
            if cached_result:
                logger.debug(f"Cache hit for query: {query_text}")
                return cached_result
        
        # Process query based on strategy
        if strategy == QueryStrategy.SEMANTIC:
            # Pure semantic search
            results = self._semantic_search(query_text, limit, min_similarity, filter_metadata)
        elif strategy == QueryStrategy.KEYWORD:
            # Pure keyword search
            results = self._keyword_search(query_text, limit, filter_metadata)
        elif strategy == QueryStrategy.EXPANDED:
            # Expanded semantic search with medical terminology
            results = self._expanded_search(query_text, limit, min_similarity, filter_metadata)
        else:
            # Default to hybrid search (semantic + keyword)
            results = self._hybrid_search(query_text, limit, min_similarity, filter_metadata)
        
        # Build response
        processing_time = time.time() - start_time
        response = {
            "query": query_text,
            "strategy": strategy.value,
            "results": results,
            "total_results": len(results),
            "processing_time": processing_time,
            "cache_hit": False
        }
        
        # Cache result if cache manager available
        if self.cache_manager and cache_key:
            self.cache_manager.set(cache_key, response)
        
        return response
    
    def _semantic_search(self, 
                        query_text: str,
                        limit: int,
                        min_similarity: float,
                        filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Perform semantic search using the vector store.
        
        Args:
            query_text: Query text
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold
            filter_metadata: Metadata filters
            
        Returns:
            List of results
        """
        # Get results from vector store
        vector_results = self.vector_store.search(
            query_text, 
            k=limit * 2,  # Request more results to account for filtering
            threshold=min_similarity
        )
        
        # Apply metadata filtering if specified
        if filter_metadata:
            vector_results = self._filter_by_metadata(vector_results, filter_metadata)
        
        # Limit results
        vector_results = vector_results[:limit]
        
        return vector_results
    
    def _keyword_search(self, 
                       query_text: str,
                       limit: int,
                       filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Perform keyword-based search.
        
        Args:
            query_text: Query text
            limit: Maximum number of results
            filter_metadata: Metadata filters
            
        Returns:
            List of results
        """
        # This is a simplified keyword search implementation
        # A more sophisticated approach would use an inverted index
        
        # Extract keywords from query
        keywords = self._extract_keywords(query_text)
        
        # Get all vector IDs and metadata
        all_metadata = self.vector_store.id_to_metadata
        
        # Score each document based on keyword matches
        scores = {}
        for doc_id, metadata in all_metadata.items():
            # Skip if doesn't match metadata filter
            if filter_metadata and not self._matches_metadata_filter(metadata, filter_metadata):
                continue
                
            # Get text if available in metadata
            if "text" in metadata:
                doc_text = metadata["text"]
            else:
                # Skip documents without text
                continue
                
            # Calculate score based on keyword matches
            score = 0
            for keyword in keywords:
                pattern = r'\b' + re.escape(keyword) + r'\b'
                matches = re.findall(pattern, doc_text, re.IGNORECASE)
                score += len(matches)
            
            if score > 0:
                scores[doc_id] = score
        
        # Sort documents by score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Build results
        results = []
        for doc_id, score in sorted_docs[:limit]:
            metadata = all_metadata.get(doc_id, {})
            results.append({
                "id": doc_id,
                "score": score / 10.0,  # Normalize score
                "metadata": metadata
            })
        
        return results
    
    def _expanded_search(self, 
                        query_text: str,
                        limit: int,
                        min_similarity: float,
                        filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Perform expanded search using medical terminology.
        
        Args:
            query_text: Query text
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold
            filter_metadata: Metadata filters
            
        Returns:
            List of results
        """
        # Check if medical vocabulary is available
        if not self.has_medical_vocabulary:
            logger.warning("Medical vocabulary not available for expanded search, falling back to semantic search")
            return self._semantic_search(query_text, limit, min_similarity, filter_metadata)
        
        # Generate expanded queries
        expanded_queries = self.medical_vocabulary.expand_query(query_text)
        if not expanded_queries:
            expanded_queries = [query_text]
        
        # Perform batch search
        all_results = self.vector_store.batch_search(
            expanded_queries,
            k=limit,
            threshold=min_similarity
        )
        
        # Combine and deduplicate results
        results = []
        seen_ids = set()
        
        for query_results in all_results:
            for result in query_results:
                doc_id = result["id"]
                if doc_id not in seen_ids:
                    # Apply metadata filtering if specified
                    if filter_metadata and not self._matches_metadata_filter(result["metadata"], filter_metadata):
                        continue
                    
                    seen_ids.add(doc_id)
                    results.append(result)
        
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Limit results
        return results[:limit]
    
    def _hybrid_search(self, 
                      query_text: str,
                      limit: int,
                      min_similarity: float,
                      filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic and keyword approaches.
        
        Args:
            query_text: Query text
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold
            filter_metadata: Metadata filters
            
        Returns:
            List of results
        """
        # Get semantic search results
        semantic_results = self._semantic_search(
            query_text, 
            limit=limit, 
            min_similarity=min_similarity,
            filter_metadata=filter_metadata
        )
        
        # Get keyword search results
        keyword_results = self._keyword_search(
            query_text, 
            limit=limit,
            filter_metadata=filter_metadata
        )
        
        # Combine and deduplicate results
        results = []
        seen_ids = set()
        
        # Add semantic results first (they're usually higher quality)
        for result in semantic_results:
            seen_ids.add(result["id"])
            results.append(result)
        
        # Add keyword results if not already included
        for result in keyword_results:
            if result["id"] not in seen_ids:
                seen_ids.add(result["id"])
                results.append(result)
        
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Limit results
        return results[:limit]
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from text.
        
        Args:
            text: Input text
            
        Returns:
            List of keywords
        """
        # Remove punctuation and convert to lowercase
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        
        # Tokenize and filter words
        tokens = [word for word in text.split() if len(word) > 2]
        
        # Remove duplicates but preserve order
        keywords = []
        for token in tokens:
            if token not in keywords:
                keywords.append(token)
        
        return keywords
    
    def _filter_by_metadata(self, 
                           results: List[Dict[str, Any]], 
                           filter_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Filter results by metadata.
        
        Args:
            results: List of results
            filter_metadata: Metadata filters
            
        Returns:
            Filtered list of results
        """
        filtered_results = []
        
        for result in results:
            metadata = result.get("metadata", {})
            
            # Skip if doesn't match filter
            if not self._matches_metadata_filter(metadata, filter_metadata):
                continue
                
            filtered_results.append(result)
        
        return filtered_results
    
    def _matches_metadata_filter(self, 
                                metadata: Dict[str, Any], 
                                filter_metadata: Dict[str, Any]) -> bool:
        """
        Check if metadata matches filter.
        
        Args:
            metadata: Document metadata
            filter_metadata: Metadata filters
            
        Returns:
            True if matches, False otherwise
        """
        for key, filter_value in filter_metadata.items():
            # Skip if key not in metadata
            if key not in metadata:
                return False
                
            value = metadata[key]
            
            # Handle different value types
            if isinstance(filter_value, list):
                # Check if any value in the list matches
                if not isinstance(value, list) and value not in filter_value:
                    return False
                elif isinstance(value, list) and not any(v in filter_value for v in value):
                    return False
            elif isinstance(filter_value, str):
                # Case-insensitive string comparison
                if isinstance(value, str) and filter_value.lower() != value.lower():
                    return False
                elif not isinstance(value, str) and str(filter_value) != str(value):
                    return False
            else:
                # Direct comparison for other types
                if filter_value != value:
                    return False
        
        return True