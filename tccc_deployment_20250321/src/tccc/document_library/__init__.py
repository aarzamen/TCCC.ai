"""
Document Library module for TCCC.ai system.

This module provides document storage, retrieval, and semantic search capabilities
using Nexa AI's all-MiniLM-L12-v2 embedding model.
"""

# Import the DocumentLibrary class from the implementation file
from tccc.document_library.document_library import DocumentLibrary
from tccc.document_library.document_processor import DocumentProcessor
from tccc.document_library.cache_manager import CacheManager
from tccc.document_library.vector_store import VectorStore
from tccc.document_library.query_engine import QueryEngine, QueryStrategy
from tccc.document_library.response_generator import ResponseGenerator
from tccc.document_library.medical_vocabulary import MedicalVocabularyManager

__all__ = [
    "DocumentLibrary", 
    "DocumentProcessor", 
    "CacheManager",
    "VectorStore",
    "QueryEngine",
    "QueryStrategy",
    "ResponseGenerator",
    "MedicalVocabularyManager"
]