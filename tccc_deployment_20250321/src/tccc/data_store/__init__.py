"""
Data Store module for TCCC.ai system.

This module handles persistent storage of conversations, analysis results, and metrics.
It provides a robust SQLite-based storage system with indexing, backup/restore capabilities,
and context generation for LLM analysis.
"""

from .data_store import DataStore, DatabaseManager, BackupManager

__all__ = ['DataStore', 'DatabaseManager', 'BackupManager']