"""
Cache Manager module for the Document Library.

This module provides caching functionality for document queries,
supporting both in-memory and disk-based caching with TTL and size limits.
"""

import os
import time
import json
import shutil
import hashlib
import threading
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

# Local imports
from tccc.utils.logging import get_logger

logger = get_logger(__name__)


class CacheManager:
    """
    Cache Manager for document queries.
    
    This class provides:
    - In-memory caching for fast access
    - Disk-based persistence for query results
    - TTL-based cache invalidation
    - Size-based cache limits
    - Thread-safe operations
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the cache manager.
        
        Args:
            config: Configuration dictionary with cache settings
        """
        self.config = config
        self.cache_dir = config["storage"]["cache_dir"]
        self.cache_timeout = config["search"]["cache_timeout"]
        self.max_memory_entries = 1000  # Default limit
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize in-memory cache
        self.memory_cache = {}
        self.cache_lock = threading.Lock()
        
        # Log initialization
        logger.info(f"Cache Manager initialized with cache directory: {self.cache_dir}")
        logger.info(f"Cache timeout: {self.cache_timeout} seconds")
        
        # Clean up expired cache entries on startup
        self._cleanup_expired_cache()
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get a cached result.
        
        Args:
            key: Cache key
            
        Returns:
            Cached result or None if not found
        """
        # First check memory cache
        with self.cache_lock:
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                
                # Check if entry is expired
                if time.time() - entry["timestamp"] < self.cache_timeout:
                    logger.debug(f"Memory cache hit for key: {key}")
                    
                    # Update result with cache hit flag
                    result = entry["result"]
                    result["cache_hit"] = True
                    return result
                else:
                    # Remove expired entry
                    del self.memory_cache[key]
        
        # If not in memory, check disk cache
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    entry = json.load(f)
                
                # Check if entry is expired
                if time.time() - entry["timestamp"] < self.cache_timeout:
                    logger.debug(f"Disk cache hit for key: {key}")
                    
                    # Add to memory cache
                    result = entry["result"]
                    result["cache_hit"] = True
                    
                    with self.cache_lock:
                        self.memory_cache[key] = {
                            "result": result,
                            "timestamp": entry["timestamp"]
                        }
                        
                        # Limit memory cache size
                        self._limit_memory_cache()
                    
                    return result
                else:
                    # Remove expired cache file
                    os.remove(cache_file)
                    logger.debug(f"Removed expired disk cache entry: {key}")
            except Exception as e:
                logger.error(f"Error reading cache file {cache_file}: {str(e)}")
                try:
                    os.remove(cache_file)
                except:
                    pass
        
        return None
    
    def set(self, key: str, result: Dict[str, Any]) -> None:
        """Cache a query result.
        
        Args:
            key: Cache key
            result: Query result to cache
        """
        # First check if we're at the disk cache limit
        self._limit_disk_cache()
        
        try:
            # Create a copy of the result for caching
            timestamp = time.time()
            cache_entry = {
                "result": result,
                "timestamp": timestamp
            }
            
            # Add to memory cache
            with self.cache_lock:
                self.memory_cache[key] = cache_entry
                
                # Limit memory cache size
                self._limit_memory_cache()
            
            # Write to disk cache
            cache_file = os.path.join(self.cache_dir, f"{key}.json")
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_entry, f, indent=2)
                
            logger.debug(f"Cached result for key: {key}")
                
        except Exception as e:
            logger.error(f"Failed to cache result for key {key}: {str(e)}")
    
    def invalidate(self, key: str) -> bool:
        """Invalidate a cache entry.
        
        Args:
            key: Cache key to invalidate
            
        Returns:
            True if entry was invalidated, False otherwise
        """
        invalidated = False
        
        # Remove from memory cache
        with self.cache_lock:
            if key in self.memory_cache:
                del self.memory_cache[key]
                invalidated = True
        
        # Remove from disk cache
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        if os.path.exists(cache_file):
            try:
                os.remove(cache_file)
                invalidated = True
                logger.debug(f"Invalidated cache entry: {key}")
            except Exception as e:
                logger.error(f"Failed to remove cache file {cache_file}: {str(e)}")
        
        return invalidated
    
    def clear(self) -> None:
        """Clear all cache entries."""
        # Clear memory cache
        with self.cache_lock:
            self.memory_cache.clear()
        
        # Clear disk cache
        try:
            # Remove all JSON files in cache directory
            for file_name in os.listdir(self.cache_dir):
                if file_name.endswith('.json'):
                    file_path = os.path.join(self.cache_dir, file_name)
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        logger.error(f"Failed to remove cache file {file_path}: {str(e)}")
            
            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear cache: {str(e)}")
    
    def _limit_memory_cache(self) -> None:
        """Limit the size of the memory cache."""
        if len(self.memory_cache) <= self.max_memory_entries:
            return
        
        # Sort entries by timestamp
        sorted_entries = sorted(
            self.memory_cache.items(),
            key=lambda x: x[1]["timestamp"]
        )
        
        # Remove oldest entries until we're under the limit
        entries_to_remove = len(self.memory_cache) - self.max_memory_entries
        for i in range(entries_to_remove):
            key, _ = sorted_entries[i]
            del self.memory_cache[key]
        
        logger.debug(f"Removed {entries_to_remove} oldest entries from memory cache")
    
    def _limit_disk_cache(self) -> None:
        """Limit the size of the disk cache."""
        max_size_mb = self.config["storage"].get("max_size_mb", 1024)
        if max_size_mb <= 0:
            return  # No limit
        
        max_size_bytes = max_size_mb * 1024 * 1024
        
        try:
            # Get current cache size
            total_size = 0
            cache_files = []
            
            for file_name in os.listdir(self.cache_dir):
                if file_name.endswith('.json'):
                    file_path = os.path.join(self.cache_dir, file_name)
                    file_size = os.path.getsize(file_path)
                    total_size += file_size
                    
                    # Get file timestamp from file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            entry = json.load(f)
                            timestamp = entry.get("timestamp", 0)
                    except:
                        # If can't read file, use file mtime
                        timestamp = os.path.getmtime(file_path)
                    
                    cache_files.append({
                        "path": file_path,
                        "size": file_size,
                        "timestamp": timestamp
                    })
            
            # If over limit, remove oldest files
            if total_size > max_size_bytes:
                # Sort by timestamp
                cache_files.sort(key=lambda x: x["timestamp"])
                
                # Calculate how much we need to remove
                bytes_to_remove = total_size - max_size_bytes
                bytes_removed = 0
                files_removed = 0
                
                for file_info in cache_files:
                    try:
                        os.remove(file_info["path"])
                        bytes_removed += file_info["size"]
                        files_removed += 1
                        
                        if bytes_removed >= bytes_to_remove:
                            break
                    except Exception as e:
                        logger.error(f"Failed to remove cache file {file_info['path']}: {str(e)}")
                
                logger.info(f"Removed {files_removed} cache files ({bytes_removed/1024/1024:.2f} MB)")
        except Exception as e:
            logger.error(f"Failed to limit disk cache size: {str(e)}")
    
    def _cleanup_expired_cache(self) -> None:
        """Clean up expired cache entries."""
        try:
            count = 0
            for file_name in os.listdir(self.cache_dir):
                if file_name.endswith('.json'):
                    file_path = os.path.join(self.cache_dir, file_name)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            entry = json.load(f)
                        
                        # Check if entry is expired
                        if time.time() - entry["timestamp"] >= self.cache_timeout:
                            os.remove(file_path)
                            count += 1
                    except Exception:
                        # If we can't read or parse the file, remove it
                        try:
                            os.remove(file_path)
                            count += 1
                        except:
                            pass
            
            if count > 0:
                logger.info(f"Cleaned up {count} expired cache entries")
        except Exception as e:
            logger.error(f"Failed to clean up expired cache entries: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            # Get memory cache stats
            memory_entries = len(self.memory_cache)
            
            # Get disk cache stats
            disk_entries = 0
            disk_size_bytes = 0
            oldest_entry = None
            newest_entry = None
            
            for file_name in os.listdir(self.cache_dir):
                if file_name.endswith('.json'):
                    disk_entries += 1
                    file_path = os.path.join(self.cache_dir, file_name)
                    disk_size_bytes += os.path.getsize(file_path)
                    
                    # Get timestamp from file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            entry = json.load(f)
                            timestamp = entry.get("timestamp", 0)
                            
                            if oldest_entry is None or timestamp < oldest_entry:
                                oldest_entry = timestamp
                            
                            if newest_entry is None or timestamp > newest_entry:
                                newest_entry = timestamp
                    except:
                        pass
            
            # Calculate age
            now = time.time()
            oldest_age = now - oldest_entry if oldest_entry is not None else None
            newest_age = now - newest_entry if newest_entry is not None else None
            
            return {
                "memory_entries": memory_entries,
                "disk_entries": disk_entries,
                "disk_size_mb": disk_size_bytes / 1024 / 1024,
                "oldest_entry_age": oldest_age,
                "newest_entry_age": newest_age,
                "cache_timeout": self.cache_timeout
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {str(e)}")
            return {
                "error": str(e)
            }
    
    @staticmethod
    def generate_key(query: str, params: Dict[str, Any] = None) -> str:
        """Generate a cache key from a query and parameters.
        
        Args:
            query: Query string
            params: Additional parameters
            
        Returns:
            Cache key string
        """
        # Create a string that includes the query and significant parameters
        key_str = query
        
        if params:
            # Sort parameters for consistent keys
            sorted_params = sorted(params.items())
            
            for param, value in sorted_params:
                if param in ['n_results', 'limit', 'offset', 'min_similarity']:
                    key_str += f"_{param}_{value}"
        
        # Generate MD5 hash
        key_hash = hashlib.md5(key_str.encode('utf-8')).hexdigest()
        
        # Add parameter count to key
        param_count = 0 if params is None else len(params)
        return f"{key_hash}_{param_count}"