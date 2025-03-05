#!/usr/bin/env python3
"""
Verification script for Data Store module.

This script manually tests key components of the Data Store implementation
without relying on unit tests.
"""

import os
import sys
import time
import json
import datetime
import tempfile
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from tccc.data_store import DataStore
from tccc.utils.logging import get_logger

# Set up logging
logger = get_logger(__name__)

def print_separator(title):
    """Print a separator with a title."""
    print("\n" + "=" * 30)
    print(f" {title} ".center(30, "="))
    print("=" * 30 + "\n")

def main():
    """Main verification function."""
    print("Starting Data Store verification...")
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "tccc.db")
        backup_dir = os.path.join(temp_dir, "backups")
        os.makedirs(backup_dir, exist_ok=True)
        
        print(f"Using temporary database at: {db_path}")
        
        # Create configuration
        config = {
            'database': {
                'type': 'sqlite',
                'sqlite': {
                    'database_path': db_path,
                    'wal_mode': True,
                    'journal_mode': 'wal',
                    'synchronous': 1,
                    'cache_size': 2000,
                    'page_size': 4096,
                    'auto_vacuum': 1,
                    'temp_store': 2,
                    'max_connections': 5
                }
            },
            'backup': {
                'enabled': True,
                'directory': backup_dir,
                'interval_hours': 24,
                'max_backups': 3
            },
            'query': {
                'cache_enabled': True,
                'cache_ttl': 300,
                'max_results': 1000,
                'default_page_size': 50
            },
            'context': {
                'max_tokens': 2000,
                'time_window': 300,
                'include_metadata': True
            },
            'performance': {
                'enable_indexes': True,
                'vacuum_interval_hours': 0,  # Disable for testing
                'analyze_interval_hours': 0,
                'jetson': {
                    'nvme_optimized': True,
                    'use_memory_mapping': True,
                    'memory_limit_mb': 256
                }
            }
        }
        
        # Create and initialize the data store
        print_separator("Initialization")
        data_store = DataStore()
        result = data_store.initialize(config)
        print(f"Initialization result: {result}")
        
        if not result:
            print("Initialization failed, exiting")
            return 1
        
        # Test storing events
        print_separator("Storing Events")
        for i in range(5):
            event = {
                'type': 'vital_signs',
                'source': 'patient_monitor',
                'severity': 'normal' if i % 3 != 0 else 'warning',
                'patient_id': f'patient_{i % 2 + 1}',
                'session_id': 'session_1',
                'timestamp': (datetime.datetime.now() - datetime.timedelta(minutes=i*5)).isoformat(),
                'data': {
                    'heart_rate': 60 + i * 5,
                    'blood_pressure': f"{110 + i * 2}/{70 + i}",
                    'temperature': 36.5 + i * 0.2,
                    'o2_saturation': 98 - i
                },
                'metadata': {
                    'device_id': f'monitor_{i + 1}',
                    'location': 'ER' if i % 2 == 0 else 'ICU'
                }
            }
            event_id = data_store.store_event(event)
            print(f"Stored event {i+1}: {event_id}")
        
        # Test storing reports
        print_separator("Storing Reports")
        for i in range(3):
            report = {
                'type': 'visit_summary' if i % 2 == 0 else 'assessment',
                'title': f"Visit Summary {i+1}" if i % 2 == 0 else f"Patient Assessment {i+1}",
                'patient_id': f'patient_{i % 2 + 1}',
                'session_id': 'session_1',
                'timestamp': (datetime.datetime.now() - datetime.timedelta(minutes=i*10)).isoformat(),
                'content': f"This is a test report content for report {i+1}.",
                'metadata': {
                    'author': f'provider_{i+1}',
                    'department': 'Cardiology' if i % 2 == 0 else 'Emergency'
                }
            }
            report_id = data_store.store_report(report)
            print(f"Stored report {i+1}: {report_id}")
        
        # Test querying events
        print_separator("Querying Events")
        print("All events:")
        events = data_store.query_events({})
        print(f"Total events: {len(events)}")
        
        print("\nEvents with warning severity:")
        warning_events = data_store.query_events({'severity': 'warning'})
        print(f"Warning events: {len(warning_events)}")
        for event in warning_events:
            print(f"  - {event['event_id']}: {event['type']} from {event['source']}")
        
        print("\nEvents for patient_1:")
        patient_events = data_store.query_events({'patient_id': 'patient_1'})
        print(f"Patient events: {len(patient_events)}")
        for event in patient_events:
            print(f"  - {event['event_id']}: {event['type']} at {event['timestamp']}")
        
        # Test timeline functionality
        print_separator("Timeline")
        start_time = (datetime.datetime.now() - datetime.timedelta(minutes=20)).isoformat()
        end_time = datetime.datetime.now().isoformat()
        timeline = data_store.get_timeline(start_time, end_time)
        print(f"Timeline events from last 20 minutes: {len(timeline)}")
        for i, event in enumerate(timeline[:3]):  # Show first 3 only
            print(f"  {i+1}. {event['timestamp']}: {event['type']} - {event['data']}")
        
        # Test context generation
        print_separator("Context Generation")
        context = data_store.get_context(datetime.datetime.now().isoformat(), 15*60)  # 15 minutes
        print(f"Context window: {context['time_window_seconds']} seconds")
        print(f"Events in context: {len(context['events'])}")
        print(f"Reports in context: {len(context['reports'])}")
        
        # Test backup functionality
        print_separator("Backup")
        backup = data_store.backup("verification_test")
        print(f"Backup created: {backup['backup_id']}")
        print(f"Backup path: {backup['path']}")
        print(f"Backup size: {backup['size']} bytes")
        
        backups = data_store.backup_manager.list_backups()
        print(f"Total backups: {len(backups)}")
        
        # Test status
        print_separator("Status")
        status = data_store.get_status()
        print(f"Data Store Status: {status['status']}")
        print(f"Database path: {status['database']['path']}")
        print(f"Database size: {status['database']['size_mb']:.2f} MB")
        print("Record counts:")
        for record_type, count in status['database']['counts'].items():
            print(f"  - {record_type}: {count}")
        
        print("\nBackup enabled: {0}".format("Yes" if status['backup']['enabled'] else "No"))
        print("Cache enabled: {0}".format("Yes" if status['cache']['enabled'] else "No"))
        
        print("\nVerification complete!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())