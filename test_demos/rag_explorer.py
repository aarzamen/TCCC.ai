#!/usr/bin/env python3
"""
RAG Database Explorer - Simple GUI for the TCCC RAG system
"""

import os
import sys
import time
import glob
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import threading
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir, 'src'))

# Import the Document Library
from tccc.document_library.document_library import DocumentLibrary
from tccc.utils.config import ConfigManager

class RAGDatabaseApp(tk.Tk):
    """GUI application for interacting with the RAG database."""
    
    def __init__(self):
        super().__init__()
        
        self.title("TCCC RAG Database Explorer")
        self.geometry("900x700")
        self.minsize(800, 600)
        
        # Initialize document library
        self.doc_lib = None
        self.config = None
        self.status_text = tk.StringVar(value="Initializing...")
        
        # Create tabs
        self.create_widgets()
        
        # Initialize document library in a separate thread
        self.init_thread = threading.Thread(target=self.initialize_doc_lib)
        self.init_thread.daemon = True
        self.init_thread.start()
        
        # Setup drag & drop for file uploads - will be configured later
        try:
            self.drop_frame.bind("<DragEnter>", lambda e: e.widget.configure(bg="#e0f0ff"))
            self.drop_frame.bind("<DragLeave>", lambda e: e.widget.configure(bg="#f0f0f0"))
        except:
            pass

    def create_widgets(self):
        """Create the GUI widgets."""
        # Create main notebook (tabbed interface)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.stats_tab = ttk.Frame(self.notebook)
        self.query_tab = ttk.Frame(self.notebook)
        self.upload_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.stats_tab, text="Database Stats")
        self.notebook.add(self.query_tab, text="Query RAG")
        self.notebook.add(self.upload_tab, text="Upload Documents")
        
        # Status bar at the bottom
        self.status_bar = ttk.Label(self, textvariable=self.status_text, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Create widgets for each tab
        self.create_stats_tab()
        self.create_query_tab()
        self.create_upload_tab()

    def create_stats_tab(self):
        """Create widgets for the statistics tab."""
        # Create a frame for the statistics
        self.stats_frame = ttk.LabelFrame(self.stats_tab, text="Database Statistics")
        self.stats_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Statistics text area
        self.stats_text = scrolledtext.ScrolledText(self.stats_frame, wrap=tk.WORD, height=20)
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.stats_text.config(state=tk.DISABLED)
        
        # Refresh button
        self.refresh_btn = ttk.Button(self.stats_frame, text="Refresh Statistics", command=self.refresh_stats)
        self.refresh_btn.pack(pady=10)

    def create_query_tab(self):
        """Create widgets for the query tab."""
        # Query input frame
        query_input_frame = ttk.Frame(self.query_tab)
        query_input_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(query_input_frame, text="Query:").pack(side=tk.LEFT, padx=5)
        
        self.query_entry = ttk.Entry(query_input_frame, width=50)
        self.query_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.query_entry.bind("<Return>", lambda e: self.execute_query())
        
        self.query_btn = ttk.Button(query_input_frame, text="Search", command=self.execute_query)
        self.query_btn.pack(side=tk.LEFT, padx=5)
        
        # Query options frame
        options_frame = ttk.LabelFrame(self.query_tab, text="Search Options")
        options_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Strategy selection
        strategy_frame = ttk.Frame(options_frame)
        strategy_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(strategy_frame, text="Strategy:").pack(side=tk.LEFT, padx=5)
        
        self.strategy_var = tk.StringVar(value="hybrid")
        strategies = [("Default", "default"), ("Semantic", "semantic"), 
                     ("Keyword", "keyword"), ("Hybrid", "hybrid")]
        
        for i, (text, value) in enumerate(strategies):
            ttk.Radiobutton(strategy_frame, text=text, value=value, 
                           variable=self.strategy_var).pack(side=tk.LEFT, padx=10)
        
        # Number of results
        results_frame = ttk.Frame(options_frame)
        results_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(results_frame, text="Max Results:").pack(side=tk.LEFT, padx=5)
        
        self.limit_var = tk.IntVar(value=5)
        self.limit_spinbox = ttk.Spinbox(results_frame, from_=1, to=20, textvariable=self.limit_var, width=5)
        self.limit_spinbox.pack(side=tk.LEFT, padx=5)
        
        # Results frame
        results_frame = ttk.LabelFrame(self.query_tab, text="Results")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.results_text.config(state=tk.DISABLED)

    def create_upload_tab(self):
        """Create widgets for the upload tab."""
        # Instructions
        instructions = "Drag and drop files here or use the button below to add documents to the RAG database."
        ttk.Label(self.upload_tab, text=instructions, wraplength=600).pack(pady=10)
        
        # Drop zone frame
        self.drop_frame = tk.LabelFrame(self.upload_tab, text="Drop Zone", bg="#f0f0f0")
        self.drop_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Drop zone message
        drop_msg = "Drag and drop files here"
        self.drop_label = ttk.Label(self.drop_frame, text=drop_msg, font=('Arial', 14))
        self.drop_label.pack(fill=tk.BOTH, expand=True, padx=20, pady=40)
        
        # Browse button
        self.browse_btn = ttk.Button(self.upload_tab, text="Browse Files", command=self.browse_files)
        self.browse_btn.pack(pady=10)
        
        # Upload log
        self.log_frame = ttk.LabelFrame(self.upload_tab, text="Upload Log")
        self.log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.log_text = scrolledtext.ScrolledText(self.log_frame, wrap=tk.WORD, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_text.config(state=tk.DISABLED)

    def update_status(self, message):
        """Update the status bar message."""
        self.status_text.set(message)
        self.update_idletasks()

    def find_config_file(self):
        """Find the document library configuration file."""
        # Direct path - most reliable
        direct_path = os.path.join(script_dir, "config", "document_library.yaml")
        if os.path.exists(direct_path):
            print(f"Found config at: {direct_path}")
            return direct_path
            
        # Try common locations
        common_paths = [
            # Parent directory
            os.path.join(os.path.dirname(script_dir), "config/document_library.yaml"),
            # Home directory
            os.path.join(os.path.expanduser("~"), "tccc-project/config/document_library.yaml"),
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                print(f"Found config at: {path}")
                return path
        
        print("Config file not found in common locations")
        return None

    def initialize_doc_lib(self):
        """Initialize the document library."""
        try:
            # Find configuration file
            config_path = self.find_config_file()
            
            if not config_path:
                self.update_status("Error: Could not find configuration file")
                return
                
            print(f"Loading config from: {config_path}")
            
            # Load configuration directly
            from tccc.utils.config import load_config
            self.config = load_config(config_path)
            
            print("Config loaded successfully")
            print(f"Config keys: {list(self.config.keys())}")
            
            # Initialize document library
            self.doc_lib = DocumentLibrary()
            print("Created document library instance")
            
            success = self.doc_lib.initialize(self.config)
            print(f"Document library initialization: {success}")
            
            if success:
                self.update_status("Document library initialized successfully")
                self.refresh_stats()
            else:
                self.update_status("Failed to initialize document library")
        
        except Exception as e:
            self.update_status(f"Error initializing document library: {str(e)}")
            import traceback
            traceback.print_exc()

    def refresh_stats(self):
        """Refresh the database statistics."""
        if not self.doc_lib:
            self.update_status("Document library not initialized")
            return
        
        try:
            # Get status from document library
            status = self.doc_lib.get_status()
            
            # Format status as readable text
            stats_text = "RAG Database Statistics\n"
            stats_text += "=====================\n\n"
            
            if "documents" in status:
                # Document statistics
                stats_text += "Documents:\n"
                stats_text += f"  Total documents: {status['documents'].get('count', 0)}\n"
                stats_text += f"  Total chunks: {status['documents'].get('chunks', 0)}\n\n"
            
            if "index" in status:
                # Index statistics
                stats_text += "Vector Index:\n"
                stats_text += f"  Total vectors: {status['index'].get('vectors', 0)}\n"
                stats_text += f"  Dimension: {status['index'].get('dimension', 0)}\n\n"
            
            # Storage path
            if self.config and "storage" in self.config:
                stats_text += "Storage:\n"
                stats_text += f"  Base directory: {self.config['storage'].get('base_dir', '')}\n"
                stats_text += f"  Index path: {self.config['storage'].get('index_path', '')}\n\n"
            
            # Component status
            if "components" in status:
                stats_text += "Components:\n"
                for component, status_value in status['components'].items():
                    status_str = "Available" if status_value else "Not available"
                    stats_text += f"  {component}: {status_str}\n"
            
            # Update text widget
            self.stats_text.config(state=tk.NORMAL)
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(tk.END, stats_text)
            self.stats_text.config(state=tk.DISABLED)
            
            self.update_status("Statistics refreshed")
            
        except Exception as e:
            self.update_status(f"Error refreshing statistics: {str(e)}")
            import traceback
            traceback.print_exc()

    def execute_query(self):
        """Execute a query against the RAG database."""
        if not self.doc_lib:
            self.update_status("Document library not initialized")
            return
        
        query_text = self.query_entry.get().strip()
        if not query_text:
            self.update_status("Please enter a query")
            return
        
        try:
            # Get query parameters
            strategy = self.strategy_var.get()
            limit = self.limit_var.get()
            
            # Execute query
            self.update_status(f"Executing query: {query_text}")
            
            # Use advanced query
            if hasattr(self.doc_lib, 'advanced_query') and strategy != "default":
                results = self.doc_lib.advanced_query(
                    query_text=query_text,
                    strategy=strategy,
                    limit=limit
                )
            else:
                # Fall back to basic query
                results = self.doc_lib.query(query_text, n_results=limit)
            
            # Display results
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            
            # Display query info
            self.results_text.insert(tk.END, f"Query: {query_text}\n")
            if "strategy" in results:
                self.results_text.insert(tk.END, f"Strategy: {results['strategy']}\n")
            self.results_text.insert(tk.END, f"Results found: {len(results['results'])}\n\n")
            
            # Display results
            if results['results']:
                for i, result in enumerate(results['results']):
                    self.results_text.insert(tk.END, f"Result {i+1} - Score: {result['score']:.4f}\n")
                    
                    # Display source if available
                    if 'source' in result:
                        self.results_text.insert(tk.END, f"Source: {result['source']}\n")
                    
                    # Display metadata if available
                    if 'metadata' in result and result['metadata']:
                        self.results_text.insert(tk.END, "Metadata:\n")
                        for key, value in result['metadata'].items():
                            if key in ['text', 'content']:  # Skip large text fields
                                continue
                            self.results_text.insert(tk.END, f"  {key}: {value}\n")
                    
                    # Display text truncated
                    if 'text' in result:
                        text = result['text']
                        if len(text) > 300:
                            text = text[:300] + "...(truncated)"
                        self.results_text.insert(tk.END, f"Text:\n{text}\n")
                    
                    self.results_text.insert(tk.END, "\n" + "-"*50 + "\n\n")
            else:
                self.results_text.insert(tk.END, "No results found.\n")
            
            self.results_text.config(state=tk.DISABLED)
            self.update_status("Query completed")
            
        except Exception as e:
            self.update_status(f"Error executing query: {str(e)}")
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Error: {str(e)}")
            self.results_text.config(state=tk.DISABLED)
            import traceback
            traceback.print_exc()

    def browse_files(self):
        """Open file browser to select files to add to the database."""
        filetypes = [
            ("Document files", "*.pdf *.docx *.txt *.md *.html"),
            ("PDF files", "*.pdf"),
            ("Word documents", "*.docx"),
            ("Text files", "*.txt"),
            ("All files", "*.*")
        ]
        
        filenames = filedialog.askopenfilenames(title="Select files to add to RAG database",
                                               filetypes=filetypes)
        if filenames:
            self.process_files(filenames)

    def handle_drop(self, event):
        """Handle files dropped onto the application."""
        file_paths = self.tk.splitlist(event.data)
        self.process_files(file_paths)

    def process_files(self, file_paths):
        """Process a list of files to add to the database."""
        if not self.doc_lib:
            self.update_status("Document library not initialized")
            return
        
        # Enable writing to log text
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"Processing {len(file_paths)} files...\n")
        self.log_text.see(tk.END)
        
        # Copy files to documents directory if configured
        documents_dir = None
        if self.config and "storage" in self.config and "base_dir" in self.config["storage"]:
            documents_dir = os.path.join(script_dir, self.config["storage"]["base_dir"])
            if not os.path.exists(documents_dir):
                os.makedirs(documents_dir, exist_ok=True)
        
        # Process each file
        successful = 0
        for file_path in file_paths:
            try:
                basename = os.path.basename(file_path)
                self.update_status(f"Processing file: {basename}")
                self.log_text.insert(tk.END, f"Adding: {basename}... ")
                self.log_text.see(tk.END)
                self.update_idletasks()
                
                # Copy file to documents directory if available
                target_path = file_path
                if documents_dir:
                    target_path = os.path.join(documents_dir, basename)
                    shutil.copy2(file_path, target_path)
                
                # Create document data
                document_data = {
                    "file_path": target_path,
                    "metadata": {
                        "title": os.path.splitext(basename)[0],
                        "source": "GUI Upload",
                        "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                }
                
                # Add document to library
                doc_id = self.doc_lib.add_document(document_data)
                
                if doc_id:
                    self.log_text.insert(tk.END, f"Success! Document ID: {doc_id}\n")
                    successful += 1
                else:
                    self.log_text.insert(tk.END, "Failed.\n")
                
                self.log_text.see(tk.END)
                
            except Exception as e:
                self.log_text.insert(tk.END, f"Error: {str(e)}\n")
                self.log_text.see(tk.END)
        
        # Summary
        self.log_text.insert(tk.END, f"\nProcessed {len(file_paths)} files, {successful} added successfully.\n")
        self.log_text.insert(tk.END, "-"*50 + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        
        # Refresh statistics
        self.refresh_stats()
        self.update_status(f"Added {successful} documents to the database")


# Create an executable with drag and drop support
if __name__ == "__main__":
    # Import tkinter without drag and drop first
    app = RAGDatabaseApp()
    
    # Try to setup drag and drop after initialization
    try:
        # For fallback drag and drop handling
        def drop_handler(event):
            print(f"Drop event: {event.data}")
            file_paths = event.data.split()
            # Clean up file paths (they might have {})
            clean_paths = [p.strip('{}') for p in file_paths]
            app.process_files(clean_paths)
        
        app.drop_frame.bind("<Drop>", drop_handler)
    except Exception as e:
        print(f"Could not set up drag and drop: {e}")
    
    print("RAG Explorer started")
    app.mainloop()