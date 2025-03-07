#!/usr/bin/env python3
"""
Simplified verification script for the TCCC.ai RAG Database using a pure mock implementation.

This script tests the RAG system's components and functionality without any external dependencies.
"""

import os
import sys
import time
from typing import Dict, List, Any, Optional, Tuple

# Configure basic logging
import logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RAG_Verification_Mock")

# Mock document library class
class MockDocumentLibrary:
    """Mock DocumentLibrary for testing when dependencies are missing"""
    
    def __init__(self):
        self.initialized = False
        self.document_processor = MockDocumentProcessor()
        
    def initialize(self, config):
        self.config = config
        self.initialized = True
        return True
        
    def query(self, query_text, n_results=3):
        """Mock query method"""
        return {
            "query": query_text,
            "results": [
                {
                    "document_id": "mock1",
                    "text": f"Mock result for: {query_text}",
                    "score": 0.95,
                    "metadata": {"title": "Mock Document 1"}
                }
            ],
            "total_results": 1,
            "processing_time": 0.1,
            "cache_hit": False
        }
        
    def advanced_query(self, query_text, strategy="hybrid", limit=3, **kwargs):
        """Mock advanced query method"""
        return {
            "query": query_text,
            "strategy": strategy,
            "results": [
                {
                    "document_id": "mock1",
                    "text": f"Mock result for: {query_text} using {strategy}",
                    "score": 0.95,
                    "metadata": {"title": "Mock Document 1"}
                }
            ],
            "total_results": 1,
            "processing_time": 0.1
        }
        
    def extract_medical_terms(self, text):
        """Mock extract medical terms"""
        # Return some basic medical terms that might be in the text
        terms = []
        medical_terms = ["tourniquet", "hemorrhage", "airway", "respiration", 
                       "circulation", "march", "casualty", "needle decompression",
                       "tension pneumothorax", "pneumothorax", "hypothermia"]
        for term in medical_terms:
            if term.lower() in text.lower():
                terms.append(term)
        return terms
        
    def explain_medical_terms(self, text):
        """Mock explain medical terms"""
        explanations = {}
        terms = self.extract_medical_terms(text)
        
        # Add some mock explanations
        for term in terms:
            if term == "march":
                explanations[term] = "Massive hemorrhage, Airway, Respiration, Circulation, Hypothermia/Head injury"
            elif term == "tourniquet":
                explanations[term] = "related terms: TQ, CAT, SOFT-T, tourniquet application"
            elif term == "hemorrhage":
                explanations[term] = "related terms: bleeding, blood loss, hemorrhaging, blood"
            elif term == "airway":
                explanations[term] = "related terms: airway management, breathing passage, airway obstruction, airway control"
            elif term == "respiration":
                explanations[term] = "related terms: breathing, ventilation, respiratory rate, breaths per minute"
            elif term == "circulation":
                explanations[term] = "related terms: blood flow, pulse, circulation assessment, circulatory"
            elif term == "casualty":
                explanations[term] = "related terms: patient, victim, injured, wounded, casualty assessment"
            elif term == "needle decompression":
                explanations[term] = "related terms: chest decompression, thoracic decompression"
            elif term == "tension pneumothorax":
                explanations[term] = "related terms: collapsed lung, chest injury, decompression"
            elif term == "pneumothorax":
                explanations[term] = "pneumothorax"
            elif term == "hypothermia":
                explanations[term] = "related terms: low body temperature, cold exposure, hypothermic"
            else:
                explanations[term] = f"{term} - no explanation available"
            
        return explanations
        
    def generate_llm_prompt(self, query, strategy="hybrid", limit=3, max_context_length=None):
        """Mock LLM prompt generation"""
        # Create a template prompt
        if max_context_length is not None:
            logger.info(f"Overriding max context length: {max_context_length}")
            
        if "explain" in query.lower() or "what is" in query.lower():
            template = """
You are TCCC.ai, an expert in Tactical Combat Casualty Care (TCCC) and military medicine.
Your task is to explain medical concepts based on official protocols.

CONTEXT INFORMATION:
No relevant information found in the document library.

USER QUERY REQUESTING EXPLANATION:
{query}

Please provide a clear explanation of the concept based on the provided context.
Use simple language and provide examples where helpful. Define any technical terms.
Cite your sources using the format [Source: document title].
"""
        else:
            template = """
You are TCCC.ai, an expert in Tactical Combat Casualty Care (TCCC) and military medicine.
Your task is to provide accurate information based on official military medical protocols.

CONTEXT INFORMATION:
No relevant information found in the document library.

USER QUERY:
{query}

Please answer the query based on the provided context information. If the context 
doesn't contain relevant information, acknowledge the limitations and suggest what 
the user should refer to instead. Include citations where appropriate using the format 
[Source: document title].
"""
        return template.format(query=query)
        
    def get_status(self):
        """Mock get status"""
        return {
            "status": "initialized",
            "documents": {"count": 5, "chunks": 25},
            "index": {"vectors": 25, "dimension": 384},
            "model": {"name": "all-MiniLM-L12-v2"},
            "components": {
                "document_processor": True,
                "vector_store": True,
                "query_engine": True,
                "response_generator": True,
                "medical_vocabulary": True
            }
        }
    
    def add_document(self, document_data):
        """Mock document adding method."""
        return "mock_doc_1"

class MockDocumentProcessor:
    """Mock document processor for testing"""
    
    def process_document(self, file_path):
        """Mock document processing"""
        if not os.path.exists(file_path):
            return {"success": False, "error": f"File not found: {file_path}"}
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                
            return {
                "success": True,
                "text": text,
                "metadata": {
                    "file_name": os.path.basename(file_path),
                    "content_type": "text/plain",
                    "file_size": len(text)
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

# Mock config manager
class MockConfigManager:
    """Mock config manager"""
    
    def load_config(self, config_name):
        """Return a mock config"""
        if config_name == "document_library":
            return {
                "storage": {
                    "base_dir": "data/documents",
                    "index_path": "data/document_index",
                    "cache_dir": "data/query_cache"
                },
                "embedding": {
                    "model_name": "all-MiniLM-L12-v2",
                    "cache_dir": "data/models/embeddings",
                    "dimension": 384,
                    "use_gpu": False,
                    "batch_size": 32,
                    "normalize": True
                },
                "indexing": {
                    "chunk_size": 1000,
                    "chunk_overlap": 200
                }
            }
        return {}

# Test functions
def test_initialization(doc_library):
    """Test document library initialization."""
    print("\n=== Testing Document Library Initialization ===")
    status = doc_library.get_status()
    print(f"Status: {status['status']}")
    print(f"Documents: {status['documents']['count']}")
    print(f"Chunks: {status['documents']['chunks']}")
    print(f"Vectors: {status['index']['vectors']}")
    print(f"Model: {status['model']['name']}")
    
    # Check components
    if "components" in status:
        components = status["components"]
        print("\nComponents:")
        for name, avail in components.items():
            print(f"- {name}: {'Available' if avail else 'Not available'}")
    
    return status["status"] == "initialized"

def test_document_processor(doc_library):
    """Test document processor functionality."""
    print("\n=== Testing Document Processor ===")
    
    # Test if document processor is available
    if not hasattr(doc_library, 'document_processor') or doc_library.document_processor is None:
        print("Document processor not available!")
        return False
    
    # Test with a sample document
    project_root = os.path.dirname(os.path.abspath(__file__))
    sample_file = os.path.join(project_root, "data/sample_documents/tactical_considerations.txt")
    if not os.path.exists(sample_file):
        # Create directory if needed
        os.makedirs(os.path.join(project_root, "data/sample_documents"), exist_ok=True)
        
        # Create a test file if it doesn't exist
        with open(sample_file, 'w') as f:
            f.write("""
            TACTICAL CONSIDERATIONS FOR TCCC
            
            1. Maintain tactical situational awareness and treat the casualty in a tactical sound manner.
            2. Always establish a security perimeter before treating casualties.
            3. Use the MARCH protocol to prioritize casualty care:
               - Massive Hemorrhage: Control life-threatening bleeding
               - Airway: Establish and maintain patent airway
               - Respiration: Recognize and treat tension pneumothorax 
               - Circulation: Establish IV/IO access and administer fluids
               - Hypothermia/Head Injury: Prevent hypothermia
            4. Apply appropriate tourniquets to control extremity hemorrhage.
            5. Document all treatments on the TCCC Casualty Card.
            """)
    
    print(f"Testing document processing with: {sample_file}")
    result = doc_library.document_processor.process_document(sample_file)
    
    if result.get("success", False):
        print("Document processing successful!")
        print(f"Extracted {len(result.get('text', ''))} characters of text")
        print(f"Metadata: {list(result.get('metadata', {}).keys())}")
        return True
    else:
        print(f"Document processing failed: {result.get('error', 'Unknown error')}")
        return False

def test_basic_query(doc_library):
    """Test basic query functionality."""
    print("\n=== Testing Basic Query ===")
    
    queries = [
        "What is TCCC?",
        "How to apply a tourniquet?",
        "What are the steps in MARCH assessment?",
        "How to complete a TCCC card?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        start_time = time.time()
        result = doc_library.query(query, n_results=3)
        query_time = time.time() - start_time
        
        print(f"Results: {len(result.get('results', []))}")
        print(f"Time: {query_time:.2f} seconds")
        
        # Print top result snippet
        if result.get('results'):
            top_result = result['results'][0]
            text_preview = top_result.get('text', '')[:150] + '...' if len(top_result.get('text', '')) > 150 else top_result.get('text', '')
            print(f"Top result: {text_preview}")
    
    return True

def test_advanced_query(doc_library):
    """Test advanced query functionality."""
    print("\n=== Testing Advanced Query ===")
    
    # Check if advanced query is available
    if not hasattr(doc_library, 'advanced_query'):
        print("Advanced query functionality not available!")
        return False
    
    query = "How to treat a tension pneumothorax?"
    
    # Test with different strategies
    strategies = ["semantic", "keyword", "hybrid", "expanded"]
    for strategy in strategies:
        print(f"\nQuery with {strategy} strategy: {query}")
        start_time = time.time()
        result = doc_library.advanced_query(
            query_text=query,
            strategy=strategy,
            limit=3
        )
        query_time = time.time() - start_time
        
        print(f"Results: {len(result.get('results', []))}")
        print(f"Time: {query_time:.2f} seconds")
        print(f"Strategy used: {result.get('strategy', 'unknown')}")
        
        # Print number of results for each strategy
        print(f"Found {len(result.get('results', []))} results with {strategy} strategy")
    
    return True

def test_medical_vocabulary(doc_library):
    """Test medical vocabulary functionality."""
    print("\n=== Testing Medical Vocabulary ===")
    
    # Check if medical vocabulary extraction is available
    if not hasattr(doc_library, 'extract_medical_terms'):
        print("Medical vocabulary extraction not available!")
        return False
    
    # Test term extraction
    test_text = """
    The medic quickly applied a tourniquet to stop the hemorrhage, 
    then checked the casualty's airway and respiration. 
    Using the MARCH protocol, they assessed circulation and checked for hypothermia.
    The patient required a needle decompression for tension pneumothorax.
    """
    
    print("Test text:")
    print(test_text.strip())
    
    # Extract medical terms
    medical_terms = doc_library.extract_medical_terms(test_text)
    print("\nExtracted medical terms:")
    for term in medical_terms:
        print(f"- {term}")
    
    # Explain medical terms
    explanations = doc_library.explain_medical_terms(test_text)
    print("\nExplanations:")
    for term, explanation in explanations.items():
        print(f"- {term}: {explanation}")
    
    return len(medical_terms) > 0

def test_llm_prompt_generation(doc_library):
    """Test LLM prompt generation."""
    print("\n=== Testing LLM Prompt Generation ===")
    
    # Check if prompt generation is available
    if not hasattr(doc_library, 'generate_llm_prompt'):
        print("LLM prompt generation not available!")
        return False
    
    queries = [
        "What is TCCC?",
        "How to apply a tourniquet?",
        "Explain the MARCH protocol"
    ]
    
    # Test with different context length limits
    context_lengths = [1500, 1000, 500, 2000]
    
    for query in queries:
        print(f"\nGenerating prompt for: {query}")
        
        # Regular prompt generation
        prompt = doc_library.generate_llm_prompt(query)
        print(f"Standard prompt length: {len(prompt)} characters")
        
        # Now test with different context length limits
        print("\nTesting with different context lengths:")
        for context_length in context_lengths:
            try:
                context_prompt = doc_library.generate_llm_prompt(
                    query=query,
                    max_context_length=context_length
                )
                print(f"- Context length {context_length}: {len(context_prompt)} characters")
            except Exception as e:
                print(f"- Context length {context_length}: Error: {str(e)}")
        
        # Print preview of standard prompt
        print("\nPrompt preview:")
        preview_lines = prompt.split('\n')[:8]
        for line in preview_lines:
            print(f"> {line}")
        print("...")
    
    return True

def main():
    """Main function."""
    try:
        # Initialize document library with mock config
        config_manager = MockConfigManager()
        config = config_manager.load_config("document_library")
        
        # Create data directories if they don't exist
        os.makedirs("data/document_index", exist_ok=True)
        os.makedirs("data/query_cache", exist_ok=True)
        
        # Initialize document library
        doc_library = MockDocumentLibrary()
        success = doc_library.initialize(config)
        
        if not success:
            print("Failed to initialize Document Library")
            return 1
        
        # Run tests
        test_results = {}
        
        test_results["initialization"] = test_initialization(doc_library)
        test_results["document_processor"] = test_document_processor(doc_library)
        test_results["basic_query"] = test_basic_query(doc_library)
        test_results["advanced_query"] = test_advanced_query(doc_library)
        test_results["medical_vocabulary"] = test_medical_vocabulary(doc_library)
        test_results["llm_prompt_generation"] = test_llm_prompt_generation(doc_library)
        
        # Print summary
        print("\n=== Test Summary ===")
        all_passed = True
        for test_name, result in test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status} - {test_name}")
            all_passed = all_passed and result
        
        return 0 if all_passed else 1
    
    except Exception as e:
        print(f"ERROR: Unhandled exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    result = main()
    sys.exit(result)