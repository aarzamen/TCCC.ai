#!/usr/bin/env python3
"""
Test script for medical terminology handling in the RAG system.

This script evaluates the system's ability to understand and process
medical terminology, specifically focusing on TCCC-related terms.
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Import TCCC modules
from tccc.document_library import DocumentLibrary
from tccc.document_library.medical_vocabulary import MedicalVocabularyManager
from tccc.utils import ConfigManager

# TCCC-specific medical queries for testing
TCCC_MEDICAL_QUERIES = [
    # Hemorrhage control queries
    "How do I apply a CAT tourniquet?",
    "Treatment for junctional hemorrhage",
    "When should hemostatic agents be used?",
    
    # Airway management queries
    "NPA insertion procedure",
    "Signs of airway obstruction in combat",
    "TCCC recommendations for airway management",
    
    # Respiratory queries
    "How to recognize tension pneumothorax",
    "Needle decompression procedure",
    "Battlefield assessment of respiration",
    
    # Circulation queries
    "Hypovolemic shock treatment",
    "How to assess circulation in TCCC",
    "Field blood transfusion protocol",
    
    # Head injury / hypothermia queries
    "Combat TBI assessment guidelines",
    "Preventing hypothermia in casualties",
    "Head injury treatment in tactical field care",
    
    # General TCCC queries
    "MARCH algorithm explanation",
    "Principles of tactical field care",
    "TCCC card documentation guidelines"
]

# Alternative query forms for testing query expansion
ALTERNATIVE_QUERY_FORMS = {
    "How do I apply a CAT tourniquet?": [
        "CAT application technique",
        "How to use a tourniquet",
        "Applying combat application tourniquet"
    ],
    "Treatment for junctional hemorrhage": [
        "How to stop bleeding in the groin",
        "Axilla bleeding management",
        "Junctional TQ procedures"
    ],
    "How to recognize tension pneumothorax": [
        "Tension pneumo symptoms",
        "Signs of collapsed lung in combat",
        "Identifying thoracic pressure injury"
    ]
}

def test_vocabulary_coverage(vocab_manager):
    """Test the coverage of the medical vocabulary."""
    print("\n=== Testing Medical Vocabulary Coverage ===")
    
    # Test key TCCC terms
    test_terms = [
        "TCCC", "TQ", "CAT", "Hemorrhage", "Tourniquet", "NPA",
        "Tension pneumothorax", "Needle decompression", "Hemostatic",
        "MARCH", "Airway", "Respiration", "Circulation", "Hypothermia"
    ]
    
    found_terms = 0
    for term in test_terms:
        info = vocab_manager.get_term_info(term)
        if info:
            found_terms += 1
            term_type = info.get("type", "unknown")
            if term_type == "abbreviation":
                expansion = info.get("expansion", "")
                print(f"✓ {term} ({term_type}): {expansion}")
            elif "synonyms" in info and info["synonyms"]:
                synonyms = ", ".join(info["synonyms"][:3])
                print(f"✓ {term} ({term_type}): synonyms: {synonyms}...")
            else:
                definition = info.get("definition", "")
                print(f"✓ {term} ({term_type}): {definition}")
        else:
            print(f"✗ {term}: Not found in vocabulary")
    
    coverage = found_terms / len(test_terms) * 100
    print(f"\nVocabulary coverage: {coverage:.1f}% ({found_terms}/{len(test_terms)} terms found)")
    
    return coverage

def test_query_expansion(vocab_manager):
    """Test query expansion functionality."""
    print("\n=== Testing Query Expansion ===")
    
    # Test with key queries
    expansion_test_queries = list(ALTERNATIVE_QUERY_FORMS.keys())
    
    total_expansions = 0
    for query in expansion_test_queries:
        expansions = vocab_manager.expand_query(query)
        total_expansions += len(expansions)
        
        print(f"\nOriginal query: '{query}'")
        print(f"Generated {len(expansions)} expansions:")
        for i, exp in enumerate(expansions[:5], 1):  # Show first 5
            print(f"  {i}. {exp}")
        if len(expansions) > 5:
            print(f"  ... {len(expansions)-5} more expansions")
            
        # Check if any of the known alternative forms were generated
        if query in ALTERNATIVE_QUERY_FORMS:
            alternatives = ALTERNATIVE_QUERY_FORMS[query]
            found_alternatives = []
            
            for alt in alternatives:
                for exp in expansions:
                    if alt.lower() in exp.lower():
                        found_alternatives.append(alt)
                        break
            
            if found_alternatives:
                print(f"✓ Found {len(found_alternatives)}/{len(alternatives)} expected alternative forms")
            else:
                print(f"✗ None of the expected alternative forms were generated")
    
    avg_expansions = total_expansions / len(expansion_test_queries)
    print(f"\nAverage expansions per query: {avg_expansions:.2f}")
    
    return avg_expansions

def test_term_extraction(vocab_manager):
    """Test medical term extraction from text."""
    print("\n=== Testing Medical Term Extraction ===")
    
    test_texts = [
        "The patient presented with a tension pneumothorax requiring immediate needle decompression.",
        "Applied a CAT tourniquet to control hemorrhage from the lower extremity.",
        "Following the MARCH algorithm, first address massive hemorrhage, then airway, respiration, circulation, and treat for hypothermia or head injury.",
        "During tactical field care (TFC), an NPA was inserted to maintain the airway.",
        "The medic documented all interventions on the TCCC card including the time of tourniquet application."
    ]
    
    total_terms = 0
    for i, text in enumerate(test_texts, 1):
        terms = vocab_manager.extract_medical_terms(text)
        total_terms += len(terms)
        
        print(f"\nText {i}: {text}")
        print(f"Extracted {len(terms)} medical terms:")
        for term in terms:
            print(f"  - {term}")
        
        # Get explanations
        explanations = vocab_manager.explain_medical_terms(text)
        if explanations:
            print(f"Sample explanations:")
            for term, explanation in list(explanations.items())[:3]:
                print(f"  {explanation}")
    
    avg_terms = total_terms / len(test_texts)
    print(f"\nAverage terms extracted per text: {avg_terms:.2f}")
    
    return avg_terms

def test_query_performance(doc_lib):
    """Test query performance with medical terminology."""
    print("\n=== Testing Query Performance with Medical Terminology ===")
    
    strategies = ["semantic", "keyword", "hybrid", "expanded"]
    results = {}
    
    for strategy in strategies:
        print(f"\nTesting strategy: {strategy}")
        total_time = 0
        total_results = 0
        query_times = []
        
        for i, query in enumerate(TCCC_MEDICAL_QUERIES[:5], 1):  # Test with first 5 queries
            start_time = time.time()
            result = doc_lib.advanced_query(query, strategy=strategy, limit=3)
            query_time = time.time() - start_time
            
            query_times.append(query_time)
            total_time += query_time
            total_results += len(result.get('results', []))
            
            print(f"Query {i}: '{query}'")
            print(f"  Time: {query_time:.4f}s, Results: {len(result.get('results', []))}")
            
            # Show first result if available
            if result.get('results'):
                score = result['results'][0].get('score', 0)
                print(f"  Top score: {score:.4f}")
        
        avg_time = total_time / len(TCCC_MEDICAL_QUERIES[:5])
        avg_results = total_results / len(TCCC_MEDICAL_QUERIES[:5])
        results[strategy] = (avg_time, avg_results)
        
        print(f"Strategy {strategy}: Avg time = {avg_time:.4f}s, Avg results = {avg_results:.1f}")
    
    # Determine best strategy based on query time
    best_strategy = min(results.items(), key=lambda x: x[1][0])[0]
    print(f"\nBest performing strategy: {best_strategy} ({results[best_strategy][0]:.4f}s)")
    
    return best_strategy, results

def main():
    parser = argparse.ArgumentParser(description="Test medical terminology handling in RAG system")
    parser.add_argument("--config", type=str, help="Path to custom config file")
    parser.add_argument("--skip-query-test", action="store_true", help="Skip query performance testing")
    args = parser.parse_args()
    
    # Load configuration
    config_manager = ConfigManager()
    if args.config:
        print(f"Loading custom config from {args.config}")
        config = config_manager.load_config_from_file(args.config)
    else:
        optimized_config = "config/optimized_jetson_rag.yaml"
        if os.path.exists(optimized_config):
            print(f"Loading optimized config from {optimized_config}")
            config = config_manager.load_config_from_file(optimized_config)
        else:
            print("Loading default document library config")
            config = config_manager.load_config("document_library")
    
    # Initialize vocabulary manager
    print("Initializing Medical Vocabulary Manager...")
    vocab_manager = MedicalVocabularyManager(config)
    vocab_success = vocab_manager.initialize()
    
    if not vocab_success:
        print("Failed to initialize medical vocabulary")
        return 1
    
    # Run vocabulary tests
    coverage = test_vocabulary_coverage(vocab_manager)
    avg_expansions = test_query_expansion(vocab_manager)
    avg_terms = test_term_extraction(vocab_manager)
    
    # Initialize document library if needed for query tests
    if not args.skip_query_test:
        print("\nInitializing Document Library for query testing...")
        doc_lib = DocumentLibrary()
        success = doc_lib.initialize(config)
        
        if not success:
            print("Failed to initialize Document Library")
            return 1
        
        # Run query performance tests
        best_strategy, query_results = test_query_performance(doc_lib)
    
    # Print summary
    print("\n=== Test Summary ===")
    print(f"Medical vocabulary coverage: {coverage:.1f}%")
    print(f"Average query expansions: {avg_expansions:.2f}")
    print(f"Average terms extracted: {avg_terms:.2f}")
    
    if not args.skip_query_test:
        print(f"Best query strategy: {best_strategy}")
        print("Query performance by strategy:")
        for strategy, (avg_time, avg_results) in query_results.items():
            print(f"  {strategy}: {avg_time:.4f}s, {avg_results:.1f} results")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())