"""
Medical Vocabulary Manager for the TCCC.ai Document Library.

This module provides medical terminology support for the Document Library,
including synonym expansion, abbreviation handling, and domain-specific
knowledge enhancement.
"""

import os
import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Set

from tccc.utils.logging import get_logger

logger = get_logger(__name__)

class MedicalVocabularyManager:
    """
    Medical Vocabulary Manager for Document Library.
    
    This class provides:
    - Medical term synonym expansion
    - Abbreviation handling (e.g., TCCC, MARCH, MIST)
    - Military-specific medical terminology
    - Query enhancement with domain knowledge
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Medical Vocabulary Manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.vocab_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "config/vocabulary/custom_terms.txt"
        )
        self.terms_dict = {}
        self.abbreviations = {}
        self.synonyms = {}
        self.initialized = False
    
    def initialize(self) -> bool:
        """
        Initialize the vocabulary from the custom terms file.
        
        Returns:
            Success status
        """
        try:
            # Check if vocabulary file exists
            if not os.path.exists(self.vocab_path):
                logger.warning(f"Vocabulary file not found: {self.vocab_path}")
                self._create_default_vocabulary()
            
            # Load vocabulary
            self._load_vocabulary()
            
            self.initialized = True
            logger.info(f"Medical vocabulary initialized with {len(self.terms_dict)} terms")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize medical vocabulary: {str(e)}")
            return False
    
    def _load_vocabulary(self) -> None:
        """Load vocabulary from file."""
        try:
            with open(self.vocab_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Process each line
            for line in lines:
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Handle abbreviations (ABBR = Full Form)
                if " = " in line:
                    abbr, full_form = [part.strip() for part in line.split(" = ", 1)]
                    self.abbreviations[abbr.upper()] = full_form
                    self.terms_dict[abbr.upper()] = full_form
                    continue
                    
                # Handle synonyms (term -> synonym1, synonym2, ...)
                if " -> " in line:
                    term, synonyms_str = [part.strip() for part in line.split(" -> ", 1)]
                    synonyms = [s.strip() for s in synonyms_str.split(",")]
                    self.synonyms[term.lower()] = synonyms
                    self.terms_dict[term.lower()] = synonyms
                    continue
                    
                # Add standard term
                self.terms_dict[line.lower()] = line
                
        except Exception as e:
            logger.error(f"Failed to load vocabulary: {str(e)}")
            raise
    
    def _create_default_vocabulary(self) -> None:
        """Create default vocabulary if file doesn't exist."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.vocab_path), exist_ok=True)
            
            # Create default vocabulary with common TCCC terms
            default_terms = [
                "# TCCC.ai Custom Medical Vocabulary",
                "# Format:",
                "# 1. Standard terms: one per line",
                "# 2. Abbreviations: ABBR = Full Form",
                "# 3. Synonyms: term -> synonym1, synonym2, ...",
                "",
                "# Common TCCC abbreviations",
                "TCCC = Tactical Combat Casualty Care",
                "TFC = Tactical Field Care",
                "CUF = Care Under Fire",
                "MARCH = Massive hemorrhage, Airway, Respiration, Circulation, Hypothermia/Head injury",
                "MIST = Mechanism, Injury, Signs, Treatment",
                "SMOG = Symptoms, Mechanism, Injury, Signs, Treatment",
                "TQ = Tourniquet",
                "NPA = Nasopharyngeal Airway",
                "CAT = Combat Application Tourniquet",
                "ATNAA = Antidote Treatment Nerve Agent Auto-Injector",
                "CBRN = Chemical, Biological, Radiological, Nuclear",
                "FAS = Focused Assessment Sonography",
                "DCR = Damage Control Resuscitation",
                "JTS = Joint Trauma System",
                "GCS = Glasgow Coma Scale",
                "MEDEVAC = Medical Evacuation",
                "TBI = Traumatic Brain Injury",
                "IO = Intraosseous",
                "IV = Intravenous",
                "LOC = Loss of Consciousness",
                "MCI = Mass Casualty Incident",
                "MOI = Mechanism of Injury",
                "PFC = Prolonged Field Care",
                "POI = Point of Injury",
                "TACEVAC = Tactical Evacuation",
                "WB = Whole Blood",
                "WBTR = Whole Blood Transfusion Record",
                "",
                "# Common medical terms",
                "hemorrhage -> bleeding, blood loss, hemorrhaging, blood",
                "tourniquet -> TQ, CAT, SOFT-T, tourniquet application",
                "casualty -> patient, victim, injured, wounded, casualty assessment",
                "airway -> airway management, breathing passage, airway obstruction, airway control",
                "respiration -> breathing, ventilation, respiratory rate, breaths per minute",
                "circulation -> blood flow, pulse, circulation assessment, circulatory",
                "junctional -> groin, axilla, junction, junctional hemorrhage",
                "trauma -> injury, wound, traumatic injury, physical trauma",
                "hypothermia -> low body temperature, cold exposure, hypothermic",
                "triage -> sorting, prioritization, triage assessment, triage category",
                "occlusive -> chest seal, occlusive dressing, wound seal, airtight",
                "hemostatic -> clotting, blood-stopping, QuikClot, hemostatic agent",
                "tension pneumothorax -> collapsed lung, chest injury, decompression",
                "needle decompression -> chest decompression, thoracic decompression",
                "splint -> immobilize, stabilize, splinting, splinted",
                "pressure dressing -> compression bandage, pressure bandage, wound pressure",
                "fracture -> broken bone, bone injury, fractured, bone fracture"
            ]
            
            with open(self.vocab_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(default_terms))
                
            logger.info(f"Created default vocabulary file at {self.vocab_path}")
            
        except Exception as e:
            logger.error(f"Failed to create default vocabulary: {str(e)}")
            raise
    
    def expand_query(self, query: str) -> List[str]:
        """
        Expand a query with medical terminology.
        
        Args:
            query: Query text
            
        Returns:
            List of expanded queries
        """
        if not self.initialized:
            logger.warning("Medical vocabulary not initialized")
            return [query]
        
        expanded_queries = [query]
        
        # Extract words from query
        words = re.findall(r'\b\w+\b', query.lower())
        
        # Check for abbreviations
        for word in words:
            if word.upper() in self.abbreviations:
                # Create new query with expanded abbreviation
                expanded = query.replace(word, self.abbreviations[word.upper()])
                expanded_queries.append(expanded)
        
        # Check for synonym expansions
        for word in words:
            if word.lower() in self.synonyms:
                for synonym in self.synonyms[word.lower()]:
                    # Create new query with synonym
                    expanded = query.replace(word, synonym)
                    expanded_queries.append(expanded)
        
        # Deduplicate
        expanded_queries = list(set(expanded_queries))
        
        logger.debug(f"Expanded query '{query}' to {len(expanded_queries)} variants")
        return expanded_queries
    
    def get_term_info(self, term: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a medical term.
        
        Args:
            term: Medical term
            
        Returns:
            Dictionary with term information or None if not found
        """
        if not self.initialized:
            logger.warning("Medical vocabulary not initialized")
            return None
        
        # Check if term is in dictionary
        term_lower = term.lower()
        term_upper = term.upper()
        
        if term_lower in self.terms_dict:
            value = self.terms_dict[term_lower]
            if isinstance(value, list):
                return {
                    "term": term,
                    "type": "synonym",
                    "synonyms": value
                }
            return {
                "term": term,
                "type": "standard",
                "definition": value
            }
        elif term_upper in self.abbreviations:
            return {
                "term": term_upper,
                "type": "abbreviation",
                "expansion": self.abbreviations[term_upper]
            }
        
        return None
    
    def extract_medical_terms(self, text: str) -> List[str]:
        """
        Extract known medical terms from text.
        
        Args:
            text: Input text
            
        Returns:
            List of recognized medical terms
        """
        if not self.initialized:
            logger.warning("Medical vocabulary not initialized")
            return []
        
        # Extract words from text
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Match against known terms
        medical_terms = []
        
        for word in words:
            word_lower = word.lower()
            word_upper = word.upper()
            
            if word_lower in self.terms_dict or word_upper in self.abbreviations:
                medical_terms.append(word)
                
        # Also check for multi-word terms
        for term in self.terms_dict:
            if ' ' in term and term.lower() in text.lower():
                medical_terms.append(term)
        
        # Deduplicate
        medical_terms = list(set(medical_terms))
        
        return medical_terms
    
    def explain_medical_terms(self, text: str) -> Dict[str, str]:
        """
        Provide explanations for medical terms in text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping terms to explanations
        """
        if not self.initialized:
            logger.warning("Medical vocabulary not initialized")
            return {}
        
        # Extract medical terms
        terms = self.extract_medical_terms(text)
        
        # Generate explanations
        explanations = {}
        
        for term in terms:
            info = self.get_term_info(term)
            if info:
                if info["type"] == "abbreviation":
                    explanations[term] = f"{term} - {info['expansion']}"
                elif info["type"] == "synonym":
                    explanations[term] = f"{term} - related terms: {', '.join(info['synonyms'])}"
                else:
                    explanations[term] = f"{term} - {info.get('definition', '')}"
        
        return explanations