"""
Medical Vocabulary Manager for the TCCC.ai Document Library.

This module provides enhanced medical terminology support for the Document Library,
including synonym expansion, abbreviation handling, and domain-specific
knowledge enhancement. Optimized for TCCC-relevant medical terminology.
"""

import os
import re
import json
import sqlite3
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from pathlib import Path

from tccc.utils.logging import get_logger

logger = get_logger(__name__)

class MedicalVocabularyManager:
    """
    Enhanced Medical Vocabulary Manager for Document Library.
    
    This class provides:
    - Medical term synonym expansion
    - Abbreviation handling (e.g., TCCC, MARCH, MIST)
    - Military-specific medical terminology
    - Query enhancement with domain knowledge
    - Term correction and fuzzy matching
    - Optimized for Jetson hardware with memory-efficient storage
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Medical Vocabulary Manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Primary medical terms path
        self.vocab_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "config/vocabulary/custom_terms.txt"
        )
        
        # SQLite database path for efficient storage
        db_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "data/vocabulary"
        )
        self.db_path = os.path.join(db_dir, "medical_terms.db")
        
        # Ensure the directory exists
        os.makedirs(db_dir, exist_ok=True)
        
        # Preparation for specialized vocabularies
        self.specialized_vocab_paths = {
            "tccc": os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                "config/vocabulary/tccc_terms.txt"
            ),
            "trauma": os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                "config/vocabulary/trauma_terms.txt"
            ),
        }
        
        # Data structures
        self.terms_dict = {}
        self.abbreviations = {}
        self.synonyms = {}
        self.corrections = {}
        self.category_terms = {}
        self.initialized = False
        
        # Specify important medical categories
        self.medical_categories = [
            "hemorrhage", "airway", "respiration", "circulation", 
            "shock", "head injury", "burns", "fracture", "medication",
            "procedure", "treatment", "assessment", "evacuation"
        ]
    
    def initialize(self) -> bool:
        """
        Initialize the vocabulary from files and build the database.
        
        Returns:
            Success status
        """
        try:
            # Create or open the SQLite database
            self._init_database()
            
            # Check if vocabulary file exists
            if not os.path.exists(self.vocab_path):
                logger.warning(f"Vocabulary file not found: {self.vocab_path}")
                self._create_default_vocabulary()
            
            # Load vocabulary from main file and specialized files
            self._load_vocabulary()
            
            # Build category index
            self._build_category_index()
            
            self.initialized = True
            logger.info(f"Medical vocabulary initialized with {len(self.terms_dict)} terms and {len(self.abbreviations)} abbreviations")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize medical vocabulary: {str(e)}")
            return False
    
    def _init_database(self) -> None:
        """Initialize the SQLite database for term storage."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create terms table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS terms (
                id INTEGER PRIMARY KEY,
                term TEXT UNIQUE,
                type TEXT,
                definition TEXT,
                category TEXT
            )
            ''')
            
            # Create synonyms table with foreign key to terms
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS synonyms (
                id INTEGER PRIMARY KEY,
                term_id INTEGER,
                synonym TEXT,
                FOREIGN KEY (term_id) REFERENCES terms(id)
            )
            ''')
            
            # Create abbreviations table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS abbreviations (
                id INTEGER PRIMARY KEY,
                abbreviation TEXT UNIQUE,
                expansion TEXT
            )
            ''')
            
            # Create corrections table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS corrections (
                id INTEGER PRIMARY KEY,
                misspelled TEXT UNIQUE,
                corrected TEXT
            )
            ''')
            
            # Create indices for faster lookups
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_term ON terms(term)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_synonym ON synonyms(synonym)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_abbreviation ON abbreviations(abbreviation)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_misspelled ON corrections(misspelled)")
            
            conn.commit()
            conn.close()
            
            logger.debug("Medical vocabulary database initialized")
        except Exception as e:
            logger.error(f"Failed to initialize medical vocabulary database: {str(e)}")
            raise
    
    def _load_vocabulary(self) -> None:
        """Load vocabulary from main file and specialized files."""
        try:
            # Load from main vocabulary file
            self._load_from_file(self.vocab_path)
            
            # Load specialized vocabularies if they exist
            for vocab_type, vocab_path in self.specialized_vocab_paths.items():
                if os.path.exists(vocab_path):
                    self._load_from_file(vocab_path, category=vocab_type)
                else:
                    logger.debug(f"Specialized vocabulary not found: {vocab_path}")
            
            # Populate the database with loaded terms
            self._populate_database()
                
        except Exception as e:
            logger.error(f"Failed to load vocabulary: {str(e)}")
            raise
    
    def _load_from_file(self, file_path: str, category: str = None) -> None:
        """
        Load vocabulary from a file.
        
        Args:
            file_path: Path to vocabulary file
            category: Optional category for the terms
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Track current category from section headers
            current_category = category
            
            # Process each line
            for line in lines:
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Check for category headers [Category: X]
                if line.startswith('[Category:') and line.endswith(']'):
                    current_category = line[10:-1].strip().lower()
                    continue
                
                # Handle abbreviations (ABBR = Full Form)
                if " = " in line:
                    abbr, full_form = [part.strip() for part in line.split(" = ", 1)]
                    self.abbreviations[abbr.upper()] = full_form
                    self.terms_dict[abbr.upper()] = {
                        "type": "abbreviation",
                        "definition": full_form,
                        "category": current_category
                    }
                    continue
                    
                # Handle synonyms (term -> synonym1, synonym2, ...)
                if " -> " in line:
                    term, synonyms_str = [part.strip() for part in line.split(" -> ", 1)]
                    synonyms = [s.strip() for s in synonyms_str.split(",")]
                    self.synonyms[term.lower()] = synonyms
                    self.terms_dict[term.lower()] = {
                        "type": "term",
                        "synonyms": synonyms,
                        "category": current_category
                    }
                    continue
                
                # Handle corrections (misspelled => correct)
                if " => " in line:
                    misspelled, correct = [part.strip() for part in line.split(" => ", 1)]
                    self.corrections[misspelled.lower()] = correct.lower()
                    continue
                    
                # Add standard term
                self.terms_dict[line.lower()] = {
                    "type": "term",
                    "definition": line,
                    "category": current_category
                }
                
        except Exception as e:
            logger.error(f"Failed to load vocabulary file {file_path}: {str(e)}")
            raise
    
    def _populate_database(self) -> None:
        """Populate the SQLite database with loaded terms."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Clear existing tables for a clean load
            cursor.execute("DELETE FROM synonyms")
            cursor.execute("DELETE FROM abbreviations")
            cursor.execute("DELETE FROM corrections")
            cursor.execute("DELETE FROM terms")
            
            # Add terms
            for term, data in self.terms_dict.items():
                term_type = data.get("type", "term")
                definition = data.get("definition", "")
                category = data.get("category", "")
                
                cursor.execute(
                    "INSERT INTO terms (term, type, definition, category) VALUES (?, ?, ?, ?)",
                    (term, term_type, definition, category)
                )
                
                # Get the term_id for synonyms
                if term_type == "term" and "synonyms" in data:
                    term_id = cursor.lastrowid
                    
                    for synonym in data["synonyms"]:
                        cursor.execute(
                            "INSERT INTO synonyms (term_id, synonym) VALUES (?, ?)",
                            (term_id, synonym)
                        )
            
            # Add abbreviations
            for abbr, expansion in self.abbreviations.items():
                cursor.execute(
                    "INSERT INTO abbreviations (abbreviation, expansion) VALUES (?, ?)",
                    (abbr, expansion)
                )
            
            # Add corrections
            for misspelled, corrected in self.corrections.items():
                cursor.execute(
                    "INSERT INTO corrections (misspelled, corrected) VALUES (?, ?)",
                    (misspelled, corrected)
                )
            
            conn.commit()
            conn.close()
            
            logger.info(f"Populated medical vocabulary database with {len(self.terms_dict)} terms")
        except Exception as e:
            logger.error(f"Failed to populate medical vocabulary database: {str(e)}")
            raise
    
    def _build_category_index(self) -> None:
        """Build index of terms by category for faster access."""
        try:
            # Prepare category dictionary
            self.category_terms = {category: [] for category in self.medical_categories}
            
            # Populate from terms_dict
            for term, data in self.terms_dict.items():
                category = data.get("category")
                if category in self.category_terms:
                    self.category_terms[category].append(term)
            
            # Log results
            for category, terms in self.category_terms.items():
                if terms:
                    logger.debug(f"Category '{category}' has {len(terms)} terms")
        except Exception as e:
            logger.error(f"Failed to build category index: {str(e)}")
    
    def _create_default_vocabulary(self) -> None:
        """Create default vocabulary files if they don't exist."""
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
                "# 4. Corrections: misspelled => correct",
                "# 5. Categories: [Category: name]",
                "",
                "[Category: general]",
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
                "[Category: hemorrhage]",
                "hemorrhage -> bleeding, blood loss, hemorrhaging, blood",
                "tourniquet -> TQ, CAT, SOFT-T, tourniquet application",
                "hemostatic -> clotting, blood-stopping, QuikClot, hemostatic agent",
                "pressure dressing -> compression bandage, pressure bandage, wound pressure",
                "junctional -> groin, axilla, junction, junctional hemorrhage",
                "tourniquet malfunction => tourniquet failure",
                "",
                "[Category: airway]",
                "airway -> airway management, breathing passage, airway obstruction, airway control",
                "nasopharyngeal airway -> NPA, nasal airway, nose airway",
                "oropharyngeal airway -> OPA, oral airway, mouth airway",
                "",
                "[Category: respiration]",
                "respiration -> breathing, ventilation, respiratory rate, breaths per minute",
                "tension pneumothorax -> collapsed lung, chest injury, decompression",
                "needle decompression -> chest decompression, thoracic decompression",
                "occlusive -> chest seal, occlusive dressing, wound seal, airtight",
                "",
                "[Category: circulation]",
                "circulation -> blood flow, pulse, circulation assessment, circulatory",
                "shock -> hypovolemic shock, bleeding shock, hemorrhagic shock",
                "",
                "[Category: medical]",
                "casualty -> patient, victim, injured, wounded, casualty assessment",
                "trauma -> injury, wound, traumatic injury, physical trauma",
                "hypothermia -> low body temperature, cold exposure, hypothermic",
                "triage -> sorting, prioritization, triage assessment, triage category",
                "splint -> immobilize, stabilize, splinting, splinted",
                "fracture -> broken bone, bone injury, fractured, bone fracture"
            ]
            
            # Write main vocabulary file
            with open(self.vocab_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(default_terms))
                
            logger.info(f"Created default vocabulary file at {self.vocab_path}")
            
            # Create specialized vocabulary files
            for vocab_type, vocab_path in self.specialized_vocab_paths.items():
                if not os.path.exists(vocab_path):
                    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
                    
                    # Create minimal content for the specialized files
                    with open(vocab_path, 'w', encoding='utf-8') as f:
                        f.write(f"# TCCC.ai Specialized {vocab_type.capitalize()} Vocabulary\n")
                        f.write("# Add specialized terms below\n\n")
                        
                    logger.info(f"Created specialized vocabulary file at {vocab_path}")
                
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
        
        # Extract words and phrases from query
        words = re.findall(r'\b\w+\b', query.lower())
        phrases = self._extract_phrases(query)
        
        # Check for corrections first
        corrected_query = self._apply_corrections(query)
        if corrected_query != query:
            expanded_queries.append(corrected_query)
        
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
        
        # Check phrase expansions
        for phrase in phrases:
            phrase_lower = phrase.lower()
            if phrase_lower in self.synonyms:
                for synonym in self.synonyms[phrase_lower]:
                    # Create new query with phrase synonym
                    expanded = query.replace(phrase, synonym)
                    expanded_queries.append(expanded)
        
        # Add category-specific expansion for medical queries
        category = self._detect_medical_category(query)
        if category:
            category_terms = self.category_terms.get(category, [])
            if category_terms:
                # Add category-specific expansion
                expanded_queries.append(f"{query} {category}")
                
                # Select a highly relevant term from the category
                for term in category_terms[:2]:  # Use just a couple of terms
                    if term not in query.lower():
                        expanded_queries.append(f"{query} {term}")
        
        # Deduplicate and limit expansions
        expanded_queries = list(dict.fromkeys(expanded_queries))  # Preserve order
        
        # Limit to reasonable number of expansions to prevent query explosion
        max_expansions = 5
        if len(expanded_queries) > max_expansions:
            expanded_queries = expanded_queries[:max_expansions]
        
        logger.debug(f"Expanded query '{query}' to {len(expanded_queries)} variants")
        return expanded_queries
    
    def _extract_phrases(self, text: str) -> List[str]:
        """
        Extract potential multi-word phrases from text.
        
        Args:
            text: Input text
            
        Returns:
            List of phrases
        """
        phrases = []
        
        # Look for 2 and 3-word phrases
        words = text.split()
        
        # Extract 2-word phrases
        for i in range(len(words) - 1):
            phrase = f"{words[i]} {words[i+1]}"
            phrases.append(phrase)
        
        # Extract 3-word phrases
        for i in range(len(words) - 2):
            phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
            phrases.append(phrase)
        
        return phrases
    
    def _apply_corrections(self, text: str) -> str:
        """
        Apply spelling corrections to text.
        
        Args:
            text: Input text
            
        Returns:
            Corrected text
        """
        if not self.corrections:
            return text
        
        words = re.findall(r'\b\w+\b', text.lower())
        result = text
        
        for word in words:
            if word in self.corrections:
                # Replace word with correction
                pattern = r'\b' + re.escape(word) + r'\b'
                result = re.sub(pattern, self.corrections[word], result, flags=re.IGNORECASE)
        
        return result
    
    def _detect_medical_category(self, text: str) -> Optional[str]:
        """
        Detect the likely medical category of a query.
        
        Args:
            text: Input text
            
        Returns:
            Category name or None
        """
        text_lower = text.lower()
        
        # Check each category
        category_scores = {}
        
        for category in self.medical_categories:
            # Simple scoring: presence of category name is strong signal
            if category in text_lower:
                category_scores[category] = 10
                continue
                
            # Check category-specific terms
            score = 0
            category_terms = self.category_terms.get(category, [])
            
            for term in category_terms:
                if term in text_lower:
                    score += 2
                # Check for partial matches in longer terms
                elif len(term) > 5 and term[:5] in text_lower:
                    score += 1
            
            if score > 0:
                category_scores[category] = score
        
        # Return highest-scoring category if any
        if category_scores:
            return max(category_scores.items(), key=lambda x: x[1])[0]
        
        return None
    
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
        
        try:
            # Check SQLite database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check for term in terms table
            cursor.execute(
                "SELECT id, type, definition, category FROM terms WHERE term = ? COLLATE NOCASE",
                (term,)
            )
            
            row = cursor.fetchone()
            if row:
                term_id, term_type, definition, category = row
                
                # If it's a term with synonyms, fetch those
                if term_type == "term":
                    cursor.execute(
                        "SELECT synonym FROM synonyms WHERE term_id = ?",
                        (term_id,)
                    )
                    synonyms = [row[0] for row in cursor.fetchall()]
                    
                    result = {
                        "term": term,
                        "type": term_type,
                        "definition": definition,
                        "category": category,
                        "synonyms": synonyms
                    }
                else:
                    # For abbreviations or other types
                    result = {
                        "term": term,
                        "type": term_type,
                        "definition": definition,
                        "category": category
                    }
                
                conn.close()
                return result
            
            # Check for abbreviation
            cursor.execute(
                "SELECT expansion FROM abbreviations WHERE abbreviation = ? COLLATE NOCASE",
                (term,)
            )
            
            row = cursor.fetchone()
            if row:
                expansion = row[0]
                
                result = {
                    "term": term,
                    "type": "abbreviation",
                    "expansion": expansion
                }
                
                conn.close()
                return result
            
            conn.close()
            return None
        
        except Exception as e:
            logger.error(f"Error getting term info: {str(e)}")
            return None
        
        # Fallback to in-memory dictionaries if database query fails
        term_lower = term.lower()
        term_upper = term.upper()
        
        if term_lower in self.terms_dict:
            return {
                "term": term,
                "type": self.terms_dict[term_lower].get("type", "term"),
                "definition": self.terms_dict[term_lower].get("definition", ""),
                "category": self.terms_dict[term_lower].get("category", ""),
                "synonyms": self.terms_dict[term_lower].get("synonyms", [])
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
        
        # Extract words and phrases
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Match against known terms
        medical_terms = []
        
        # Check words
        for word in words:
            word_lower = word.lower()
            word_upper = word.upper()
            
            if word_lower in self.terms_dict or word_upper in self.abbreviations:
                medical_terms.append(word)
        
        # Check multi-word phrases
        for term in self.terms_dict:
            if ' ' in term and term.lower() in text.lower():
                medical_terms.append(term)
        
        # Check phrases against database
        phrases = self._extract_phrases(text)
        for phrase in phrases:
            if self.get_term_info(phrase):
                medical_terms.append(phrase)
        
        # Deduplicate and normalize case
        normalized_terms = []
        seen = set()
        
        for term in medical_terms:
            term_lower = term.lower()
            if term_lower not in seen:
                seen.add(term_lower)
                # Use original case from term dictionary if possible
                for dict_term in self.terms_dict:
                    if dict_term.lower() == term_lower:
                        normalized_terms.append(dict_term)
                        break
                else:
                    normalized_terms.append(term)
        
        return normalized_terms
    
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
                    explanations[term] = f"{term} - {info.get('expansion', '')}"
                elif "synonyms" in info and info["synonyms"]:
                    explanations[term] = f"{term} - related terms: {', '.join(info['synonyms'])}"
                elif "definition" in info and info["definition"]:
                    explanations[term] = f"{term} - {info['definition']}"
        
        return explanations
    
    def get_category_terms(self, category: str) -> List[str]:
        """
        Get terms for a specific medical category.
        
        Args:
            category: Medical category
            
        Returns:
            List of terms in that category
        """
        if not self.initialized:
            logger.warning("Medical vocabulary not initialized")
            return []
        
        return self.category_terms.get(category.lower(), [])