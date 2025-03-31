"""
Field Extractor module for TCCC.ai Form Generator.

This module extracts field values for military medical forms from 
audio transcriptions and LLM analysis results.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dateutil import parser as date_parser

# Import utilities
from tccc.utils.logging import get_logger

logger = get_logger(__name__)


class FieldExtractor:
    """
    Extract form fields from medical event data.
    
    This class provides methods to extract structured form fields from
    medical events extracted by the LLM Analysis module.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the field extractor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
    
    def extract_field(self, events: List[Dict[str, Any]], field_name: str, 
                    extraction_rules: Dict[str, Any]) -> Tuple[Any, float]:
        """Extract a single field value from medical events.
        
        Args:
            events: Medical event data
            field_name: Name of the field to extract
            extraction_rules: Rules for extracting this field
            
        Returns:
            Tuple of (extracted_value, confidence)
        """
        method = extraction_rules.get("method", "direct")
        
        if method == "direct":
            return self._extract_direct(events, field_name, extraction_rules)
        elif method == "pattern":
            return self._extract_pattern(events, field_name, extraction_rules)
        elif method == "derived":
            return self._extract_derived(events, field_name, extraction_rules)
        else:
            logger.warning(f"Unknown extraction method: {method}")
            return None, 0.0
    
    def extract_fields(self, events: List[Dict[str, Any]], 
                      field_definitions: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Extract multiple field values from medical events.
        
        Args:
            events: Medical event data
            field_definitions: Definitions and extraction rules for fields
            
        Returns:
            Dictionary of field values and metadata
        """
        result = {}
        
        for field_name, field_def in field_definitions.items():
            value, confidence = self.extract_field(events, field_name, field_def)
            
            result[field_name] = {
                "value": value,
                "confidence": confidence,
                "source": field_def.get("source", "derived")
            }
            
            # Add metadata if available
            if "metadata" in field_def:
                result[field_name]["metadata"] = field_def["metadata"]
        
        return result
    
    def _extract_direct(self, events: List[Dict[str, Any]], field_name: str, 
                      rules: Dict[str, Any]) -> Tuple[Any, float]:
        """Extract a field directly from event properties.
        
        Args:
            events: Medical event data
            field_name: Name of the field to extract
            rules: Rules for extracting this field
            
        Returns:
            Tuple of (extracted_value, confidence)
        """
        # Get properties to look for
        properties = rules.get("properties", [field_name])
        categories = rules.get("categories", [])
        
        best_value = None
        best_confidence = 0.0
        
        for event in events:
            # Filter by category if specified
            if categories and event.get("category") not in categories:
                continue
            
            # Check each property
            for prop in properties:
                if prop in event and event[prop]:
                    confidence = self._calculate_confidence(event, rules)
                    
                    if confidence > best_confidence:
                        best_value = event[prop]
                        best_confidence = confidence
        
        return best_value, best_confidence
    
    def _extract_pattern(self, events: List[Dict[str, Any]], field_name: str, 
                       rules: Dict[str, Any]) -> Tuple[Any, float]:
        """Extract a field using regex patterns.
        
        Args:
            events: Medical event data
            field_name: Name of the field to extract
            rules: Rules for extracting this field
            
        Returns:
            Tuple of (extracted_value, confidence)
        """
        patterns = rules.get("patterns", [])
        if not patterns:
            return None, 0.0
        
        # Get text sources to search in
        text_props = rules.get("text_properties", ["text", "value"])
        categories = rules.get("categories", [])
        
        best_value = None
        best_confidence = 0.0
        
        for event in events:
            # Filter by category if specified
            if categories and event.get("category") not in categories:
                continue
            
            # Try each text property
            for prop in text_props:
                if prop not in event or not event[prop]:
                    continue
                
                text = str(event[prop])
                
                # Try each pattern
                for pattern_info in patterns:
                    pattern = pattern_info.get("pattern", "")
                    group = pattern_info.get("group", 0)
                    
                    try:
                        match = re.search(pattern, text, re.IGNORECASE)
                        if match:
                            value = match.group(group)
                            
                            # Apply post-processing if defined
                            if "process" in pattern_info:
                                value = self._apply_processing(value, pattern_info["process"])
                            
                            confidence = pattern_info.get("confidence", 0.7)
                            
                            if confidence > best_confidence:
                                best_value = value
                                best_confidence = confidence
                    except re.error:
                        logger.warning(f"Invalid regex pattern: {pattern}")
        
        return best_value, best_confidence
    
    def _extract_derived(self, events: List[Dict[str, Any]], field_name: str, 
                       rules: Dict[str, Any]) -> Tuple[Any, float]:
        """Extract a field derived from other events or fields.
        
        Args:
            events: Medical event data
            field_name: Name of the field to extract
            rules: Rules for extracting this field
            
        Returns:
            Tuple of (extracted_value, confidence)
        """
        derivation = rules.get("derivation", "")
        
        if derivation == "latest_vital_sign":
            return self._extract_latest_vital_sign(events, rules)
        elif derivation == "combine_values":
            return self._combine_values(events, rules)
        elif derivation == "datetime":
            return self._extract_datetime(events, rules)
        else:
            logger.warning(f"Unknown derivation method: {derivation}")
            return None, 0.0
    
    def _extract_latest_vital_sign(self, events: List[Dict[str, Any]], 
                                 rules: Dict[str, Any]) -> Tuple[Any, float]:
        """Extract the latest value for a vital sign.
        
        Args:
            events: Medical event data
            rules: Rules for extraction
            
        Returns:
            Tuple of (extracted_value, confidence)
        """
        vital_type = rules.get("vital_type", "")
        if not vital_type:
            return None, 0.0
        
        # Look for vitals
        latest_value = None
        latest_time = None
        confidence = 0.0
        
        for event in events:
            if event.get("category") == "vitals" and event.get("type") == vital_type:
                event_time = event.get("time", "")
                event_value = event.get("value", "")
                
                if not event_value:
                    continue
                
                # If we don't have a value yet, use this one
                if latest_value is None:
                    latest_value = event_value
                    latest_time = event_time
                    confidence = self._calculate_confidence(event, rules)
                    continue
                
                # Otherwise, only use if it's newer
                if event_time and latest_time:
                    # Try to compare times
                    try:
                        if self._compare_times(event_time, latest_time) > 0:
                            latest_value = event_value
                            latest_time = event_time
                            confidence = self._calculate_confidence(event, rules)
                    except:
                        # If time comparison fails, just use event with highest confidence
                        new_confidence = self._calculate_confidence(event, rules)
                        if new_confidence > confidence:
                            latest_value = event_value
                            latest_time = event_time
                            confidence = new_confidence
        
        return latest_value, confidence
    
    def _combine_values(self, events: List[Dict[str, Any]], rules: Dict[str, Any]) -> Tuple[Any, float]:
        """Combine values from multiple events.
        
        Args:
            events: Medical event data
            rules: Rules for combination
            
        Returns:
            Tuple of (combined_value, confidence)
        """
        fields = rules.get("fields", [])
        separator = rules.get("separator", " ")
        templates = rules.get("templates", {})
        
        values = {}
        confidences = {}
        
        # Extract each field
        for field in fields:
            field_rules = rules.get("field_rules", {}).get(field, {})
            value, confidence = self.extract_field(events, field, field_rules)
            
            if value is not None:
                values[field] = value
                confidences[field] = confidence
        
        # If we have a specific template for the combination of available fields,
        # use that template
        available_fields = set(values.keys())
        template = None
        
        # Find the best matching template
        best_match_size = 0
        for template_fields, template_str in templates.items():
            template_fields_set = set(template_fields.split(","))
            if template_fields_set.issubset(available_fields) and len(template_fields_set) > best_match_size:
                template = template_str
                best_match_size = len(template_fields_set)
        
        # If we found a template, use it to format the result
        if template:
            try:
                result = template.format(**values)
                avg_confidence = sum(confidences.values()) / len(confidences) if confidences else 0.0
                return result, avg_confidence
            except KeyError as e:
                logger.warning(f"Missing field in template: {e}")
        
        # Otherwise, just join the values with the separator
        if values:
            result = separator.join(str(v) for v in values.values())
            avg_confidence = sum(confidences.values()) / len(confidences) if confidences else 0.0
            return result, avg_confidence
        
        return None, 0.0
    
    def _extract_datetime(self, events: List[Dict[str, Any]], rules: Dict[str, Any]) -> Tuple[Any, float]:
        """Extract a datetime from events.
        
        Args:
            events: Medical event data
            rules: Rules for extraction
            
        Returns:
            Tuple of (datetime_string, confidence)
        """
        properties = rules.get("properties", ["timestamp", "time", "date"])
        categories = rules.get("categories", ["temporal"])
        format_str = rules.get("format", "%Y-%m-%d %H:%M:%S")
        
        best_datetime = None
        best_confidence = 0.0
        
        for event in events:
            # Filter by category if specified
            if categories and event.get("category") not in categories:
                continue
            
            # Check each property
            for prop in properties:
                if prop in event and event[prop]:
                    value = event[prop]
                    
                    try:
                        # Parse datetime string
                        dt = date_parser.parse(value)
                        formatted_dt = dt.strftime(format_str)
                        
                        confidence = self._calculate_confidence(event, rules)
                        if confidence > best_confidence:
                            best_datetime = formatted_dt
                            best_confidence = confidence
                    except:
                        # Not a valid datetime
                        pass
        
        return best_datetime, best_confidence
    
    def _calculate_confidence(self, event: Dict[str, Any], rules: Dict[str, Any]) -> float:
        """Calculate confidence for an extraction.
        
        Args:
            event: Event containing the value
            rules: Extraction rules
            
        Returns:
            Confidence value (0.0-1.0)
        """
        # Base confidence from rules
        base_confidence = rules.get("base_confidence", 0.7)
        
        # Adjust based on event confidence if available
        if "confidence" in event:
            if isinstance(event["confidence"], str):
                if event["confidence"].lower() == "high":
                    return max(base_confidence, 0.8)
                elif event["confidence"].lower() == "medium":
                    return max(base_confidence, 0.5)
                elif event["confidence"].lower() == "low":
                    return min(base_confidence, 0.3)
            else:
                # Assume numeric confidence
                try:
                    return float(event["confidence"])
                except:
                    pass
        
        return base_confidence
    
    def _apply_processing(self, value: str, process_type: str) -> Any:
        """Apply post-processing to an extracted value.
        
        Args:
            value: The extracted value
            process_type: Type of processing to apply
            
        Returns:
            Processed value
        """
        if process_type == "int":
            try:
                return int(value)
            except:
                return None
        elif process_type == "float":
            try:
                return float(value)
            except:
                return None
        elif process_type == "boolean":
            return value.lower() in ["true", "yes", "y", "1"]
        elif process_type == "trim":
            return value.strip()
        elif process_type == "uppercase":
            return value.upper()
        elif process_type == "lowercase":
            return value.lower()
        else:
            # No processing
            return value
    
    def _compare_times(self, time1: str, time2: str) -> int:
        """Compare two time strings.
        
        Args:
            time1: First time string
            time2: Second time string
            
        Returns:
            1 if time1 > time2, -1 if time1 < time2, 0 if equal
        """
        try:
            dt1 = date_parser.parse(time1)
            dt2 = date_parser.parse(time2)
            
            if dt1 > dt2:
                return 1
            elif dt1 < dt2:
                return -1
            else:
                return 0
        except:
            # If parsing fails, compare as strings
            if time1 > time2:
                return 1
            elif time1 < time2:
                return -1
            else:
                return 0