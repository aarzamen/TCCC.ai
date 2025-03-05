"""
Response Generator for the TCCC.ai Document Library.

This module handles integration of retrieved documents with the LLM,
optimizing the way context is provided to generate accurate responses.
"""

import re
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Union

from tccc.utils.logging import get_logger
from tccc.document_library.query_engine import QueryEngine, QueryStrategy

logger = get_logger(__name__)

class ResponseGenerator:
    """
    Response Generator for Document Library.
    
    This class provides:
    - Integration with the LLM
    - Optimized context injection
    - Template-based prompt formatting
    - Citation tracking
    - Response quality validation
    - Context length management for API constraints
    """
    
    def __init__(self, config: Dict[str, Any], query_engine: QueryEngine):
        """
        Initialize the Response Generator.
        
        Args:
            config: Configuration dictionary
            query_engine: Query Engine instance
        """
        self.config = config
        self.query_engine = query_engine
        # Default context length - will be loaded from config or overridden by LLM config
        self.max_context_length = config.get("llm", {}).get("max_context_length", 1500)
        self.default_prompt_template = self._get_default_prompt_template()
        self.specialized_templates = self._load_specialized_templates()
        self.adaptive_context_sizing = True  # Enable automatic context length adjustment
    
    def _get_default_prompt_template(self) -> str:
        """
        Get the default prompt template.
        
        Returns:
            Default prompt template string
        """
        return """
You are TCCC.ai, an expert in Tactical Combat Casualty Care (TCCC) and military medicine.
Your task is to provide accurate information based on official military medical protocols.

CONTEXT INFORMATION:
{context}

USER QUERY:
{query}

Please answer the query based on the provided context information. If the context 
doesn't contain relevant information, acknowledge the limitations and suggest what 
the user should refer to instead. Include citations where appropriate using the format 
[Source: document title].
"""

    def _load_specialized_templates(self) -> Dict[str, str]:
        """
        Load specialized prompt templates for different query types.
        
        Returns:
            Dictionary of template names to template strings
        """
        return {
            "procedure": """
You are TCCC.ai, an expert in Tactical Combat Casualty Care (TCCC) and military medicine.
Your task is to explain a medical procedure based on official protocols.

CONTEXT INFORMATION:
{context}

USER QUERY ABOUT PROCEDURE:
{query}

Please provide a clear, step-by-step explanation of the procedure based on the provided context.
Format your response with numbered steps. Include any cautions or warnings.
Cite your sources using the format [Source: document title].
""",
            "explanation": """
You are TCCC.ai, an expert in Tactical Combat Casualty Care (TCCC) and military medicine.
Your task is to explain medical concepts based on official protocols.

CONTEXT INFORMATION:
{context}

USER QUERY REQUESTING EXPLANATION:
{query}

Please provide a clear explanation of the concept based on the provided context.
Use simple language and provide examples where helpful. Define any technical terms.
Cite your sources using the format [Source: document title].
""",
            "form": """
You are TCCC.ai, an expert in Tactical Combat Casualty Care (TCCC) and military medicine.
Your task is to provide guidance on completing a military medical form.

CONTEXT INFORMATION:
{context}

USER QUERY ABOUT FORM:
{query}

Please provide specific guidance on completing the form based on the provided context.
Explain each field that needs to be completed and what information should be entered.
Cite your sources using the format [Source: document title].
"""
        }
    
    def set_max_context_length(self, max_length: int) -> None:
        """
        Set the maximum context length for the LLM.
        
        Args:
            max_length: Maximum context length in tokens/characters
        """
        self.max_context_length = max_length
        logger.info(f"Maximum context length set to {max_length}")
    
    def _detect_query_type(self, query: str) -> str:
        """
        Detect the type of query to select appropriate template.
        
        Args:
            query: User query
            
        Returns:
            Query type string
        """
        query_lower = query.lower()
        
        # Check for procedure questions
        procedure_patterns = [
            r"how (to|do|should) .* (apply|perform|conduct|do|use)",
            r"steps (for|to|in) .*",
            r"procedure for .*",
            r"what is the (process|protocol|technique) for .*"
        ]
        for pattern in procedure_patterns:
            if re.search(pattern, query_lower):
                return "procedure"
        
        # Check for explanation questions
        explanation_patterns = [
            r"what (is|are) .*",
            r"explain .*",
            r"define .*",
            r"describe .*",
            r"meaning of .*"
        ]
        for pattern in explanation_patterns:
            if re.search(pattern, query_lower):
                return "explanation"
        
        # Check for form questions
        form_patterns = [
            r"how (to|do|should) .* (fill|complete|document|record|report) .*",
            r".* (form|card|report|documentation) .*",
            r"DD Form .*",
            r"TCCC card .*"
        ]
        for pattern in form_patterns:
            if re.search(pattern, query_lower):
                return "form"
        
        # Default to standard if no specific type detected
        return "standard"
    
    def _format_context(self, results: List[Dict[str, Any]], max_length: int) -> str:
        """
        Format search results into context for the LLM.
        
        Args:
            results: Search results
            max_length: Maximum context length
            
        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant information found in the document library."
        
        context_parts = []
        current_length = 0
        skipped_docs = 0
        truncated_docs = 0
        total_docs = len(results)
        
        # Calculate dynamic max length if adaptive sizing is enabled
        adjusted_max_length = max_length
        if self.adaptive_context_sizing:
            # Estimate template overhead (prompt structure, user query, instructions)
            template_overhead = 500
            # Adjust max length to account for template overhead
            adjusted_max_length = max(500, max_length - template_overhead)
            logger.debug(f"Adjusted max context length: {adjusted_max_length} characters")
        
        for result in results:
            # Extract information
            doc_id = result.get("id", "unknown")
            text = result.get("text", "")
            metadata = result.get("metadata", {})
            title = metadata.get("title", metadata.get("file_name", f"Document {doc_id}"))
            
            # Format section with citation
            section_header = f"\n--- From: {title} ---\n"
            
            # Calculate how much space we have left
            remaining_space = adjusted_max_length - current_length - len(section_header)
            
            # If we don't have enough space for even a minimal excerpt
            if remaining_space < 100:
                skipped_docs += 1
                continue
                
            # If text is too long, truncate it
            if len(text) > remaining_space:
                truncated_text = text[:remaining_space - 20] + "... [truncated]"
                section = section_header + truncated_text + "\n"
                truncated_docs += 1
            else:
                section = section_header + text + "\n"
            
            section_length = len(section)
            
            # Add section
            context_parts.append(section)
            current_length += section_length
            
            # Break if we've reached the maximum length
            if current_length >= adjusted_max_length:
                break
        
        # Join parts
        context = "\n".join(context_parts)
        
        # If no content was added
        if not context:
            return "The retrieved information is too large to include in full. Please ask a more specific question."
        
        # Add metadata about what was included/excluded
        if skipped_docs > 0 or truncated_docs > 0:
            included_docs = total_docs - skipped_docs
            status = f"\n[Context constraints: {included_docs}/{total_docs} documents included"
            if truncated_docs > 0:
                status += f", {truncated_docs} documents truncated"
            status += "]"
            context += status
        
        return context
    
    def _get_template_for_query(self, query: str) -> str:
        """
        Get the appropriate template for a query.
        
        Args:
            query: User query
            
        Returns:
            Template string
        """
        query_type = self._detect_query_type(query)
        
        if query_type in self.specialized_templates:
            return self.specialized_templates[query_type]
        
        return self.default_prompt_template
    
    def generate_prompt(self, 
                      query: str, 
                      strategy: Union[str, QueryStrategy] = QueryStrategy.HYBRID,
                      limit: int = 3) -> str:
        """
        Generate a prompt for the LLM with relevant context.
        
        Args:
            query: User query
            strategy: Query strategy
            limit: Maximum number of results to include
            
        Returns:
            Formatted prompt string
        """
        try:
            # Start with a higher limit if adaptive context sizing is enabled
            effective_limit = limit
            if self.adaptive_context_sizing:
                effective_limit = min(limit * 2, 10)  # Get more results, but no more than 10
            
            # Query the document library
            query_result = self.query_engine.query(
                query_text=query,
                strategy=strategy,
                limit=effective_limit
            )
            
            # Format context
            context = self._format_context(
                query_result["results"],
                self.max_context_length
            )
            
            # Get template
            template = self._get_template_for_query(query)
            
            # Format prompt
            prompt = template.format(
                query=query,
                context=context
            )
            
            # Check prompt length and apply additional trimming if needed
            prompt_length = len(prompt)
            target_length = self.max_context_length + 500  # Add padding for template
            
            if prompt_length > target_length:
                logger.warning(f"Prompt exceeds target length ({prompt_length} > {target_length}), applying additional trimming")
                
                # Recalculate with a smaller context allowance
                reduced_context_length = max(500, int(self.max_context_length * 0.7))
                context = self._format_context(
                    query_result["results"][:min(limit, 3)],  # Use fewer results
                    reduced_context_length
                )
                
                # Regenerate prompt
                prompt = template.format(
                    query=query,
                    context=context
                )
                
                logger.info(f"Reduced prompt length from {prompt_length} to {len(prompt)} characters")
            
            return prompt
            
        except Exception as e:
            logger.error(f"Failed to generate prompt: {str(e)}")
            # Return a fallback prompt
            return f"""
You are TCCC.ai, an expert in Tactical Combat Casualty Care (TCCC) and military medicine.
Your task is to provide accurate information based on official military medical protocols.

USER QUERY:
{query}

Please answer the query based on your knowledge of TCCC protocols. 
Note: Due to a technical error, I couldn't retrieve specific context from the document library.
If you're uncertain, please acknowledge this and suggest the user consult official TCCC guidelines.
"""
    
    def extract_citations(self, response: str) -> List[str]:
        """
        Extract citations from a response.
        
        Args:
            response: LLM response text
            
        Returns:
            List of cited sources
        """
        # Extract citations in [Source: document title] format
        citations = re.findall(r'\[Source: (.*?)\]', response)
        
        # Deduplicate
        return list(set(citations))
    
    def analyze_response_quality(self, 
                                query: str, 
                                response: str, 
                                context: str) -> Dict[str, Any]:
        """
        Analyze the quality of an LLM response.
        
        Args:
            query: Original user query
            response: LLM response
            context: Context provided to the LLM
            
        Returns:
            Quality analysis dictionary
        """
        citations = self.extract_citations(response)
        
        analysis = {
            "has_citations": len(citations) > 0,
            "citations": citations,
            "response_length": len(response),
            "context_used": bool(context and context in response)
        }
        
        # Simple length-based relevance heuristic
        if len(response) < 50:
            analysis["quality"] = "low"
        elif len(response) > 500 and analysis["has_citations"]:
            analysis["quality"] = "high"
        else:
            analysis["quality"] = "medium"
        
        return analysis