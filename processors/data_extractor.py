import json
import logging
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field, create_model

# Set up logger for this module
logger = logging.getLogger(__name__)

class DataExtractor:
    def __init__(self, schema_path: str = "config/extraction_schema.json", config_path: str = "config/config.json"):
        # Load extraction schema
        with open(schema_path, 'r') as f:
            self.schemas = json.load(f)
        
        # Load main configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.extraction_config = self.config["extraction"]
        self.openai_config = self.config["openai"]
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.openai_config.get("model", "gpt-3.5-turbo"), 
            temperature=self.openai_config.get("temperature", 0),
            max_tokens=self.openai_config.get("max_tokens", 1000)
        )
        
        # Set up the output parser
        self.output_parser = JsonOutputParser()
        
        # Create extraction prompt template
        self.extraction_prompt = ChatPromptTemplate.from_template("""
        Extract the following information from this {doc_type} document:
        
        {extraction_fields}
        
        Document text:
        {text}
        
        Respond with a JSON object containing only the extracted fields.
        If you cannot find a specific field, set its value to null.
        """)
        
        # Build extraction chains for each document type
        self.extraction_chains = {}
        for doc_type, schema in self.schemas.items():
            self.extraction_chains[doc_type] = self.extraction_prompt | self.llm | self.output_parser
    
    def _create_field_instructions(self, doc_type: str) -> str:
        """Create detailed field instructions for the prompt"""
        if doc_type not in self.schemas:
            logger.warning(f"Unknown document type: {doc_type}")
            return "Error: Unknown document type"
            
        schema = self.schemas[doc_type]
        field_instructions = []
        
        for field_name, field_info in schema["properties"].items():
            required = "Required" if field_name in schema.get("required", []) else "Optional"
            field_instructions.append(
                f"- {field_name}: {field_info['description']} ({required})"
            )
            
        return "\n".join(field_instructions)
    
    async def extract_data(self, text: str, doc_type: str) -> Dict[str, Any]:
        """Extract data from document using LLM"""
        if doc_type not in self.schemas:
            logger.error(f"No extraction schema for document type: {doc_type}")
            return {"error": f"No extraction schema for document type: {doc_type}"}
        
        try:
            # Get extraction fields
            extraction_fields = self._create_field_instructions(doc_type)
            
            # Sample the text based on config
            text_limit = self.extraction_config.get("text_sample_limit", 3000)
            text_sample = text[:text_limit] + ("..." if len(text) > text_limit else "")
            
            # Extract data using LLM
            result = self.extraction_chains[doc_type].invoke({
                "doc_type": doc_type,
                "extraction_fields": extraction_fields,
                "text": text_sample
            })
            
            # Validate results against schema
            schema = self.schemas[doc_type]
            for field in schema.get("required", []):
                if field not in result or result[field] is None:
                    logger.warning(f"Required field '{field}' not found in extraction result for {doc_type}")
                    result[field] = "Not found"
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting data: {str(e)}", exc_info=True)
            return {"error": f"Error extracting data: {str(e)}"}