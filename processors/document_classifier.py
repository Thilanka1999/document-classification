import json
import os
import logging
from typing import Dict, Tuple, Optional
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Set up logger for this module
logger = logging.getLogger(__name__)

class DocumentClassifier:
    def __init__(self, categories_path: str = "config/categories.json", config_path: str = "config/config.json"):
        # Load categories
        with open(categories_path, 'r') as f:
            self.categories_config = json.load(f)
        
        # Load main configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.categories = self.categories_config["categories"]
        self.classification_config = self.config["classification"]
        self.openai_config = self.config["openai"]
        
        # Initialize embedding model for rule-based classification
        embedding_model_name = self.classification_config.get("embedding_model", "paraphrase-MiniLM-L6-v2")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Pre-compute category embeddings
        self.category_embeddings = {}
        for category in self.categories:
            description = category["description"]
            self.category_embeddings[category["name"]] = self.embedding_model.encode(description)
        
        # Initialize LLM classifier
        self.llm = ChatOpenAI(
            model=self.openai_config.get("model", "gpt-3.5-turbo"), 
            temperature=self.openai_config.get("temperature", 0)
        )
        
        # Prepare prompt for classification
        self.classification_prompt = ChatPromptTemplate.from_template("""
        You are a document classification expert. Analyze the following document text and 
        classify it into one of these categories:
        
        {categories_str}
        
        Document text:
        {text}
        
        Respond with a JSON object containing:
        - "category": The name of the best matching category
        - "confidence": A value between 0 and 1 indicating your confidence
        - "reasoning": Brief explanation of your classification
        """)
        
        # Set up the output parser
        self.output_parser = JsonOutputParser()
        
        # Build the classification chain
        self.classification_chain = self.classification_prompt | self.llm | self.output_parser
    
    def classify_with_embeddings(self, text: str) -> Tuple[Optional[str], float]:
        """Classify document using embedding similarity"""
        # Get document embedding
        doc_embedding = self.embedding_model.encode(text[:5000])  # Limit to first 5000 chars
        
        # Calculate similarity with each category
        similarities = {}
        for category, category_embedding in self.category_embeddings.items():
            similarity = cosine_similarity(
                [doc_embedding], 
                [category_embedding]
            )[0][0]
            similarities[category] = similarity
        
        # Find the best match
        if similarities:
            best_category = max(similarities, key=similarities.get)
            confidence = similarities[best_category]
            return best_category, float(confidence)
        
        return None, 0.0
    
    async def classify_with_llm(self, text: str) -> Tuple[Optional[str], float, str]:
        """Classify document using LLM"""
        try:
            # Create a string representation of categories
            categories_str = "\n".join([
                f"- {cat['name']}: {cat['description']}" 
                for cat in self.categories
            ])
            
            # Extract a sample of the text based on config
            text_limit = self.classification_config.get("text_sample_limit", 2000)
            text_sample = text[:text_limit] + ("..." if len(text) > text_limit else "")
            
            # Get classification from LLM (handle both sync & async mocks)
            maybe_coro = self.classification_chain.invoke({
                "categories_str": categories_str,
                "text": text_sample
            })
            if hasattr(maybe_coro, "__await__"):
                # Support AsyncMock or async invoke implementations
                result = await maybe_coro  # type: ignore[arg-type]
            else:
                result = maybe_coro
            
            category = result.get("category", "").lower()
            confidence = float(result.get("confidence", 0.0))
            reasoning = result.get("reasoning", "")
            
            # Validate category is in our list
            valid_categories = [cat["name"].lower() for cat in self.categories]
            if category in valid_categories:
                return category, confidence, reasoning
            else:
                return None, 0.0, "Invalid category returned"
                
        except Exception as e:
            logger.error(f"Error classifying with LLM: {str(e)}", exc_info=True)
            return None, 0.0, str(e)
    
    async def classify(self, text: str, use_llm: bool = True) -> Dict:
        """Intelligent classification that ensembles LLM and embedding predictions.

        The method always evaluates *both* the LLM-based and the embedding-based
        classifiers (unless ``use_llm`` is False).  The final decision follows
        these rules:
        1.  If both classifiers agree on the category -> choose that category
            and return a weighted average of their confidences (``ensemble_agreement``).
        2.  If they disagree:
            a.  When ``use_llm`` is True and the LLM confidence is **strictly**
                higher than the embedding confidence **and** greater than the
                configured ``confidence_threshold`` -> trust the LLM (method
                ``llm``).
            b.  Otherwise fall back to the embedding prediction (method
                ``embeddings``).

        An additional ``uncertainty`` field is returned which is simply the
        absolute difference between the two confidence scores (larger means the
        two models disagree more).
        """

        # Default response structure
        result = {
            "category": None,
            "confidence": 0.0,
            "method": None,
            "reasoning": "",
            # New keys (consumers can ignore safely)
            "uncertainty": 1.0,
            "explanation": {}
        }

        # ------------------------------------------------------------------
        # 1. Run both classifiers (unless the caller explicitly disables LLM)
        # ------------------------------------------------------------------
        # LLM branch ---------------------------------------------------------
        llm_category: Optional[str] = None
        llm_confidence: float = 0.0
        llm_reasoning: str = ""
        if use_llm:
            try:
                llm_category, llm_confidence, llm_reasoning = await self.classify_with_llm(text)
            except Exception as e:
                logger.warning(f"LLM classification failed, continuing with embeddings only: {str(e)}")

        # Embedding branch ---------------------------------------------------
        emb_category, emb_confidence = self.classify_with_embeddings(text)

        # Ensure we have numeric types
        llm_confidence = float(llm_confidence or 0.0)
        emb_confidence = float(emb_confidence or 0.0)

        # ------------------------------------------------------------------
        # 2. Decide final label via ensemble logic
        # ------------------------------------------------------------------
        llm_weight = self.classification_config.get("llm_weight", 0.75)
        emb_weight = self.classification_config.get("embedding_weight", 0.25)
        confidence_threshold = self.classification_config.get("confidence_threshold", 0.7)

        decision_path = []
        final_category: Optional[str] = None
        final_confidence: float = 0.0
        method: str = "embeddings"  # default fallback

        # Case A – agreement -------------------------------------------------
        if llm_category and llm_category == emb_category:
            final_category = llm_category
            # Weighted average of confidences
            final_confidence = (
                llm_confidence * llm_weight + emb_confidence * emb_weight
            ) / (llm_weight + emb_weight)
            method = "ensemble_agreement"
            decision_path.append(
                f"Both classifiers agree on '{final_category}'. Weighted confidence computed."
            )
        else:
            # Disagreement ---------------------------------------------------
            decision_path.append(
                f"Disagreement – LLM: '{llm_category}' ({llm_confidence:.2f}) vs. "
                f"Embeddings: '{emb_category}' ({emb_confidence:.2f})."
            )
            # Compare weighted confidences to mitigate over-reliance on raw scores
            llm_weighted_conf = llm_confidence * llm_weight
            emb_weighted_conf = emb_confidence * emb_weight

            if use_llm and llm_category and llm_weighted_conf >= emb_weighted_conf and llm_confidence >= confidence_threshold:
                # Trust the LLM prediction (weighted)
                final_category = llm_category
                final_confidence = llm_confidence
                method = "llm"
                decision_path.append("Chosen LLM prediction due to higher confidence.")
            else:
                # Fall back to embeddings
                final_category = emb_category
                final_confidence = emb_confidence
                method = "embeddings"
                decision_path.append("Chosen embedding prediction.")

        # ------------------------------------------------------------------
        # 3. Populate result dict
        # ------------------------------------------------------------------
        uncertainty = abs(llm_confidence - emb_confidence) if llm_category else 1.0

        result.update({
            "category": final_category,
            "confidence": float(min(max(final_confidence, 0.0), 1.0)),  # clamp 0-1
            "method": method,
            "reasoning": llm_reasoning if method == "llm" else f"Similarity score: {emb_confidence:.4f}",
            "uncertainty": float(uncertainty),
            "explanation": {
                "decision_path": decision_path,
                "llm_category": llm_category,
                "llm_confidence": llm_confidence,
                "embedding_category": emb_category,
                "embedding_confidence": emb_confidence,
            },
        })

        return result