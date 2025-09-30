import logging
import os
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from amplify_client import AmplifyClient

logger = logging.getLogger(__name__)

class SimpleRAG:
    """Simplified RAG system using Amplify and sentence-transformers"""
    
    def __init__(
        self,
        corpus_path: str,
        retrive_model_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2",
        generate_model_name: str = "gpt-4o-mini",
        context_columns: List[str] = ["title", "text"],
        template: str = "Please answer the question based on the context: {context} Question: {question}",
        top_k: int = 3,
        use_amplify: bool = True,
        **kwargs
    ):
        self.corpus_path = corpus_path
        self.generate_model_name = generate_model_name
        self.top_k = top_k
        self.template = template
        self.context_columns = context_columns
        self.use_amplify = use_amplify
        
        # Initialize embedding model
        self.encoder = SentenceTransformer(retrive_model_name_or_path)
        
        # Initialize Amplify client
        if self.use_amplify:
            self.amplify_client = AmplifyClient(model=generate_model_name)
        
        # Load and process corpus
        self.documents = []
        self.embeddings = None
        self._load_corpus()
    
    def _load_corpus(self):
        """Load corpus from CSV or JSON file"""
        try:
            if self.corpus_path.endswith('.csv'):
                df = pd.read_csv(self.corpus_path)
                self.documents = df.to_dict('records')
            elif self.corpus_path.endswith('.json'):
                with open(self.corpus_path, 'r') as f:
                    self.documents = json.load(f)
            else:
                # Try to load sample data from trial dataset
                fallback_paths = [
                    Path(__file__).resolve().parents[1] / 'trial_data_by_ass3_agent' / 'step3_extracted_larger.csv',
                    Path(__file__).resolve().parents[1] / 'trial_data_by_ass3_agent' / 'generated_qa_984rows.csv'
                ]
                for sample_path in fallback_paths:
                    if sample_path.exists():
                        df = pd.read_csv(sample_path).head(50)
                        self.documents = df.to_dict('records')
                        break
                if not self.documents:
                    # Create dummy data if no corpus found
                    self.documents = [
                        {"title": "Sample Medical Article", "text": "This is a sample medical article about cardiovascular health."},
                        {"title": "Treatment Guidelines", "text": "Guidelines for treating common medical conditions."},
                        {"title": "Research Study", "text": "A research study on the effectiveness of new treatments."}
                    ]
            
            # Create embeddings for documents
            texts = []
            for doc in self.documents:
                content = []
                for col in self.context_columns:
                    if col in doc and doc[col]:
                        content.append(str(doc[col]))
                texts.append(" ".join(content))
            
            self.embeddings = self.encoder.encode(texts)
            logger.info(f"Loaded {len(self.documents)} documents")
            
        except Exception as e:
            logger.error(f"Error loading corpus: {e}")
            # Fallback to dummy data
            self.documents = [
                {"title": "Sample Medical Article", "text": "This is a sample medical article about cardiovascular health."},
                {"title": "Treatment Guidelines", "text": "Guidelines for treating common medical conditions."},
                {"title": "Research Study", "text": "A research study on the effectiveness of new treatments."}
            ]
            texts = [f"{doc['title']} {doc['text']}" for doc in self.documents]
            self.embeddings = self.encoder.encode(texts)
    
    def retrieve(self, question: str, n_docs: int = None) -> List[Dict]:
        """Retrieve relevant documents"""
        if n_docs is None:
            n_docs = self.top_k
        
        # Encode query
        query_embedding = self.encoder.encode([question])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top k documents
        top_indices = np.argsort(similarities)[::-1][:n_docs]
        
        results = []
        for idx in top_indices:
            results.append({
                "doc_id": idx,
                "score": float(similarities[idx]),
                "series": self.documents[idx]
            })
        
        return results
    
    def generate(self, question: str, n_docs: int = None) -> Dict[str, Any]:
        """Generate response using RAG"""
        if n_docs is None:
            n_docs = self.top_k
        
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(question, n_docs)
        
        # Build context
        context_parts = []
        for doc in retrieved_docs:
            series = doc['series']
            content = []
            for col in self.context_columns:
                if col in series and series[col]:
                    content.append(str(series[col]))
            context_parts.append(" ".join(content))
        
        context = "\n\n".join(context_parts)
        
        # Generate response using Amplify
        if self.use_amplify:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful medical assistant. Use the provided context to answer questions accurately."
                },
                {
                    "role": "user", 
                    "content": self.template.format(context=context, question=question)
                }
            ]
            
            try:
                response = self.amplify_client.chat(messages, temperature=0.2)
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                response = f"Sorry, I encountered an error generating a response: {str(e)}"
        else:
            response = "Amplify not configured - using fallback response based on context."
        
        return {
            "response": response,
            "documents": retrieved_docs
        }
    
    def generate_raw(self, question: str) -> Dict[str, Any]:
        """Generate response without RAG (direct LLM)"""
        if self.use_amplify:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful medical assistant."
                },
                {
                    "role": "user",
                    "content": question
                }
            ]
            
            try:
                response = self.amplify_client.chat(messages, temperature=0.2)
            except Exception as e:
                logger.error(f"Error generating raw response: {e}")
                response = f"Sorry, I encountered an error: {str(e)}"
        else:
            response = "Amplify not configured - this is a fallback response."
        
        return {
            "response": response
        }


# Legacy compatibility
RAG = SimpleRAG
