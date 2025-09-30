import logging
import os
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from amplify_client import AmplifyClient

logger = logging.getLogger(__name__)

class SimpleRAG:
    """Ultra-simple RAG system using basic text matching"""
    
    def __init__(
        self,
        corpus_path: str,
        retrive_model_name_or_path: str = None,
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
        
        # Initialize Amplify client
        if self.use_amplify:
            try:
                self.amplify_client = AmplifyClient(model=generate_model_name)
            except Exception as e:
                logger.error(f"Failed to initialize Amplify client: {e}")
                self.use_amplify = False
        
        # Load corpus
        self.documents = []
        self._load_corpus()
    
    def _load_corpus(self):
        """Load corpus from CSV or JSON file"""
        try:
            if os.path.exists(self.corpus_path):
                if self.corpus_path.endswith('.csv'):
                    df = pd.read_csv(self.corpus_path)
                    self.documents = df.to_dict('records')
                elif self.corpus_path.endswith('.json'):
                    with open(self.corpus_path, 'r') as f:
                        self.documents = json.load(f)
            else:
                fallback_paths = [
                    Path(__file__).resolve().parents[1] / 'trial_data_by_ass3_agent' / 'step3_extracted_larger.csv',
                    Path(__file__).resolve().parents[1] / 'trial_data_by_ass3_agent' / 'generated_qa_984rows.csv'
                ]
                for sample_path in fallback_paths:
                    if sample_path.exists():
                        df = pd.read_csv(sample_path).head(20)
                        self.documents = df.to_dict('records')
                        break
            
            # Fallback to dummy data if nothing worked
            if not self.documents:
                self.documents = [
                    {
                        "title": "Cardiovascular Disease Prevention",
                        "text": "Regular exercise and healthy diet are key factors in preventing cardiovascular disease. Studies show moderate exercise for 30 minutes daily reduces heart disease risk by up to 50%.",
                        "doc_id": "CVD_001"
                    },
                    {
                        "title": "Diabetes Management",
                        "text": "Type 2 diabetes management requires blood glucose monitoring, medication adherence, and dietary control. HbA1c levels should be maintained below 7% for most adults.",
                        "doc_id": "DM_002"
                    },
                    {
                        "title": "Hypertension Treatment",
                        "text": "First-line treatments for hypertension include ACE inhibitors, ARBs, calcium channel blockers, and thiazide diuretics. Target blood pressure is less than 130/80 mmHg.",
                        "doc_id": "HTN_003"
                    },
                    {
                        "title": "Depression Screening",
                        "text": "The PHQ-9 questionnaire is a validated tool for depression screening. SSRIs and SNRIs are typically first-line pharmacological treatments for major depressive disorder.",
                        "doc_id": "MH_004"
                    }
                ]
            
            logger.info(f"Loaded {len(self.documents)} documents")
            
        except Exception as e:
            logger.error(f"Error loading corpus: {e}")
            # Use minimal fallback
            self.documents = [
                {"title": "Medical Knowledge", "text": "Basic medical information available.", "doc_id": "DEFAULT"}
            ]
    
    def _simple_search(self, question: str, n_docs: int) -> List[Dict]:
        """Simple keyword-based search"""
        question_lower = question.lower()
        scored_docs = []
        
        for i, doc in enumerate(self.documents):
            score = 0
            
            # Check title and text for keyword matches
            for col in self.context_columns:
                if col in doc and doc[col]:
                    content = str(doc[col]).lower()
                    # Simple word matching
                    words = question_lower.split()
                    for word in words:
                        if len(word) > 2:  # Skip very short words
                            score += content.count(word) * (2 if col == 'title' else 1)
            
            scored_docs.append({
                "doc_id": i,
                "score": score / max(len(question_lower.split()), 1),  # Normalize by query length
                "series": doc
            })
        
        # Sort by score and return top k
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        return scored_docs[:n_docs]
    
    def retrieve(self, question: str, n_docs: int = None) -> List[Dict]:
        """Retrieve relevant documents"""
        if n_docs is None:
            n_docs = self.top_k
        
        return self._simple_search(question, n_docs)
    
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
        if self.use_amplify and hasattr(self, 'amplify_client'):
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful medical assistant. Use the provided context to answer questions accurately. If the context doesn't contain relevant information, say so clearly."
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
                response = f"I apologize, but I encountered an error generating a response. Based on the available information: {context[:200]}..."
        else:
            response = f"Based on the available medical literature: {context[:300]}... (Note: Amplify connection not available for full response generation)"
        
        return {
            "response": response,
            "documents": retrieved_docs
        }
    
    def generate_raw(self, question: str) -> Dict[str, Any]:
        """Generate response without RAG (direct LLM)"""
        if self.use_amplify and hasattr(self, 'amplify_client'):
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
                response = f"I apologize, but I encountered an error: {str(e)}"
        else:
            response = "I can provide general medical information, but Amplify connection is not available for detailed responses."
        
        return {
            "response": response
        }


# Legacy compatibility
RAG = SimpleRAG
