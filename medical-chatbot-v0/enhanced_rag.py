import logging
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from amplify_client import AmplifyClient

logger = logging.getLogger(__name__)

class EnhancedRAG:
    """Enhanced RAG system with proper cosine similarity and real medical data"""
    
    def __init__(
        self,
        corpus_path: str,
        retrive_model_name_or_path: str = None,
        generate_model_name: str = "gpt-4o-mini", 
        context_columns: List[str] = ["title", "abstract"],
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
        
        # Initialize TF-IDF vectorizer optimized for medical text
        self.vectorizer = TfidfVectorizer(
            max_features=12000,
            stop_words='english',
            ngram_range=(1, 2),  # Focus on unigrams and bigrams for better matches
            lowercase=True,
            min_df=1,  # Allow rare medical terms
            max_df=0.85,  # Exclude very common terms
            sublinear_tf=True,  # Use sublinear TF scaling
            norm='l2',  # L2 normalization
            token_pattern=r'(?u)\b\w\w+\b',  # Include medical abbreviations
            analyzer='word'
        )
        
        # Initialize Amplify client
        if self.use_amplify:
            try:
                from dotenv import load_dotenv
                load_dotenv()  # Make sure environment variables are loaded
                self.amplify_client = AmplifyClient(model=generate_model_name)
                logger.info("✓ Amplify client initialized successfully")
            except Exception as e:
                logger.error(f"✗ Failed to initialize Amplify client: {e}")
                logger.error("Amplify will not be available for response generation")
                self.use_amplify = False
        
        # Load corpus and create embeddings
        self.documents = []
        self.document_vectors = None
        self._load_and_process_corpus()
    
    def _load_and_process_corpus(self):
        """Load corpus (validated sets preferred) and create TF-IDF vectors."""
        validated_docs = self._load_validated_sets()
        if validated_docs:
            self.documents = validated_docs
        else:
            try:
                if os.path.exists(self.corpus_path):
                    logger.info(f"Loading fallback corpus from: {self.corpus_path}")
                    if self.corpus_path.endswith('.csv'):
                        df = pd.read_csv(self.corpus_path)
                        self.documents = df.to_dict('records')
                    elif self.corpus_path.endswith('.json'):
                        with open(self.corpus_path, 'r') as f:
                            self.documents = json.load(f)
                if not self.documents:
                    raise FileNotFoundError('No corpus loaded, using sample data.')
            except Exception as e:
                logger.error(f"Error loading corpus: {e}")
                self.documents = self._create_sample_medical_data()

        # Create optimized document texts for better TF-IDF performance
        document_texts = []
        for doc in self.documents:
            text_parts = []
            
            # For enhanced documents, extract meaningful text
            if doc.get('type') == 'qa':
                if doc.get('question'):
                    text_parts.append(doc['question'])
                if doc.get('answer'):
                    text_parts.append(doc['answer'])
                if doc.get('explanation'):
                    text_parts.append(doc['explanation'])
            elif doc.get('type') == 'extraction':
                if doc.get('summary'):
                    text_parts.append(doc['summary'])
                # Add structured metadata
                for field in ['population', 'symptoms', 'riskFactors', 'interventions', 'outcomes']:
                    values = doc.get(field)
                    if values:
                        if isinstance(values, list):
                            text_parts.extend(values)
                        else:
                            text_parts.append(str(values))
            
            # Always include title and abstract if available
            if doc.get('title'):
                text_parts.append(doc['title'])
            if doc.get('abstract'):
                text_parts.append(doc['abstract'])
            
            # Fallback to combined_text or other fields
            if not text_parts:
                text_parts.append(doc.get('combined_text') or doc.get('text') or 'No content')
            
            document_texts.append(' '.join(text_parts))
        self.document_vectors = self.vectorizer.fit_transform(document_texts)
        logger.info(f"Indexed {len(self.documents)} documents with {self.document_vectors.shape[1]} features")

    def _load_validated_sets(self):
        validated_dir = Path(__file__).resolve().parent / 'datasets' / 'validated'
        qa_path = validated_dir / 'qa.jsonl'
        extraction_path = validated_dir / 'extractions.jsonl'
        documents = []
        
        # Load original abstracts to enhance validated documents
        abstracts_map = {}
        original_paths = [
            "/Users/jacinda/Desktop/tmp storage for vcareslocal rag/pubmed_2500_filtered_extraction.csv",
            Path(__file__).parent / "pubmed_2500_filtered_extraction.csv"
        ]
        
        for original_path in original_paths:
            try:
                if os.path.exists(original_path):
                    import pandas as pd
                    df = pd.read_csv(original_path)
                    for _, row in df.iterrows():
                        pmid = str(row.get('pmid', ''))
                        if pmid and pd.notna(row.get('abstract')):
                            abstracts_map[pmid] = row['abstract']
                    logger.info(f"Loaded {len(abstracts_map)} abstracts from original dataset")
                    break
            except Exception as e:
                logger.warning(f"Could not load abstracts from {original_path}: {e}")
                continue

        if qa_path.exists():
            with qa_path.open() as f:
                for line in f:
                    data = json.loads(line)
                    meta_parts = []
                    for key in ['population', 'symptoms', 'riskFactors', 'interventions', 'outcomes']:
                        vals = data.get(key)
                        if vals:
                            meta_parts.append(f"{key}: {', '.join(vals if isinstance(vals, list) else [vals])}")
                    combined = "\n".join(filter(None, [
                        f"Question: {data.get('question')}",
                        f"Answer: {data.get('answer')}",
                        data.get('explanation'),
                        data.get('abstract'),
                        " | ".join(meta_parts)
                    ]))
                    # Add abstract from original dataset if available
                    pmid = str(data.get('pmid', ''))
                    abstract = abstracts_map.get(pmid, '')
                    
                    documents.append({
                        'doc_id': data.get('doc_id'),
                        'pmid': data.get('pmid'),
                        'title': data.get('title') or data.get('question'),
                        'journal': data.get('journal'),
                        'year': data.get('year'),
                        'type': 'qa',
                        'question': data.get('question'),
                        'answer': data.get('answer'),
                        'explanation': data.get('explanation'),
                        'abstract': abstract,  # Add abstract from original data
                        'combined_text': combined,
                        'summary': data.get('answer')
                    })

        if extraction_path.exists():
            with extraction_path.open() as f:
                for line in f:
                    data = json.loads(line)
                    meta_parts = []
                    for key in ['population', 'symptoms', 'riskFactors', 'interventions', 'outcomes']:
                        vals = data.get(key)
                        if vals:
                            meta_parts.append(f"{key}: {', '.join(vals)}")
                    combined = "\n".join(filter(None, [
                        f"Summary: {data.get('summary')}",
                        data.get('abstract'),
                        " | ".join(meta_parts)
                    ]))
                    # Add abstract from original dataset if available
                    pmid = str(data.get('pmid', ''))
                    abstract = abstracts_map.get(pmid, '')
                    
                    documents.append({
                        'doc_id': data.get('doc_id'),
                        'pmid': data.get('pmid'),
                        'title': data.get('title'),
                        'journal': data.get('journal'),
                        'year': data.get('year'),
                        'type': 'extraction',
                        'summary': data.get('summary'),
                        'population': data.get('population'),
                        'symptoms': data.get('symptoms'),
                        'riskFactors': data.get('riskFactors'),
                        'interventions': data.get('interventions'),
                        'outcomes': data.get('outcomes'),
                        'abstract': abstract,  # Add abstract from original data
                        'combined_text': combined
                    })

        return documents
    
    def _create_sample_medical_data(self):
        """Create sample medical data as fallback"""
        return [
            {
                "pmid": "SAMPLE_001",
                "title": "Cardiovascular Disease Prevention Guidelines",
                "abstract": "Regular exercise, maintaining healthy diet, and monitoring blood pressure are key factors in cardiovascular disease prevention. Studies show that moderate exercise for 30 minutes daily can reduce heart disease risk by up to 50%.",
                "journal": "American Journal of Cardiology",
                "year": "2023",
                "classification": "Cardiology",
                "type": "extraction",
                "combined_text": "Summary: Regular exercise and blood pressure monitoring reduce heart disease risk."
            },
            {
                "pmid": "SAMPLE_002", 
                "title": "Diabetes Management Best Practices",
                "abstract": "Type 2 diabetes management requires a comprehensive approach including blood glucose monitoring, medication adherence, dietary control, and regular physical activity. HbA1c levels should be maintained below 7% for most adults.",
                "journal": "Diabetes Care",
                "year": "2023",
                "classification": "Endocrinology",
                "type": "extraction",
                "combined_text": "Summary: Maintain HbA1c <7%, monitor glucose, use medication, diet, and exercise."
            },
            {
                "pmid": "SAMPLE_003",
                "title": "Depression Screening and Treatment",
                "abstract": "The PHQ-9 questionnaire is a validated tool for depression screening in primary care settings. Major depressive disorder treatment options include psychotherapy, antidepressant medications, or combination therapy.",
                "journal": "Journal of Clinical Psychiatry", 
                "year": "2023",
                "classification": "Mental Health",
                "type": "qa",
                "question": "What tool screens for depression in primary care?",
                "answer": "Use the PHQ-9 questionnaire; treat with psychotherapy, antidepressants, or both.",
                "combined_text": "Question: What tool screens for depression in primary care?\nAnswer: Use the PHQ-9 questionnaire; treat with psychotherapy, antidepressants, or both."
            }
        ]
    
    def retrieve(self, question: str, n_docs: int = None) -> List[Dict]:
        """Retrieve relevant documents using proper cosine similarity"""
        if n_docs is None:
            n_docs = self.top_k
        
        try:
            # Vectorize the query
            query_vector = self.vectorizer.transform([question])
            
            # Calculate cosine similarities (this gives values between 0 and 1)
            similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
            
            # Get top k document indices
            top_indices = np.argsort(similarities)[::-1][:n_docs]
            
            # Build results with full document information and deduplication
            results = []
            seen_pmids = set()
            
            for idx in top_indices:
                doc = self.documents[idx].copy()
                pmid = doc.get('pmid', f'doc_{idx}')
                
                # Skip duplicates based on PMID
                if pmid in seen_pmids:
                    continue
                seen_pmids.add(pmid)
                
                result = doc.copy()
                result.setdefault("doc_id", str(idx))
                result["score"] = float(similarities[idx])
                result["rank"] = int(idx)
                results.append(result)
                
                # Stop when we have enough unique documents
                if len(results) >= n_docs:
                    break
            
            return results
            
        except Exception as e:
            logger.error(f"Error in retrieval: {e}")
            # Return first few documents as fallback
            results = []
            for i, doc in enumerate(self.documents[:n_docs]):
                result = doc.copy()
                result["doc_id"] = str(i)
                result["score"] = 0.5  # Default score
                results.append(result)
            return results
    
    def generate(self, question: str, n_docs: int = None) -> Dict[str, Any]:
        """Generate response using RAG with proper document retrieval"""
        if n_docs is None:
            n_docs = self.top_k
        
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(question, n_docs)
        
        # Build context from retrieved documents
        context_parts = []
        for idx, doc in enumerate(retrieved_docs, start=1):
            if doc.get('combined_text'):
                bullet = [f"[{idx}] PMID {doc.get('pmid') or doc.get('doc_id')} – {doc.get('title') or 'Untitled'}"]
                if doc['type'] == 'qa':
                    bullet.append(f"Question: {doc.get('question')}")
                    bullet.append(f"Answer: {doc.get('answer')}")
                elif doc['type'] == 'extraction':
                    bullet.append(f"Summary: {doc.get('summary')}")
                bullet.append(doc['combined_text'])
                context_parts.append("\n".join(filter(None, bullet)))
            else:
                content_parts = []
                for col in self.context_columns:
                    if col in doc and doc[col] and pd.notna(doc[col]):
                        content_parts.append(f"{col.title()}: {str(doc[col])}")
                if not content_parts:
                    for alt_col in ['title', 'abstract', 'text']:
                        if alt_col in doc and doc[alt_col] and pd.notna(doc[alt_col]):
                            content_parts.append(f"{alt_col.title()}: {str(doc[alt_col])}")
                if content_parts:
                    context_parts.append(" | ".join(content_parts))

        context = "\n\n".join(context_parts)

        # Generate response using Amplify
        if self.use_amplify and hasattr(self, 'amplify_client'):
            messages = [
                {
                    "role": "system",
                    "content": "You are a medical assistant providing concise, evidence-based answers. Use bullet points and tables when appropriate. Keep responses under 150 words. Always cite sources as [PMID 123456]. Format your response in markdown."
                },
                {
                    "role": "user", 
                    "content": f"Based on this medical literature, provide a concise answer using markdown formatting:\n\nContext: {context}\n\nQuestion: {question}"
                }
            ]
            
            try:
                response = self.amplify_client.chat(messages, temperature=0.2, max_tokens=300)
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                response = f"I apologize, but I encountered an error generating a response. Based on the retrieved medical literature: {context[:300]}..."
        else:
            response = f"Based on the retrieved medical literature:\n\n{context[:500]}...\n\n(Note: Amplify connection not available for enhanced response generation)"
        
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
                    "content": "You are a helpful medical assistant. Provide accurate medical information while emphasizing the importance of consulting healthcare professionals."
                },
                {
                    "role": "user",
                    "content": question
                }
            ]
            
            try:
                response = self.amplify_client.chat(messages, temperature=0.2, max_tokens=300)
            except Exception as e:
                logger.error(f"Error generating raw response: {e}")
                response = f"I apologize, but I encountered an error: {str(e)}"
        else:
            response = "I can provide general medical information, but Amplify connection is not available for detailed responses."
        
        return {
            "response": response
        }


# Legacy compatibility
RAG = EnhancedRAG
