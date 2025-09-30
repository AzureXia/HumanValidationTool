#!/usr/bin/env python3
"""Test the enhanced RAG system with validated datasets."""

from enhanced_rag import EnhancedRAG
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

print("=== Testing Enhanced RAG System ===\n")

try:
    # Initialize RAG system
    print("1. Initializing RAG system...")
    rag = EnhancedRAG(
        corpus_path="datasets/sample_corpus.json",
        context_columns=["title", "abstract"],
        use_amplify=True
    )
    
    print(f"✓ RAG system initialized with {len(rag.documents)} documents")
    
    # Check what type of documents we loaded
    if rag.documents:
        first_doc = rag.documents[0]
        print(f"✓ Document type: {first_doc.get('type', 'unknown')}")
        if 'combined_text' in first_doc:
            print(f"✓ Using enhanced validated datasets")
            print(f"  Sample combined text: {first_doc['combined_text'][:100]}...")
        else:
            print(f"✓ Using regular dataset format")
    
    # Test questions
    questions = [
        "What is depression?",
        "How effective is cognitive behavioral therapy?",
        "What are the symptoms of anxiety?"
    ]
    
    for i, question in enumerate(questions, 2):
        print(f"\n{i}. Testing question: {question}")
        
        try:
            result = rag.generate(question, 2)
            print(f"✓ Response generated ({len(result['response'])} chars)")
            print(f"✓ Retrieved {len(result['documents'])} documents")
            
            for j, doc in enumerate(result['documents'][:2]):
                pmid = doc.get('pmid', 'N/A')
                title = doc.get('title', 'N/A')
                score = doc.get('score', 0)
                doc_type = doc.get('type', 'regular')
                print(f"  {j+1}. [{doc_type}] PMID {pmid} | Score: {score:.3f} | {title[:50]}...")
                
        except Exception as e:
            print(f"✗ Error with question '{question}': {e}")
    
    print(f"\n=== Testing Complete ===")
    
except Exception as e:
    print(f"✗ System initialization failed: {e}")
    import traceback
    traceback.print_exc()