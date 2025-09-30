#!/usr/bin/env python3
"""
Test the complete RAG pipeline
"""
from enhanced_rag import EnhancedRAG
import os

print("Testing complete RAG pipeline...")

try:
    # Initialize RAG system
    rag = EnhancedRAG(
        corpus_path="datasets/sample_corpus.json",
        context_columns=["title", "abstract"],
        use_amplify=True
    )
    
    print(f"✓ RAG system initialized with {len(rag.documents)} documents")
    
    # Test question about mental health
    question = "What's the relationship between depression and chronic medical conditions?"
    
    print(f"\nTesting question: {question}")
    
    # Test retrieval
    retrieved_docs = rag.retrieve(question, 3)
    print(f"✓ Retrieved {len(retrieved_docs)} documents")
    
    for i, doc in enumerate(retrieved_docs):
        print(f"  {i+1}. Score: {doc.get('score', 0):.3f} | PMID: {doc.get('pmid', 'N/A')} | Title: {doc.get('title', 'N/A')[:60]}...")
    
    # Test complete generation
    print(f"\n🧠 Testing complete RAG generation...")
    result = rag.generate(question, 3)
    
    print(f"✓ Generated response:")
    print(f"Response: {result['response'][:200]}...")
    print(f"Documents returned: {len(result['documents'])}")
    
    # Check if it's using Amplify (not showing fallback message)
    if "Note: Amplify connection not available" in result['response']:
        print("✗ Still showing Amplify unavailable message")
    else:
        print("✓ Amplify generation working properly!")
        
except Exception as e:
    print(f"✗ Error in RAG pipeline: {e}")
    import traceback
    traceback.print_exc()