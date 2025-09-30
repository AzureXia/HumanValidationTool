#!/usr/bin/env python3
"""
Test RAG with the new PubMed 2500 dataset
"""
from enhanced_rag import EnhancedRAG
import logging

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO)

print("Testing RAG with new PubMed 2500 dataset...")

try:
    # Initialize RAG system
    rag = EnhancedRAG(
        corpus_path="datasets/sample_corpus.json",  # This will fallback to pubmed dataset
        context_columns=["title", "abstract"],
        use_amplify=True
    )
    
    print(f"✓ RAG system initialized with {len(rag.documents)} documents")
    
    # Check if we loaded the right dataset
    first_doc = rag.documents[0]
    print(f"First document PMID: {first_doc.get('pmid', 'N/A')}")
    print(f"First document has gpt_output: {'gpt_output' in first_doc}")
    
    # Test mental health question
    question = "What's the relationship between depression and chronic medical conditions?"
    
    print(f"\nTesting question: {question}")
    
    # Test complete generation
    result = rag.generate(question, 3)
    
    print(f"\n📝 Generated Response:")
    print(result['response'])
    
    print(f"\n📚 Retrieved Documents:")
    for i, doc in enumerate(result['documents']):
        print(f"  {i+1}. PMID: {doc.get('pmid', 'N/A')} | Score: {doc.get('score', 0):.3f}")
        print(f"     Title: {doc.get('title', 'N/A')[:80]}...")
    
    # Check if Amplify is actually working
    if "Note: Amplify connection not available" in result['response']:
        print("\n✗ Still showing Amplify unavailable - need to debug further")
    else:
        print("\n✓ Amplify generation working!")
        
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()