#!/usr/bin/env python3
"""
Quick test script to verify the 984-row dataset loads correctly
"""
import pandas as pd
import os

# Test loading the dataset
dataset_path = "/Users/jacinda/amplify-api-course/assignment5/trial_data_by_ass3_agent/generated_qa_984rows.csv"

print(f"Testing dataset loading from: {dataset_path}")
print(f"File exists: {os.path.exists(dataset_path)}")

try:
    df = pd.read_csv(dataset_path)
    print(f"Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Sample first few rows
    sample = df.head(3)
    print(f"\nSample data:")
    for idx, row in sample.iterrows():
        print(f"\nRow {idx}:")
        print(f"  PMID: {row['pmid']}")
        print(f"  Title: {row['title'][:60]}...")
        print(f"  Journal: {row['journal']}")
        print(f"  Year: {row['year']}")
        print(f"  Abstract: {row['abstract'][:100]}...")
        
except Exception as e:
    print(f"Error loading dataset: {e}")

# Test the enhanced RAG loading
print(f"\nTesting RAG system loading...")
try:
    from enhanced_rag import EnhancedRAG
    
    rag = EnhancedRAG(
        corpus_path="datasets/sample_corpus.json",  # This should fallback to 984 rows
        context_columns=["title", "abstract"],
        use_amplify=False  # Skip Amplify for testing
    )
    
    print(f"RAG system initialized!")
    print(f"Documents loaded: {len(rag.documents)}")
    
    if rag.documents:
        first_doc = rag.documents[0]
        print(f"First document sample:")
        print(f"  PMID: {first_doc.get('pmid', 'N/A')}")
        print(f"  Title: {first_doc.get('title', 'N/A')[:60]}...")
        print(f"  Journal: {first_doc.get('journal', 'N/A')}")
        
        # Test retrieval
        print(f"\nTesting retrieval...")
        results = rag.retrieve("depression treatment", 3)
        print(f"Retrieved {len(results)} documents")
        
        for i, result in enumerate(results):
            score = result['score']
            doc = result.get('series') or result
            print(f"  {i+1}. Score: {score:.3f} | PMID: {doc.get('pmid', 'N/A')} | Title: {doc.get('title', 'N/A')[:50]}...")
            
except Exception as e:
    print(f"Error with RAG system: {e}")
    import traceback
    traceback.print_exc()
