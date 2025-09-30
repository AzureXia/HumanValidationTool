#!/usr/bin/env python3
"""
Debug script to test API response structure
"""
import requests
import json

# Test the API directly
api_url = "http://localhost:8000/chat3/"

test_question = "How do I manage type 2 diabetes?"

print(f"Testing API with question: {test_question}")

try:
    response = requests.post(api_url, json={"text": test_question})
    
    print(f"Status code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Response structure:")
        print(json.dumps(data, indent=2))
        
        # Check documents structure
        if 'data' in data and 'document' in data['data']:
            docs = data['data']['document']
            print(f"\nFound {len(docs)} documents:")
            
            for i, doc in enumerate(docs):
                print(f"\nDocument {i+1}:")
                print(f"  doc_id: {doc.get('doc_id')}")
                print(f"  score: {doc.get('score')}")
                
                if 'series' in doc:
                    series = doc['series']
                    print(f"  series keys: {list(series.keys())}")
                    print(f"  title: {series.get('title', 'N/A')}")
                    print(f"  pmid: {series.get('pmid', 'N/A')}")
                    print(f"  journal: {series.get('journal', 'N/A')}")
                    print(f"  year: {series.get('year', 'N/A')}")
                else:
                    print(f"  No 'series' key found")
                    print(f"  Doc keys: {list(doc.keys())}")
    else:
        print(f"Error response: {response.text}")
        
except requests.exceptions.ConnectionError:
    print("Cannot connect to API. Make sure the server is running on port 8000")
except Exception as e:
    print(f"Error: {e}")