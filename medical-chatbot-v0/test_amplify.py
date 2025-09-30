#!/usr/bin/env python3
"""
Test Amplify connection
"""
from amplify_client import AmplifyClient

print("Testing Amplify connection...")

try:
    client = AmplifyClient(model="gpt-4o-mini")
    print("✓ Amplify client initialized successfully")
    
    # Test a simple call
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello"}
    ]
    
    response = client.chat(messages, temperature=0.2)
    print(f"✓ Amplify response: {response}")
    
except Exception as e:
    print(f"✗ Amplify error: {e}")
    print(f"Error type: {type(e)}")
    
    # Check environment variables
    import os
    print(f"\nEnvironment check:")
    print(f"AMPLIFY_API_KEY exists: {'AMPLIFY_API_KEY' in os.environ}")
    print(f"AMPLIFY_API_URL exists: {'AMPLIFY_API_URL' in os.environ}")
    
    if 'AMPLIFY_API_KEY' in os.environ:
        print(f"API Key (first 10 chars): {os.environ['AMPLIFY_API_KEY'][:10]}...")
    if 'AMPLIFY_API_URL' in os.environ:
        print(f"API URL: {os.environ['AMPLIFY_API_URL']}")