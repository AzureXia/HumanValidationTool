#!/usr/bin/env python3
"""
Simple startup script for the medical chatbot.
This fixes the port connection issues by ensuring proper startup sequence.
"""
import uvicorn
import os
import sys
from pathlib import Path

def main():
    # Add current directory to Python path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    # Set environment variables
    os.environ.setdefault("HOST", "localhost")
    os.environ.setdefault("PORT", "8000")
    
    print("Starting Medical Chatbot...")
    print("Backend: http://localhost:8000")
    print("Frontend: http://localhost:8000/static (after starting)")
    print("API Docs: http://localhost:8000/docs")
    print("Use Ctrl+C to stop")
    
    # Start FastAPI server
    uvicorn.run(
        "main:app",
        host="localhost",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()