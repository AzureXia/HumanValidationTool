# Medical RAG Chatbot v0

A **medical RAG (Retrieval-Augmented Generation) chatbot** that combines document retrieval with Amplify-powered language generation to answer medical questions based on a curated knowledge base.

## What's Been Fixed

This version addresses the **port connection issues** from the original RAG system by:
- **Simplified architecture** - Using FastAPI with a single unified startup script
- **Local development focus** - No GPU dependencies, works on CPU
- **Amplify integration** - Seamless connection to Vanderbilt's Amplify API
- **Enhanced frontend** - Modern chat interface with history and citations
- **Reliable startup** - Single command to run everything

## Features

### Core RAG Functionality
- **Document Retrieval** - Semantic search through medical literature
- **Context-Aware Generation** - LLM responses based on retrieved documents  
- **Citation Display** - Shows source documents with relevance scores
- **Flexible Knowledge Base** - Supports JSON/CSV medical datasets

### Enhanced UI Features
- **Real-time Chat** - Clean, responsive chat interface
- **Source Citations** - Documents used for each response with scores
- **Chat History** - Persistent conversation history in sidebar
- **Mobile Responsive** - Works on desktop and mobile devices
- **Live Status** - Connection status and real-time feedback

### Technical Improvements  
- **No Port Issues** - Single startup script manages everything
- **Fast Setup** - Minimal dependencies, works out of the box
- **Amplify Ready** - Pre-configured for Vanderbilt Amplify API
- **Local Development** - CPU-only, no GPU requirements

## Prerequisites

1. **Python 3.8+**
2. **Amplify API Access** - Ensure your `.env` file has valid credentials:
   ```
   AMPLIFY_API_KEY=your_api_key_here
   AMPLIFY_API_URL=https://your-amplify-endpoint/chat
   AMPLIFY_MODEL=gpt-4o-mini
   ```

## Quick Start

### 1. Install Dependencies
```bash
cd medical-chatbot-v0
pip install -r requirements.txt
```

### 2. Start the Chatbot
```bash
python start_chatbot.py
```

### 3. Access the Interface
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs  
- **Backend Status**: http://localhost:8000/config

## Project Structure

```
medical-chatbot-v0/
├── start_chatbot.py          # Main startup script (run this!)
├── main.py                   # FastAPI application  
├── corpus_rag_amplify.py     # Simplified RAG implementation
├── amplify_client.py         # Amplify API integration
├── enhanced_chatbot.html     # Modern chat interface
├── settings.py               # Configuration
├── datasets/
│   └── sample_corpus.json    # Medical knowledge base
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables
└── README.md                 # This file
```

## Demo Guide

### Sample Questions to Try:
1. **"What are the guidelines for hypertension treatment?"**
2. **"How do I manage type 2 diabetes?"**  
3. **"What are the preventive care screening recommendations?"**
4. **"Tell me about antibiotic resistance prevention"**
5. **"How should I treat depression in primary care?"**

### Expected Demo Flow:
1. **Start Server** - Run `python start_chatbot.py`
2. **Open Browser** - Navigate to http://localhost:8000
3. **Ask Question** - Type a medical question and press Enter
4. **View Response** - See AI-generated answer with source citations
5. **Check Sources** - Review cited documents with relevance scores
6. **Browse History** - See conversation history in left sidebar

## API Endpoints

- **GET /** - Main chat interface
- **POST /chat3/** - RAG chat with full citations  
- **POST /chat_raw/** - Direct LLM (no RAG)
- **GET /config** - Server configuration
- **GET /docs** - Interactive API documentation

## Customization

### Adding Your Own Data
Validated pipeline:

1. Review content in `human-expert-validation-v1`.
2. Run the bridge exporter (`assignment5/human-chatbot-bridge-v1/export_validated.py`) or the demo prep script.
3. The chatbot automatically looks for `datasets/validated/qa.jsonl` and `datasets/validated/extractions.jsonl` on startup.

If you prefer manual JSON, the structure is:

```json
[
  {
    "title": "Your Document Title",
    "text": "Full document text content...",
    "doc_id": "UNIQUE_ID", 
    "category": "Medical Specialty"
  }
]
```

### Changing Models
Edit `settings.py`:
```python
generate_model_name = "gpt-4o"  # or "gpt-4o-mini"
retrive_model_name_or_path = "sentence-transformers/all-MiniLM-L6-v2"
```

## Performance Notes

- **Startup Time**: ~30 seconds (downloads embedding model on first run)
- **Query Response**: 2-5 seconds per question
- **Memory Usage**: ~500MB RAM for embeddings
- **Concurrent Users**: Supports multiple users simultaneously

## Troubleshooting

### Common Issues:

**"Connection Error" in UI:**
- Check that `start_chatbot.py` is running
- Verify backend is accessible at http://localhost:8000/config

**"Missing AMPLIFY_API_KEY":**
- Ensure `.env` file exists with valid Amplify credentials
- Check that environment variables are loaded correctly

**Slow responses:**
- First query downloads embedding models (~100MB)
- Subsequent queries should be faster
- Consider using smaller models for development

**Port already in use:**
- Change PORT in `settings.py` to a different value (e.g., 8001)
- Kill existing processes: `lsof -ti:8000 | xargs kill -9`

## Production Deployment

For production use:
1. Set `HOST = "0.0.0.0"` in `settings.py`
2. Use a production WSGI server like `gunicorn`
3. Set up proper authentication and rate limiting
4. Use a dedicated vector database for large datasets

---

**Ready for Demo!** This medical chatbot demonstrates the successful integration of RAG with Amplify, providing reliable medical information with proper source attribution.
