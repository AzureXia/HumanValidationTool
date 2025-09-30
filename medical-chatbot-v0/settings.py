HOST = "localhost"
PORT = 8000

# Local development paths - using sample data from assignment5
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
corpus_path = os.path.join(current_dir, "datasets", "sample_corpus.json")

# Use CPU for local development - no GPU dependencies
retrive_model_name_or_path = "sentence-transformers/all-MiniLM-L6-v2"
reranker_model_name_or_path = None  # Disable reranker for now

# Amplify integration - will use amplify_client.py
generate_model_name = "gpt-4o-mini"
use_amplify = True
use_azure = False

# Legacy settings (not used with Amplify)
base_url = ""
api_key = ""
azure_openai_args = {}

# Knowledge Base - Updated for 984-row medical dataset
context_columns = ["title", "abstract"]
column_names: str = "title,abstract"


template: str = "Please answer the question very strictly based on the your knowledge base: {context} Do not use outside literature. Question: {question}"

system_prompt_template: str = (
    "You are a retrieval-augmented assistant. Use only the passages in the knowledge base context "
    "below. Do not invent or cite external sources.\n\nContext:\n{context}"
)

user_prompt_template: str = "{question}"


top_k: int = 3
device: str = "cpu"  # Use CPU for local development
