from typing import Union
import schema as sc
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from enhanced_rag import EnhancedRAG as RAG
from settings import (
    corpus_path,
    retrive_model_name_or_path,
    generate_model_name,
    context_columns,
    template,
    top_k,
    use_amplify,
    HOST,
    PORT
)
from fastapi.middleware.cors import CORSMiddleware
import os

rgg = RAG(
    corpus_path=corpus_path,
    retrive_model_name_or_path=retrive_model_name_or_path,
    generate_model_name=generate_model_name,
    context_columns=context_columns,
    template=template,
    top_k=top_k,
    use_amplify=use_amplify
)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return FileResponse("enhanced_ui.html")

@app.get("/chatbot")
def get_chatbot():
    return FileResponse("enhanced_ui.html")

@app.get("/config")
def get_config():
    return {
        "api_base_url": f"http://{HOST}:{PORT}",
        "host": HOST,
        "port": PORT
    }


@app.post("/chat/", response_model=sc.ChatResponse)
def read_item(req: sc.ChatRequest):
    res = rgg.generate(req.text)
    ret = res["documents"]

    docs = []

    for x in range(len(ret)):
        doc = ret[x]
        # Document data is now directly in doc, not under 'series'

        def safe_str(value, default):
            if value is None or (isinstance(value, float) and str(value).lower() == 'nan'):
                return default
            return str(value)
        
        docs.append(
            sc.Document(
                pmid=safe_str(doc.get('pmid', doc.get('doc_id')), 'N/A'),
                title=safe_str(doc.get('title'), 'Untitled'),
                abstract=safe_str(doc.get('abstract'), 'No abstract available'),
                date=safe_str(doc.get('date'), '2023-01-01'),
                journal=safe_str(doc.get('journal'), 'Unknown Journal'),
                publication_type=safe_str(doc.get('publication_type'), 'Research Article'),
                year=safe_str(doc.get('year'), '2023'),
                classification=safe_str(doc.get('classification'), 'Unknown'),
                gpt_output=safe_str(doc.get('gpt_output', doc.get('abstract')), 'No content')
            )
        )

    return sc.ChatResponse(
        message="Request successful",
        data=sc.ChatMessage(
            text = res['response'],
            document = docs
        ),
        code=1
    )

@app.post("/chat2/", response_model=sc.ChatNewResponse)
def read_item(req: sc.ChatRequest):
    res = rgg.generate(req.text)
    ret = res["documents"]

    docs = []

    for x in range(len(ret)):
        doc = ret[x]
        # Document data is now directly in doc
        ks = ["doc_id", "score" ]
        vs = [ str(doc["doc_id"]), str(doc["score"]) ]

        for k, v in doc.items():
            if k in ["embeddings", "doc_id", "score"]: continue
            ks.append(str(k))
            vs.append(str(v))

        docs.append(
            sc.DocumentNew(
                keys=ks,
                values=vs
            )
        )

    return sc.ChatNewResponse(
        message="Request successful",
        data=sc.ChatMessageNew(
            text = res['response'],
            document = docs
        ),
        code=1
    )


@app.post("/chat3/", response_model=sc.FinalChatNewResponse)
def read_item(req: sc.ChatRequest):
    res = rgg.generate(req.text)
    ret = res["documents"]

    docs = []

    for x in range(len(ret)):
        doc = ret[x]
        # Document data is now directly in doc
        item = {
            "doc_id": str(doc["doc_id"]),
            "score": str(doc["score"])
        }
        for k, v in doc.items():
            if k in ["embeddings", "doc_id", "score"]: continue
            item[k] = str(v)

        docs.append(
            item
        )

    return sc.FinalChatNewResponse(
        message="Request successful",
        data=sc.FinalChatMessageNew(
            text = res['response'],
            document = docs
        ),
        code=1
    )


@app.post("/chat_raw/", response_model=sc.ChatNewResponse)
def chat_raw(req: sc.ChatRequest):
    res = rgg.generate_raw(req.text)
    return sc.ChatNewResponse(
        message="Request successful",
        data=sc.ChatMessageNew(
            text = res['response'],
            document = None
        ),
        code=1
    )
