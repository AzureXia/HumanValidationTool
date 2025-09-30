from pydantic import BaseModel
from typing import Optional, Union, Any, List, Dict


class ChatRequest(BaseModel):
    text: str = None


class BaseResponse(BaseModel):
    message: str = "Request successful"
    data: Optional[Any] = None
    code: int = 1


class Document(BaseModel):
    pmid: str = None
    title: str = None
    abstract: str = None
    date: str = None
    journal: str = None
    publication_type: str = None
    year: str = None
    classification: str = None
    gpt_output: str = None

class DocumentNew(BaseModel):
    keys: List[str] = None
    values: List[str] = None


class ChatMessage(BaseResponse):
    text: str = None
    document: List[Document] = None

class ChatMessageNew(BaseResponse):
    text: str = None
    document: List[DocumentNew] = None


class ChatResponse(BaseResponse):
    data: Optional[ChatMessage] = None


class ChatNewResponse(BaseResponse):
    data: Optional[ChatMessageNew] = None



class FinalChatMessageNew(BaseResponse):
    text: str = None
    document: List[Dict] = None

class FinalChatNewResponse(BaseResponse):
    data: Optional[FinalChatMessageNew] = None