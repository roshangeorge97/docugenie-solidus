from pydantic import BaseModel
from typing import List, Optional, Tuple

# Model definitions
class Payload(BaseModel):
    query: str
    history: List[Tuple[Optional[str], Optional[str]]] = []

class Request(BaseModel):
    method: str
    payload: Payload

class LangChainRequestCall(BaseModel):
    userToken: str
    requestId: str
    request: Request

class UploadPayload(BaseModel):
    urls: List[str]

class UploadRequest(BaseModel):
    method: str
    payload: UploadPayload

class UploadRequestCall(BaseModel):
    userToken: str
    requestId: str
    request: UploadRequest

class TaskRequest(BaseModel):
    taskId: str
