from pydantic import BaseModel


class TextDocumentRequest(BaseModel):
    tenant_id: str = "default"
    title: str
    text: str


class URLDocumentRequest(BaseModel):
    tenant_id: str = "default"
    url: str
    async_mode: bool = False
