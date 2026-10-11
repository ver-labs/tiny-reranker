"""Shared request validation for the E18 Python coordinator and worker."""
from pydantic import BaseModel, Field, model_validator


class Document(BaseModel):
    document_id: str = Field(min_length=1, max_length=256)
    text: str = Field(min_length=1, max_length=20000)


class Request(BaseModel):
    request_id: str = Field(min_length=1, max_length=256)
    query: str = Field(min_length=1, max_length=2000)
    documents: list[Document] = Field(min_length=1, max_length=256)
    top_k: int = Field(default=10, ge=1, le=256)

    @model_validator(mode='after')
    def unique_ids(self):
        if len({d.document_id for d in self.documents}) != len(self.documents):
            raise ValueError('Document IDs must be unique')
        return self
