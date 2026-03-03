"""Pydantic schemas for requests, responses, and the {success, data, message} envelope."""

from typing import Generic, Optional, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)


class Citation(BaseModel):
    index: int
    id: str
    source: str
    text: str
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    rrf_score: float


class FaithfulnessInfo(BaseModel):
    checked: bool
    grounded: bool
    faithfulness_score: float
    unsupported_claims: list[str] = []
    threshold: Optional[float] = None


class Guardrails(BaseModel):
    relevance_gate_passed: bool
    top_dense_score: float
    relevance_threshold: float
    faithfulness: FaithfulnessInfo
    refused: bool


class Timings(BaseModel):
    retrieval: float
    generation: float
    faithfulness: float
    total: float


class QueryResult(BaseModel):
    answer: str
    cited_indices: list[int]
    citations: list[Citation]
    guardrails: Guardrails
    cache_hit: bool
    timings_ms: Timings


class IngestResult(BaseModel):
    documents_ingested: int
    chunks_added: int
    total_chunks: int


class HealthData(BaseModel):
    status: str
    document_chunks: int
    embedding_backend: str
    cache: dict


class ApiResponse(BaseModel, Generic[T]):
    success: bool = True
    data: Optional[T] = None
    message: str = "OK"
