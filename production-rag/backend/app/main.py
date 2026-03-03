"""FastAPI application exposing the RAG pipeline."""

import tempfile
from pathlib import Path

from fastapi import Depends, FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from app.llm import GroqLLM 
from app.bm25_index import BM25Index
from app.cache import QueryCache
from app.config import Settings, get_settings
from app.embeddings import get_embedder
from app.ingest import ingest_paths, seed_sample_documents
from app.llm import AnthropicLLM
from app.pipeline import RagPipeline
from app.retrieval import HybridRetriever
from app.schemas import (
    ApiResponse,
    HealthData,
    IngestResult,
    QueryRequest,
    QueryResult,
)
from app.vectorstore import VectorStore


def build_state(settings: Settings) -> dict:
    embedder = get_embedder(settings)
    vectorstore = VectorStore(settings.chroma_persist_dir, settings.chroma_collection)
    bm25 = BM25Index(str(Path(settings.chroma_persist_dir) / "bm25.json"))
    retriever = HybridRetriever(
        vectorstore, bm25, embedder, candidate_k=settings.candidate_k, rrf_k=settings.rrf_k
    )
    cache = QueryCache(
        settings.redis_url,
        ttl_seconds=settings.cache_ttl_seconds,
        enabled=settings.cache_enabled,
    )
    llm = GroqLLM(settings.groq_api_key, settings.generation_model)
    pipeline = RagPipeline(retriever=retriever, llm=llm, cache=cache, settings=settings)
    return {
        "settings": settings,
        "embedder": embedder,
        "vectorstore": vectorstore,
        "bm25": bm25,
        "cache": cache,
        "pipeline": pipeline,
    }


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title="Production RAG API", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in settings.client_url.split(",")],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def _startup() -> None:
        for key, value in build_state(settings).items():
            setattr(app.state, key, value)
        if settings.auto_seed and app.state.vectorstore.count() == 0:
            seed_sample_documents(
                app.state.embedder, app.state.vectorstore, app.state.bm25, settings
            )

    def get_pipeline(request: Request) -> RagPipeline:
        return request.app.state.pipeline

    @app.get("/health", response_model=ApiResponse[HealthData])
    async def health(request: Request):
        state = request.app.state
        data = HealthData(
            status="ok",
            document_chunks=state.vectorstore.count(),
            embedding_backend=state.settings.embedding_backend,
            cache=state.cache.stats(),
        )
        return ApiResponse(data=data, message="healthy")

    @app.post("/query", response_model=ApiResponse[QueryResult])
    async def query(req: QueryRequest, pipeline: RagPipeline = Depends(get_pipeline)):
        result = await pipeline.answer(req.query, req.top_k)
        return ApiResponse(data=QueryResult(**result), message="ok")

    @app.post("/ingest", response_model=ApiResponse[IngestResult])
    async def ingest(request: Request, files: list[UploadFile] = File(...)):
        state = request.app.state
        saved: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            for upload in files:
                dest = Path(tmp) / (upload.filename or "document.txt")
                dest.write_bytes(await upload.read())
                saved.append(str(dest))
            documents, chunks = ingest_paths(
                saved, state.embedder, state.vectorstore, state.bm25, state.settings
            )
        data = IngestResult(
            documents_ingested=documents,
            chunks_added=chunks,
            total_chunks=state.vectorstore.count(),
        )
        return ApiResponse(data=data, message="ingested")

    @app.post("/ingest/seed", response_model=ApiResponse[IngestResult])
    async def ingest_seed(request: Request):
        state = request.app.state
        documents, chunks = seed_sample_documents(
            state.embedder, state.vectorstore, state.bm25, state.settings
        )
        data = IngestResult(
            documents_ingested=documents,
            chunks_added=chunks,
            total_chunks=state.vectorstore.count(),
        )
        return ApiResponse(data=data, message="seeded")

    @app.post("/reset", response_model=ApiResponse[IngestResult])
    async def reset(request: Request):
        state = request.app.state
        state.vectorstore.reset()
        state.bm25.reset()
        data = IngestResult(documents_ingested=0, chunks_added=0, total_chunks=0)
        return ApiResponse(data=data, message="reset")

    return app


app = create_app()
