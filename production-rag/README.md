# RegRAG — Production-Grade RAG with Eval & Guardrails

A retrieval-augmented question-answering service for regulatory documents, built
like a **product** rather than a demo. It pairs **hybrid retrieval** (dense + sparse,
fused with Reciprocal Rank Fusion) with **two layers of hallucination guardrails**,
a **Redis answer cache**, and an **offline evaluation harness** that scores answer
quality with RAGAS-style metrics.

The sample corpus is a set of pharmaceutical **GMP Standard Operating Procedures**
(cleaning validation, equipment calibration, data integrity / ALCOA+), so every
answer is grounded, cited, and checked for faithfulness — the qualities a regulated
environment actually requires.

> This is the productionised evolution of a basic "chat with your SOPs" RAG: the
> difference between a notebook that retrieves-then-prompts and a service that
> retrieves well, refuses when it should, verifies what it says, caches, and can be
> measured.

- 🔎 **Live demo:** _add deployed URL here_
- 🖼️ **Screenshot:** _add screenshot of the retrieval console here_

---

## Why this is not a toy RAG

| Concern | Toy RAG | RegRAG |
|---|---|---|
| Retrieval | dense-only top-k | dense + BM25, fused with RRF |
| Out-of-domain question | answers anyway (hallucinates) | relevance gate refuses pre-generation |
| Grounding | trusts the model | post-generation faithfulness verifier |
| Repeat questions | re-runs everything | Redis cache (skips retrieve + generate) |
| "Is it any good?" | vibes | eval harness with per-question metrics |
| Transparency | a black-box answer | every rank, score, and latency is surfaced |

---

## Architecture

```
                       ┌──────────────────────────────────────────────┐
  React console  ────► │  FastAPI                                      │
  (Vite + TS)          │                                              │
                       │   POST /query                                │
                       │     │                                        │
                       │     ▼                                        │
                       │   RagPipeline.answer()                       │
                       │     │                                        │
                       │     ├─ 1. Redis cache lookup ────────────────┼─► Redis
                       │     │      (hit → return immediately)         │
                       │     │                                        │
                       │     ├─ 2. Hybrid retrieve                     │
                       │     │      ├─ dense  → Chroma (cosine) ───────┼─► fastembed (ONNX)
                       │     │      └─ sparse → BM25 (rank_bm25)        │
                       │     │      └─ fuse   → Reciprocal Rank Fusion  │
                       │     │                                        │
                       │     ├─ 3. Guardrail: relevance gate           │
                       │     │      (weak retrieval → refuse, no LLM)   │
                       │     │                                        │
                       │     ├─ 4. Generate (context-only + citations)─┼─► Claude (Sonnet)
                       │     │                                        │
                       │     └─ 5. Guardrail: faithfulness check ──────┼─► Claude (Haiku)
                       │            (claims unsupported → flag)         │
                       └──────────────────────────────────────────────┘
```

A request returns the answer **plus** the evidence: which chunks were retrieved,
their dense and sparse ranks, the fused RRF score, the guardrail verdicts, and a
per-stage latency breakdown.

---

## Key design decisions

**Hybrid retrieval + RRF.** Dense vectors capture meaning; BM25 captures exact terms
(part numbers, "ALCOA", "4:1 ratio") that embeddings blur. Their scores live on
different scales, so they're combined by **rank**, not score: `score = Σ 1/(k + rank)`.
This is robust and tuning-free.
> Why: a regulatory query like "the 4:1 test accuracy ratio" needs exact-term recall
> that pure semantic search misses, while paraphrased questions need semantic recall
> that keyword search misses. Fusing both beats either alone.

**Two-layer guardrails.** A cheap, deterministic **relevance gate** runs *before*
generation — if the top dense similarity is below threshold, the service refuses
without paying for an LLM call. A **faithfulness check** runs *after* generation — a
cheap model judges whether each claim is supported by the retrieved context.
> Why: the gate stops out-of-domain hallucination at the door; the verifier catches
> the subtler case where retrieval succeeded but the model drifted. Both fail *open*
> on verifier error so a flaky judge never blocks a good answer.

**Embeddings: fastembed with a zero-dependency fallback.** Production uses `fastembed`
(ONNX, no PyTorch → lean image). A deterministic `HashingEmbedder` is built in for
CI/offline, selectable via `EMBEDDING_BACKEND`.
> Why: keeps the container small and the test suite fast and network-free, without
> changing a line of pipeline code.

**Caching with graceful degradation.** Answers are cached in Redis keyed by
`(normalized query, top_k, model)`. If Redis is down, the cache disables itself and
requests still succeed.
> Why: a dependency being unavailable should degrade performance, not availability.

**Evaluation, not vibes.** `eval/` runs the pipeline over a golden dataset and reports
faithfulness, answer relevancy, context precision, and answer similarity. Metrics are
implemented from first principles (no fragile heavy deps); an optional `run_ragas.py`
cross-checks against the real RAGAS library.

---

## Tech stack

**Backend:** Python · FastAPI · Chroma (vector store) · rank_bm25 (sparse) · fastembed ·
Redis · Anthropic Claude API · PyMuPDF
**Frontend:** React 18 · TypeScript · Vite · Tailwind CSS · Axios
**Eval:** RAGAS-style metrics (built-in) + optional RAGAS library
**Tooling:** pytest (30 tests) · Docker · Docker Compose

---

## API reference

All responses use the envelope `{ "success": bool, "data": <T>, "message": str }`.

| Method | Path | Body | Description |
|---|---|---|---|
| `GET` | `/health` | — | Doc count, embedding backend, cache stats |
| `POST` | `/query` | `{ query, top_k? }` | Answer + citations + guardrails + timings |
| `POST` | `/ingest` | multipart `files[]` | Ingest uploaded `.pdf/.md/.txt` |
| `POST` | `/ingest/seed` | — | Ingest the bundled sample SOPs |
| `POST` | `/reset` | — | Clear the dense + sparse indexes |

<details>
<summary><code>POST /query</code> response shape</summary>

```jsonc
{
  "success": true,
  "data": {
    "answer": "Three consecutive successful cleaning runs are required [1].",
    "cited_indices": [1],
    "citations": [
      {
        "index": 1, "id": "…", "source": "sop-cleaning-validation.md",
        "text": "…", "dense_rank": 0, "sparse_rank": 1,
        "dense_score": 0.71, "sparse_score": 4.2, "rrf_score": 0.0325
      }
    ],
    "guardrails": {
      "relevance_gate_passed": true, "top_dense_score": 0.71,
      "relevance_threshold": 0.30,
      "faithfulness": { "checked": true, "grounded": true,
        "faithfulness_score": 1.0, "unsupported_claims": [], "threshold": 0.6 },
      "refused": false
    },
    "cache_hit": false,
    "timings_ms": { "retrieval": 12.4, "generation": 840.1, "faithfulness": 410.7, "total": 1263.2 }
  },
  "message": "ok"
}
```
</details>

---

## Getting started

### 1. Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then add your ANTHROPIC_API_KEY
uvicorn app.main:app --reload
```

API on `http://localhost:8000`. With `AUTO_SEED=true` the sample SOPs load on startup;
otherwise `POST /ingest/seed` (or use the **Seed sample SOPs** button in the UI).

> Redis is optional locally. Leave `CACHE_ENABLED=true` and the app simply runs
> without caching if no Redis is reachable. To enable it: `docker run -p 6379:6379 redis:7-alpine`.

### 2. Frontend

```bash
cd frontend
npm install
cp .env.example .env        # VITE_API_URL defaults to http://localhost:8000
npm run dev
```

Console on `http://localhost:5173`.

### 3. Everything via Docker Compose

```bash
export ANTHROPIC_API_KEY=sk-ant-...
docker compose up --build
```

Frontend on `http://localhost:8080`, backend on `http://localhost:8000`, Redis wired
up and the corpus auto-seeded.

---

## Evaluation

```bash
cd backend
python -m eval.evaluate            # full golden dataset
python -m eval.evaluate --limit 3  # quick run
```

Writes `eval/eval_report.json` and `eval/eval_report.md`:

| Metric | Meaning |
|---|---|
| `faithfulness` | fraction of answer claims supported by retrieved context |
| `answer_relevancy` | how well the answer addresses the question |
| `context_precision` | fraction of retrieved chunks relevant to the question |
| `answer_similarity` | embedding similarity between answer and reference |

Optional cross-check against the real library (in a **clean** virtualenv):

```bash
pip install -r eval/requirements-eval.txt
python -m eval.run_ragas --limit 3
```

See `eval/README.md` for details.

---

## Testing

```bash
cd backend
pip install -r requirements.txt
pytest                # 30 tests, fully offline (hashing embedder + fakeredis + fake LLM)
```

Coverage spans chunking, embeddings, BM25, RRF fusion, the hybrid retriever, the Redis
cache, both guardrails, and the API routes (including the deterministic refusal path).

---

## Project structure

```
production-rag/
├── backend/
│   ├── app/            # FastAPI app + RAG pipeline (retrieval, generation, guardrails, cache)
│   ├── eval/           # RAGAS-style harness + golden dataset
│   ├── scripts/        # sample GMP SOPs
│   ├── tests/          # 30 pytest tests
│   └── Dockerfile
├── frontend/           # React + TS retrieval console
│   └── Dockerfile
└── docker-compose.yml
```

---

## Interview talking points

- **Hybrid retrieval & RRF.** Why rank-based fusion sidesteps the cosine-vs-BM25 scale
  mismatch, and when sparse retrieval rescues queries that embeddings miss.
- **Guardrails as defence in depth.** A pre-generation gate (cheap, deterministic) and a
  post-generation verifier (LLM-judged), and the deliberate choice to fail *open*.
- **Cost-aware model routing.** A strong model for generation, a cheap one for the
  faithfulness judge; caching to avoid paying twice for the same question.
- **Evaluation.** Implementing the RAGAS metrics from first principles vs. depending on
  the library — and the real version-pinning pain that motivated keeping a no-deps path.
- **Production posture.** Graceful degradation when Redis is down, a lean ONNX image with
  an offline fallback embedder, an `app`/factory split that makes the API testable.
- **Domain fit.** Why grounding, citations, and refusal aren't features but requirements
  in a GMP / regulated context.

---

## Roadmap

Part of an AI/ML engineering portfolio. Natural next steps: persist evaluation runs and
chart them over time, add hybrid-weight tuning, and deploy to a cloud free tier with a
CI/CD pipeline (covered by a separate "cloud-deployed AI microservice" project).
