export interface Citation {
  index: number;
  id: string;
  source: string;
  text: string;
  dense_rank: number | null;
  sparse_rank: number | null;
  dense_score: number | null;
  sparse_score: number | null;
  rrf_score: number;
}

export interface Faithfulness {
  checked: boolean;
  grounded: boolean;
  faithfulness_score: number;
  unsupported_claims: string[];
  threshold: number | null;
}

export interface Guardrails {
  relevance_gate_passed: boolean;
  top_dense_score: number;
  relevance_threshold: number;
  faithfulness: Faithfulness;
  refused: boolean;
}

export interface Timings {
  retrieval: number;
  generation: number;
  faithfulness: number;
  total: number;
}

export interface QueryResult {
  answer: string;
  cited_indices: number[];
  citations: Citation[];
  guardrails: Guardrails;
  cache_hit: boolean;
  timings_ms: Timings;
}

export interface CacheStats {
  available: boolean;
  hits: number;
  misses: number;
  hit_rate?: number;
}

export interface HealthData {
  status: string;
  document_chunks: number;
  embedding_backend: string;
  cache: CacheStats;
}

export interface IngestResult {
  documents_ingested: number;
  chunks_added: number;
  total_chunks: number;
}

export interface ApiResponse<T> {
  success: boolean;
  data: T;
  message: string;
}
