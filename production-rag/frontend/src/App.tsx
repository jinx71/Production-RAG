import { useCallback, useEffect, useState } from 'react';
import StatusBar from './components/StatusBar';
import QueryConsole from './components/QueryConsole';
import AnswerPanel from './components/AnswerPanel';
import RetrievalBreakdown from './components/RetrievalBreakdown';
import IngestPanel from './components/IngestPanel';
import { askQuestion, fetchHealth, ingestFiles, seedSamples } from './api/client';
import type { HealthData, QueryResult } from './types';

export default function App() {
  const [health, setHealth] = useState<HealthData | null>(null);
  const [result, setResult] = useState<QueryResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeCitation, setActiveCitation] = useState<number | null>(null);

  const refreshHealth = useCallback(async () => {
    try {
      setHealth(await fetchHealth());
    } catch {
      setHealth(null);
    }
  }, []);

  useEffect(() => {
    refreshHealth();
  }, [refreshHealth]);

  const onAsk = async (query: string, topK: number) => {
    setLoading(true);
    setError(null);
    setActiveCitation(null);
    try {
      const res = await askQuestion(query, topK);
      setResult(res);
      refreshHealth();
    } catch {
      setError('Request failed. Is the backend running on the configured API URL?');
    } finally {
      setLoading(false);
    }
  };

  const onSeed = async () => {
    setBusy(true);
    setError(null);
    try {
      await seedSamples();
      await refreshHealth();
    } catch {
      setError('Could not seed documents. Check the backend connection.');
    } finally {
      setBusy(false);
    }
  };

  const onUpload = async (files: File[]) => {
    setBusy(true);
    setError(null);
    try {
      await ingestFiles(files);
      await refreshHealth();
    } catch {
      setError('Upload failed. Supported types: .pdf, .md, .txt.');
    } finally {
      setBusy(false);
    }
  };

  const noDocs = (health?.document_chunks ?? 0) === 0;

  return (
    <div className="min-h-screen">
      <StatusBar health={health} />

      <main className="mx-auto max-w-6xl space-y-4 px-5 py-6">
        <div className="space-y-1">
          <h2 className="font-display text-2xl font-semibold text-ink">Ask the regulatory corpus</h2>
          <p className="max-w-2xl text-sm text-ink-muted">
            Hybrid retrieval over GMP SOPs with a relevance gate before generation and a
            faithfulness check after it. Every answer shows its sources, retrieval ranks, and
            latency — so you can see why it said what it said.
          </p>
        </div>

        <QueryConsole onAsk={onAsk} loading={loading} disabled={noDocs} />

        {error && (
          <div className="rounded-md border border-danger/40 bg-danger/10 px-4 py-3 text-sm text-danger">
            {error}
          </div>
        )}

        {result && (
          <>
            <AnswerPanel result={result} onCiteClick={setActiveCitation} />
            <RetrievalBreakdown citations={result.citations} activeCitation={activeCitation} />
          </>
        )}

        <IngestPanel onSeed={onSeed} onUpload={onUpload} busy={busy} />
      </main>

      <footer className="mx-auto max-w-6xl px-5 py-8">
        <p className="font-mono text-[11px] text-ink-faint">
          RegRAG · hybrid search · RRF fusion · relevance gate · faithfulness verification · Redis
          cache
        </p>
      </footer>
    </div>
  );
}
