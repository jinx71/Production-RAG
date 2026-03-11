import type { HealthData } from '../types';

interface Props {
  health: HealthData | null;
}

function Readout({ label, value, accent }: { label: string; value: string; accent?: string }) {
  return (
    <div className="flex flex-col">
      <span className="text-[10px] uppercase tracking-[0.18em] text-ink-faint">{label}</span>
      <span className={`font-mono text-sm ${accent ?? 'text-ink'}`}>{value}</span>
    </div>
  );
}

export default function StatusBar({ health }: Props) {
  const online = health?.status === 'ok';
  const cache = health?.cache;
  const hitRate =
    cache?.available && cache.hit_rate !== undefined
      ? `${Math.round(cache.hit_rate * 100)}%`
      : '—';

  return (
    <header className="sticky top-0 z-10 border-b border-line bg-panel/70 backdrop-blur">
      <div className="mx-auto flex max-w-6xl flex-col gap-4 px-5 py-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-3">
          <div className="grid h-9 w-9 place-items-center rounded-md border border-line bg-panel-raised">
            <span className="font-display text-base font-bold text-signal">R</span>
          </div>
          <div>
            <h1 className="font-display text-lg font-semibold leading-none text-ink">RegRAG</h1>
            <p className="mt-1 text-xs text-ink-muted">Production retrieval console</p>
          </div>
        </div>

        <div className="flex items-center gap-6">
          <Readout label="Chunks" value={health ? String(health.document_chunks) : '—'} />
          <Readout label="Embeddings" value={health?.embedding_backend ?? '—'} />
          <Readout label="Cache hit" value={hitRate} accent="text-signal" />
          <div className="flex items-center gap-2">
            <span
              className={`h-2 w-2 rounded-full ${online ? 'bg-pass motion-safe:animate-pulse' : 'bg-danger'}`}
            />
            <span className="font-mono text-xs text-ink-muted">{online ? 'online' : 'offline'}</span>
          </div>
        </div>
      </div>
    </header>
  );
}
