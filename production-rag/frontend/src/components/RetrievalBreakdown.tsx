import { useEffect, useRef, useState } from 'react';
import type { Citation } from '../types';

interface Props {
  citations: Citation[];
  activeCitation: number | null;
}

function RankPill({ label, rank, tone }: { label: string; rank: number | null; tone: string }) {
  const present = rank !== null;
  return (
    <div className="flex items-center gap-1.5">
      <span className="font-mono text-[10px] uppercase tracking-wider text-ink-faint">{label}</span>
      <span
        className={`inline-flex h-5 min-w-[1.25rem] items-center justify-center rounded px-1 font-mono text-[11px] ${
          present ? tone : 'border border-line-soft text-ink-faint'
        }`}
      >
        {present ? `#${rank + 1}` : '—'}
      </span>
    </div>
  );
}

function Row({ c, maxRrf, active }: { c: Citation; maxRrf: number; active: boolean }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const pct = maxRrf > 0 ? Math.max(3, Math.round((c.rrf_score / maxRrf) * 100)) : 0;
  const both = c.dense_rank !== null && c.sparse_rank !== null;

  useEffect(() => {
    if (active) {
      setOpen(true);
      ref.current?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [active]);

  return (
    <div
      ref={ref}
      className={`rounded-md border bg-panel-raised transition-colors ${active ? 'border-signal' : 'border-line-soft'}`}
    >
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-3 rounded-md px-3 py-2.5 text-left focus:outline-none focus-visible:ring-1 focus-visible:ring-signal"
      >
        <span className="inline-flex h-5 min-w-[1.25rem] items-center justify-center rounded border border-signal/40 bg-signal/10 px-1 font-mono text-[11px] text-signal">
          {c.index}
        </span>
        <span className="flex-1 truncate font-mono text-xs text-ink-muted">{c.source}</span>

        <div className="hidden items-center gap-3 sm:flex">
          <RankPill label="dense" rank={c.dense_rank} tone="bg-dense/15 text-dense" />
          <RankPill label="sparse" rank={c.sparse_rank} tone="bg-sparse/15 text-sparse" />
          {both && (
            <span className="rounded-full border border-signal/30 bg-signal/10 px-2 py-0.5 font-mono text-[10px] text-signal">
              hybrid
            </span>
          )}
        </div>

        <div className="flex w-28 items-center gap-2">
          <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-panel-sunken">
            <div className="h-full rounded-full bg-signal" style={{ width: `${pct}%` }} />
          </div>
          <span className="w-12 text-right font-mono text-[10px] text-ink-muted">
            {c.rrf_score.toFixed(3)}
          </span>
        </div>
      </button>

      {open && (
        <div className="border-t border-line-soft px-3 py-3">
          <div className="mb-2 flex flex-wrap gap-3 sm:hidden">
            <RankPill label="dense" rank={c.dense_rank} tone="bg-dense/15 text-dense" />
            <RankPill label="sparse" rank={c.sparse_rank} tone="bg-sparse/15 text-sparse" />
          </div>
          <div className="grid grid-cols-2 gap-2 pb-2 font-mono text-[11px] text-ink-faint sm:grid-cols-4">
            <span>
              dense sim: <span className="text-dense">{c.dense_score?.toFixed(3) ?? '—'}</span>
            </span>
            <span>
              bm25: <span className="text-sparse">{c.sparse_score?.toFixed(2) ?? '—'}</span>
            </span>
            <span>
              rrf: <span className="text-signal">{c.rrf_score.toFixed(4)}</span>
            </span>
            <span className="truncate">id: {c.id.slice(0, 8)}</span>
          </div>
          <p className="whitespace-pre-wrap border-t border-line-soft pt-2 font-sans text-[13px] leading-relaxed text-ink-muted">
            {c.text}
          </p>
        </div>
      )}
    </div>
  );
}

export default function RetrievalBreakdown({ citations, activeCitation }: Props) {
  if (citations.length === 0) return null;
  const maxRrf = Math.max(...citations.map((c) => c.rrf_score));

  return (
    <section className="rounded-lg border border-line bg-panel shadow-panel">
      <div className="flex items-center justify-between border-b border-line-soft px-4 py-2.5">
        <span className="font-mono text-xs uppercase tracking-[0.18em] text-ink-muted">
          Hybrid retrieval · {citations.length} chunks
        </span>
        <div className="hidden items-center gap-3 font-mono text-[10px] sm:flex">
          <span className="flex items-center gap-1">
            <span className="h-2 w-2 rounded-full bg-dense" /> dense
          </span>
          <span className="flex items-center gap-1">
            <span className="h-2 w-2 rounded-full bg-sparse" /> sparse
          </span>
          <span className="flex items-center gap-1">
            <span className="h-2 w-2 rounded-full bg-signal" /> fused
          </span>
        </div>
      </div>

      <div className="space-y-2 p-4">
        {citations.map((c) => (
          <Row key={c.id} c={c} maxRrf={maxRrf} active={activeCitation === c.index} />
        ))}
        <p className="pt-1 font-mono text-[11px] leading-relaxed text-ink-faint">
          Rank shows each chunk's position in the dense and sparse lists; the bar is its fused RRF
          score. Chunks surfaced by both retrievers are tagged hybrid.
        </p>
      </div>
    </section>
  );
}
