import { useState, type KeyboardEvent } from 'react';

interface Props {
  onAsk: (query: string, topK: number) => void;
  loading: boolean;
  disabled: boolean;
}

const SAMPLES = [
  'How many consecutive successful cleaning runs are required?',
  'What is the 4:1 test accuracy ratio in calibration?',
  'What do the letters in ALCOA stand for?',
];

export default function QueryConsole({ onAsk, loading, disabled }: Props) {
  const [query, setQuery] = useState('');
  const [topK, setTopK] = useState(5);

  const submit = () => {
    const trimmed = query.trim();
    if (!trimmed || loading) return;
    onAsk(trimmed, topK);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') submit();
  };

  return (
    <section className="rounded-lg border border-line bg-panel shadow-panel">
      <div className="flex items-center justify-between border-b border-line-soft px-4 py-2.5">
        <span className="font-mono text-xs uppercase tracking-[0.18em] text-ink-muted">Query</span>
        <span className="font-mono text-[11px] text-ink-faint">⌘ / Ctrl + ↵ to run</span>
      </div>

      <div className="p-4">
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={onKeyDown}
          rows={3}
          placeholder="Ask a question grounded in the indexed documents…"
          className="w-full resize-y rounded-md border border-line bg-panel-sunken px-3 py-2.5 font-sans text-sm text-ink placeholder:text-ink-faint focus:border-signal focus:outline-none focus:ring-1 focus:ring-signal/50"
        />

        <div className="mt-3 flex flex-wrap items-center gap-2">
          {SAMPLES.map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => setQuery(s)}
              className="rounded-full border border-line-soft bg-panel-raised px-3 py-1 text-xs text-ink-muted transition-colors hover:border-signal/50 hover:text-ink focus:outline-none focus-visible:ring-1 focus-visible:ring-signal"
            >
              {s}
            </button>
          ))}
        </div>

        <div className="mt-4 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <span className="font-mono text-[11px] uppercase tracking-[0.18em] text-ink-faint">
              top_k
            </span>
            <div className="flex items-center rounded-md border border-line bg-panel-sunken">
              <button
                type="button"
                onClick={() => setTopK((k) => Math.max(1, k - 1))}
                className="px-2.5 py-1 font-mono text-ink-muted hover:text-ink focus:outline-none"
                aria-label="Decrease top_k"
              >
                −
              </button>
              <span className="w-7 text-center font-mono text-sm text-ink">{topK}</span>
              <button
                type="button"
                onClick={() => setTopK((k) => Math.min(15, k + 1))}
                className="px-2.5 py-1 font-mono text-ink-muted hover:text-ink focus:outline-none"
                aria-label="Increase top_k"
              >
                +
              </button>
            </div>
          </div>

          <button
            type="button"
            onClick={submit}
            disabled={loading || disabled || !query.trim()}
            className="inline-flex items-center gap-2 rounded-md bg-signal px-4 py-2 font-display text-sm font-semibold text-ground transition-opacity hover:opacity-90 focus:outline-none focus-visible:ring-2 focus-visible:ring-signal focus-visible:ring-offset-2 focus-visible:ring-offset-ground disabled:cursor-not-allowed disabled:opacity-40"
          >
            {loading ? 'Running…' : 'Ask'}
          </button>
        </div>

        {disabled && (
          <p className="mt-3 text-xs text-warn">
            No documents indexed yet. Seed the sample SOPs below to start asking.
          </p>
        )}
      </div>
    </section>
  );
}
