import type { QueryResult } from '../types';

interface Props {
  result: QueryResult;
  onCiteClick: (index: number) => void;
}

type Tone = 'pass' | 'warn' | 'danger' | 'muted';

const TONE_CLASS: Record<Tone, string> = {
  pass: 'border-pass/40 bg-pass/10 text-pass',
  warn: 'border-warn/40 bg-warn/10 text-warn',
  danger: 'border-danger/40 bg-danger/10 text-danger',
  muted: 'border-line bg-panel-raised text-ink-muted',
};

function GuardrailBadges({ result }: { result: QueryResult }) {
  const g = result.guardrails;
  const f = g.faithfulness;
  const badges: { label: string; tone: Tone }[] = [];

  if (g.refused) {
    badges.push({ label: 'Refused · low retrieval confidence', tone: 'danger' });
  } else if (f.checked && !f.grounded) {
    badges.push({ label: `Unverified · faithfulness ${f.faithfulness_score.toFixed(2)}`, tone: 'warn' });
  } else if (f.checked && f.grounded) {
    badges.push({ label: `Grounded · faithfulness ${f.faithfulness_score.toFixed(2)}`, tone: 'pass' });
  } else {
    badges.push({ label: 'Faithfulness not checked', tone: 'muted' });
  }

  badges.push({
    label: `relevance ${g.top_dense_score.toFixed(2)} / ${g.relevance_threshold.toFixed(2)}`,
    tone: g.relevance_gate_passed ? 'pass' : 'danger',
  });

  if (result.cache_hit) badges.push({ label: 'Cached', tone: 'muted' });

  return (
    <div className="flex flex-wrap gap-2">
      {badges.map((b) => (
        <span key={b.label} className={`rounded-full border px-2.5 py-1 font-mono text-[11px] ${TONE_CLASS[b.tone]}`}>
          {b.label}
        </span>
      ))}
    </div>
  );
}

function AnswerText({ answer, onCiteClick }: { answer: string; onCiteClick: (i: number) => void }) {
  const parts = answer.split(/(\[\d+\])/g);
  return (
    <p className="whitespace-pre-wrap font-sans text-[15px] leading-relaxed text-ink">
      {parts.map((part, i) => {
        const m = part.match(/^\[(\d+)\]$/);
        if (m) {
          const idx = Number(m[1]);
          return (
            <button
              key={i}
              type="button"
              onClick={() => onCiteClick(idx)}
              className="mx-0.5 inline-flex h-5 min-w-[1.25rem] items-center justify-center rounded border border-signal/40 bg-signal/10 px-1 align-baseline font-mono text-[11px] text-signal transition-colors hover:bg-signal/20 focus:outline-none focus-visible:ring-1 focus-visible:ring-signal"
            >
              {idx}
            </button>
          );
        }
        return <span key={i}>{part}</span>;
      })}
    </p>
  );
}

function Timing({ label, ms, max }: { label: string; ms: number; max: number }) {
  const pct = max > 0 ? Math.max(2, Math.round((ms / max) * 100)) : 0;
  return (
    <div className="flex items-center gap-3">
      <span className="w-20 font-mono text-[11px] text-ink-faint">{label}</span>
      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-panel-sunken">
        <div className="h-full rounded-full bg-signal/60" style={{ width: `${pct}%` }} />
      </div>
      <span className="w-16 text-right font-mono text-[11px] text-ink-muted">{ms.toFixed(0)} ms</span>
    </div>
  );
}

export default function AnswerPanel({ result, onCiteClick }: Props) {
  const t = result.timings_ms;
  const maxStage = Math.max(t.retrieval, t.generation, t.faithfulness);
  const unsupported = result.guardrails.faithfulness.unsupported_claims;

  return (
    <section className="rounded-lg border border-line bg-panel shadow-panel">
      <div className="flex items-center justify-between border-b border-line-soft px-4 py-2.5">
        <span className="font-mono text-xs uppercase tracking-[0.18em] text-ink-muted">Answer</span>
        <span className="font-mono text-[11px] text-ink-faint">{t.total.toFixed(0)} ms total</span>
      </div>

      <div className="space-y-4 p-4">
        <GuardrailBadges result={result} />
        <AnswerText answer={result.answer} onCiteClick={onCiteClick} />

        {unsupported.length > 0 && (
          <div className="rounded-md border border-warn/30 bg-warn/5 p-3">
            <p className="font-mono text-[11px] uppercase tracking-wider text-warn">Unsupported claims</p>
            <ul className="mt-1.5 list-disc space-y-1 pl-4 text-xs text-ink-muted">
              {unsupported.map((c, i) => (
                <li key={i}>{c}</li>
              ))}
            </ul>
          </div>
        )}

        <div className="space-y-1.5 border-t border-line-soft pt-3">
          <Timing label="retrieval" ms={t.retrieval} max={maxStage} />
          <Timing label="generation" ms={t.generation} max={maxStage} />
          <Timing label="faithfulness" ms={t.faithfulness} max={maxStage} />
        </div>
      </div>
    </section>
  );
}
