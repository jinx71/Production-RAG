import { useRef, useState } from 'react';

interface Props {
  onSeed: () => void;
  onUpload: (files: File[]) => void;
  busy: boolean;
}

export default function IngestPanel({ onSeed, onUpload, busy }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [picked, setPicked] = useState<string[]>([]);

  const handleFiles = (list: FileList | null) => {
    if (!list || list.length === 0) return;
    const files = Array.from(list);
    setPicked(files.map((f) => f.name));
    onUpload(files);
  };

  return (
    <section className="rounded-lg border border-line bg-panel shadow-panel">
      <div className="border-b border-line-soft px-4 py-2.5">
        <span className="font-mono text-xs uppercase tracking-[0.18em] text-ink-muted">Corpus</span>
      </div>

      <div className="flex flex-col gap-3 p-4 sm:flex-row sm:items-center sm:justify-between">
        <p className="text-xs text-ink-muted">
          Load the bundled GMP SOPs, or add your own{' '}
          <span className="font-mono text-ink-faint">.pdf / .md / .txt</span>.
        </p>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => inputRef.current?.click()}
            disabled={busy}
            className="rounded-md border border-line bg-panel-raised px-3 py-1.5 text-xs text-ink-muted transition-colors hover:border-signal/50 hover:text-ink focus:outline-none focus-visible:ring-1 focus-visible:ring-signal disabled:opacity-40"
          >
            Upload files
          </button>
          <button
            type="button"
            onClick={onSeed}
            disabled={busy}
            className="rounded-md border border-signal/40 bg-signal/10 px-3 py-1.5 text-xs font-medium text-signal transition-colors hover:bg-signal/20 focus:outline-none focus-visible:ring-1 focus-visible:ring-signal disabled:opacity-40"
          >
            {busy ? 'Working…' : 'Seed sample SOPs'}
          </button>
          <input
            ref={inputRef}
            type="file"
            multiple
            accept=".pdf,.md,.txt"
            className="hidden"
            onChange={(e) => handleFiles(e.target.files)}
          />
        </div>
      </div>

      {picked.length > 0 && (
        <div className="border-t border-line-soft px-4 py-2 font-mono text-[11px] text-ink-faint">
          last upload: {picked.join(', ')}
        </div>
      )}
    </section>
  );
}
