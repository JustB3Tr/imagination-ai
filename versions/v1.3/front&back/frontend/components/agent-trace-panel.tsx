'use client';

import type { AgentTraceEntry } from '@/lib/types';
import { cn } from '@/lib/utils';
import { Cpu, AlertCircle, Wrench, CheckCircle2, XCircle } from 'lucide-react';

function KindIcon({ kind, ok }: { kind: AgentTraceEntry['kind']; ok?: boolean }) {
  if (kind === 'thought') return <Cpu className="h-3.5 w-3.5 shrink-0 text-accent" aria-hidden />;
  if (kind === 'error') return <AlertCircle className="h-3.5 w-3.5 shrink-0 text-destructive" aria-hidden />;
  if (kind === 'tool_call') return <Wrench className="h-3.5 w-3.5 shrink-0 text-muted-foreground" aria-hidden />;
  if (ok) return <CheckCircle2 className="h-3.5 w-3.5 shrink-0 text-emerald-500" aria-hidden />;
  return <XCircle className="h-3.5 w-3.5 shrink-0 text-amber-500" aria-hidden />;
}

export function AgentTracePanel({ entries }: { entries: AgentTraceEntry[] }) {
  if (!entries.length) return null;

  return (
    <div className="rounded-xl border border-border/80 bg-muted/15 p-3 shadow-sm">
      <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        Agent trace
      </p>
      <ul className="max-h-[min(45vh,26rem)] space-y-2 overflow-y-auto overscroll-contain pr-1 [scrollbar-gutter:stable]">
        {entries.map(e => (
          <li
            key={e.id}
            className={cn(
              'rounded-lg border border-border/50 bg-card/90 px-2.5 py-2 text-xs leading-relaxed',
              e.kind === 'error' && 'border-destructive/40 bg-destructive/5'
            )}
          >
            <div className="flex items-start gap-2">
              <KindIcon kind={e.kind} ok={e.ok} />
              <div className="min-w-0 flex-1 space-y-1">
                <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[11px] text-muted-foreground">
                  <span className="font-medium uppercase tracking-wide text-foreground/80">
                    {e.kind === 'thought'
                      ? 'Thinking'
                      : e.kind === 'error'
                        ? 'Error'
                        : e.kind === 'tool_call'
                          ? `Tool · ${e.name || '?'}`
                          : `Result · ${e.name || '?'}`}
                  </span>
                  {e.callId ? (
                    <span className="font-mono text-[10px] opacity-70" title={e.callId}>
                      {e.callId.slice(0, 8)}…
                    </span>
                  ) : null}
                </div>
                {e.text ? (
                  <p className="whitespace-pre-wrap text-foreground/90">{e.text}</p>
                ) : null}
                {e.detail ? (
                  <pre className="max-h-32 overflow-auto rounded bg-muted/50 p-2 font-mono text-[10px] text-muted-foreground">
                    {e.detail}
                  </pre>
                ) : null}
              </div>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
