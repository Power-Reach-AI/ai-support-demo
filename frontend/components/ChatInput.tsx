"use client";

import { ArrowUp } from "lucide-react";
import { FormEvent, useState } from "react";

export function ChatInput({ onSubmit, disabled, loading }: { onSubmit: (value: string) => void; disabled?: boolean; loading?: boolean }) {
  const [value, setValue] = useState("");
  const submit = (event: FormEvent) => { event.preventDefault(); sendValue(); };
  const sendValue = () => { if (value.trim() && !disabled) { onSubmit(value.trim()); setValue(""); } };
  return (
    <div>
      <form onSubmit={submit} className="flex items-end gap-2 rounded-xl border border-zinc-800 bg-zinc-950 p-1.5 focus-within:border-violet-400/50">
        <textarea value={value} onChange={(e) => setValue(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendValue(); } }} rows={1} placeholder="Message Dot..." disabled={disabled} className="max-h-24 min-h-9 flex-1 resize-none bg-transparent px-2 py-2 text-xs text-white outline-none placeholder:text-zinc-600" />
        <button type="submit" disabled={disabled || !value.trim()} aria-label="Send message" className="flex h-9 w-9 items-center justify-center rounded-lg bg-violet-600 text-white transition hover:scale-105 hover:bg-violet-500 active:scale-95 disabled:cursor-not-allowed disabled:bg-zinc-800 disabled:text-zinc-600">
          {loading ? <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-violet-200" /> : <ArrowUp size={16} />}
        </button>
      </form>
    </div>
  );
}
