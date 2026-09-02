"use client";

import { motion } from "framer-motion";

export type Message = { id: string; role: "user" | "assistant"; content: string; error?: boolean };

export function ChatMessage({ message, onRetry }: { message: Message; onRetry?: () => void }) {
  const user = message.role === "user";
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.2 }}
      className={`flex items-start ${user ? "justify-end" : "justify-start"}`}
    >
      <div className={`max-w-[88%] rounded-xl p-2.5 text-xs leading-5 ${user ? "rounded-tr-sm bg-violet-600 text-white" : message.error ? "rounded-tl-sm border border-amber-400/20 bg-amber-400/[0.07] text-amber-100" : "rounded-tl-sm border border-zinc-800 bg-zinc-900 text-zinc-200"}`}>
        <p className="whitespace-pre-wrap">{message.content}</p>
        {message.error && onRetry && <button onClick={onRetry} className="mt-2 rounded-full border border-amber-300/20 px-2.5 py-1 text-[11px] text-amber-200 transition hover:bg-amber-300/10">Retry</button>}
      </div>
    </motion.div>
  );
}
