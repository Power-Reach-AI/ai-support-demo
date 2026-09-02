"use client";

import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { MessageCircle, X } from "lucide-react";
import { ChatInput } from "../components/ChatInput";
import { ChatMessage, Message } from "../components/ChatMessage";

const API_ENDPOINT = "http://localhost:8000/api/chat";
const sessionId = typeof crypto !== "undefined" && crypto.randomUUID ? crypto.randomUUID() : `session-${Date.now()}`;

export default function Home() {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const lastPrompt = useRef("");
  const abortRef = useRef<AbortController | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  async function sendMessage(message: string, addUser = true) {
    if (loading) return;
    lastPrompt.current = message;
    const assistantId = `${Date.now()}-assistant`;
    if (addUser) setMessages((current) => [...current, { id: `${Date.now()}-user`, role: "user", content: message }, { id: assistantId, role: "assistant", content: "" }]);
    else setMessages((current) => [...current.filter((item) => !item.error), { id: assistantId, role: "assistant", content: "" }]);
    setLoading(true);
    const controller = new AbortController(); abortRef.current = controller;
    let assistantMessage = "";
    let animationFrame: number | null = null;
    let streamFailed = false;
    const renderStreamUpdate = () => {
      animationFrame = null;
      setMessages((current) => current.map((item) => item.id === assistantId ? { ...item, content: assistantMessage } : item));
    };
    const appendToken = (token: string) => {
      assistantMessage += token;
      if (animationFrame === null) animationFrame = requestAnimationFrame(renderStreamUpdate);
    };
    try {
      const response = await fetch(API_ENDPOINT, { method: "POST", mode: "cors", cache: "no-store", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message, session_id: sessionId }), signal: controller.signal });
      if (!response.ok) {
        const detail = await response.text();
        console.error("Dot API request failed", { status: response.status, statusText: response.statusText, detail });
        throw new Error(`Dot could not connect right now (${response.status}).`);
      }
      if (!response.body) {
        console.error("Dot API returned no streaming response body");
        throw new Error("Dot could not connect right now.");
      }
      const reader = response.body.getReader(); const decoder = new TextDecoder(); let buffer = "";
      while (true) {
        const { value, done } = await reader.read(); if (done) break; buffer += decoder.decode(value, { stream: true });
        const frames = buffer.split("\n\n"); buffer = frames.pop() ?? "";
        for (const frame of frames) {
          const line = frame.split("\n").find((entry) => entry.startsWith("data: ")); if (!line) continue;
          const data = line.slice(6); if (data === "[DONE]") continue;
          const parsed = JSON.parse(data); if (typeof parsed !== "string" && parsed.error) { console.error("Dot API stream error", parsed.error); throw new Error(parsed.error); }
          const token = typeof parsed === "string" ? parsed : parsed.token ?? "";
          appendToken(token);
        }
      }
    } catch (error) {
      if ((error as Error).name !== "AbortError") { streamFailed = true; console.error("Dot request error", error); setMessages((current) => current.map((item) => item.id === assistantId ? { ...item, error: true, content: error instanceof Error ? error.message : "Dot is temporarily unavailable." } : item)); }
    } finally {
      if (animationFrame !== null) cancelAnimationFrame(animationFrame);
      if (assistantMessage && !streamFailed) setMessages((current) => current.map((item) => item.id === assistantId ? { ...item, content: assistantMessage } : item));
      setLoading(false); abortRef.current = null;
    }
  }

  const retry = () => { if (lastPrompt.current) void sendMessage(lastPrompt.current, false); };

  return <main className="min-h-screen bg-[#0b0d12]" aria-label="Host landing page">
    <div className="flex min-h-screen items-center justify-center px-6 text-center"><div><div className="mx-auto mb-5 h-2 w-2 rounded-full bg-violet-400 shadow-[0_0_24px_8px_rgba(167,139,250,.25)]" /><p className="text-sm tracking-[0.3em] text-zinc-600 uppercase">Your workspace</p></div></div>
    <AnimatePresence>
      {open && <motion.section initial={{ opacity: 0, scale: 0.95, y: 10 }} animate={{ opacity: 1, scale: 1, y: 0 }} exit={{ opacity: 0, scale: 0.95, y: 10 }} transition={{ type: "spring", stiffness: 300, damping: 25 }} className="fixed bottom-20 right-6 z-50 flex h-[580px] max-h-[80vh] w-[calc(100vw-3rem)] max-w-[380px] flex-col overflow-hidden rounded-2xl border border-zinc-800/80 bg-zinc-900/95 shadow-2xl backdrop-blur-md" aria-label="Dot chat widget">
        <header className="flex h-12 shrink-0 items-center justify-between border-b border-zinc-800/80 p-3"><p className="text-sm font-medium text-zinc-100">Dot</p><button onClick={() => setOpen(false)} className="rounded-lg p-1.5 text-zinc-500 transition hover:bg-zinc-800 hover:text-zinc-200" aria-label="Close Dot"><X size={16} /></button></header>
        <div className="flex-1 overflow-y-auto p-3"><div className="space-y-3">{messages.length === 0 && <div className="py-12 text-center"><p className="text-xs text-zinc-300">Hi, I&apos;m Dot.</p><p className="mt-1 text-[11px] leading-5 text-zinc-600">Ask me anything to get started.</p></div>}{messages.map((message) => <ChatMessage key={message.id} message={message} onRetry={message.error ? retry : undefined} />)}{loading && messages[messages.length - 1]?.content === "" && <div className="py-2"><span className="block h-1.5 w-1.5 animate-pulse rounded-full bg-violet-400" /></div>}<div ref={messagesEndRef} /></div></div>
        <div className="shrink-0 border-t border-zinc-800/80 p-3"><ChatInput onSubmit={sendMessage} disabled={loading} loading={loading} /></div>
      </motion.section>}
    </AnimatePresence>
    <motion.button whileTap={{ scale: 0.92 }} whileHover={{ scale: 1.05 }} onClick={() => setOpen((value) => !value)} className="fixed bottom-6 right-6 z-50 flex h-14 w-14 items-center justify-center rounded-full bg-violet-600 text-white shadow-xl shadow-violet-900/40" aria-label={open ? "Close Dot" : "Open Dot"}>{open ? <X size={22} /> : <MessageCircle size={23} />}</motion.button>
  </main>;
}
