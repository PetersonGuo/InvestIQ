"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import ArticleBody, { type ArticleContent } from "@/components/article-body";
import { api } from "@/lib/stockassist";
export default function ArticleReader({provider, article, title, publisher}: {provider: string; article: string; title: string; publisher: string}) {
  const [content, setContent] = useState<ArticleContent | null>(null);
  const [error, setError] = useState("");
  const [revision, setRevision] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    setContent(null); setError("");
    if (!provider || !article) { setError("This article link is missing its provider or article ID."); return; }
    void api<ArticleContent>(`news/article?${new URLSearchParams({provider, article})}`, {signal: controller.signal})
      .then(result => { if (!controller.signal.aborted) setContent(result); })
      .catch(e => { if (!controller.signal.aborted) setError(e instanceof Error ? e.message : "Article unavailable."); });
    return () => controller.abort();
  }, [provider, article, revision]);
  return <div className="min-h-screen bg-[#0b1118]"><main className="mx-auto max-w-3xl space-y-6 p-6 text-slate-200">
    <Link href="/" className="text-emerald-300">← Market overview</Link>
    <article className="space-y-5 rounded-2xl border border-slate-700 bg-[#101923] p-6">
      <h1 className="break-words text-2xl font-semibold">{title || "News article"}</h1>
      <p className="text-sm text-slate-400">{publisher || provider} · Supplied by IBKR</p>
      {error ? <div role="alert"><p>{error}</p><button className="mt-3 text-emerald-300" onClick={() => setRevision(v => v + 1)}>Retry article</button></div> : content === null ? <p role="status">Loading article…</p> : <ArticleBody {...content} />}
    </article>
  </main></div>;
}
