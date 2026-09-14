"use client";
import { useState } from "react";

export default function DocsCode({code, language}: {code: string; language: string}) {
  const [status, setStatus] = useState("Copy");
  return <div className="my-5 overflow-hidden rounded-xl border border-slate-700 bg-[#080f17]">
    <div className="flex items-center justify-between border-b border-slate-800 px-4 py-2 text-xs text-slate-400">
      <span>{language || "Code"}</span>
      <button className="text-emerald-300 hover:text-emerald-200" aria-label={`Copy ${language || "code"} example`} onClick={async () => {
        try { await navigator.clipboard.writeText(code); setStatus("Copied"); }
        catch { setStatus("Select and copy the code below"); }
      }}>{status}</button>
    </div>
    <pre className="overflow-x-auto p-4 text-[13px] leading-6 text-slate-200"><code>{code}</code></pre>
  </div>;
}
