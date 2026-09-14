import Link from "next/link";
import { guides, readGuide, type Guide } from "@/lib/docs";
import DocsDocument, { headingId } from "./docs-document";

export default async function DocsView({guide}: {guide: Guide}) {
  const source = await readGuide(guide);
  const headings = source.split("\n").filter(line=>line.startsWith("## ")).map(line=>line.slice(3));
  return <div className="min-h-screen bg-[#0b1118] text-slate-200">
    <header className="border-b border-slate-800 px-5 py-5"><div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-4"><Link href="/" className="font-semibold">StockAssist <span className="ml-2 font-normal text-slate-400">Developer docs</span></Link><Link href="/research" className="text-sm text-emerald-300">Open strategy lab →</Link></div></header>
    <div className="mx-auto grid max-w-7xl gap-8 px-5 py-8 lg:grid-cols-[220px_minmax(0,1fr)]">
      <aside className="min-w-0"><nav aria-label="Developer guides" className="space-y-2 lg:sticky lg:top-6">{Object.entries(guides).map(([key,label])=><Link key={key} href={`/docs/${key}`} aria-current={guide===key ? "page" : undefined} className={`block rounded-lg px-3 py-2 text-sm ${guide===key ? "bg-emerald-400/10 text-emerald-300" : "text-slate-400 hover:bg-slate-800 hover:text-white"}`}>{label}</Link>)}<a href={`/docs/${guide}.md`} download className="block px-3 pt-4 text-xs text-slate-400 underline">Download Markdown</a></nav></aside>
      <main className="min-w-0 max-w-4xl">
        <details className="mb-8 rounded-xl border border-slate-800 px-4 py-3"><summary className="cursor-pointer text-sm text-emerald-300">On this page</summary><nav aria-label="On this page" className="mt-3 grid gap-2 text-sm sm:grid-cols-2">{headings.map(title=><a key={title} href={`#${headingId(title)}`} className="text-slate-400 hover:text-white">{title}</a>)}</nav></details>
        <DocsDocument source={source} />
      </main>
    </div>
  </div>;
}
