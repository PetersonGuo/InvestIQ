import type { ReactNode } from "react";
import DocsCode from "./docs-code";

export function headingId(text: string) {
  return text.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

function inline(text: string): ReactNode[] {
  // The guides use a small Markdown subset: code, emphasis and links.
  // Render as React nodes, with no raw HTML handling.
  return text.split(/(`[^`]+`|\*\*[^*]+\*\*|\[[^\]]+\]\([^)]+\))/g).map((part, index) => {
    if (part.startsWith("`")) return <code key={index} className="rounded bg-slate-800 px-1 py-0.5 text-sm text-emerald-200">{part.slice(1,-1)}</code>;
    if (part.startsWith("**")) return <strong key={index} className="font-semibold text-white">{part.slice(2,-2)}</strong>;
    const link = part.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
    if (link) {
      const href = link[2].endsWith(".md") ? `/docs/${link[2].slice(0,-3)}` : link[2];
      return <a key={index} href={href} className="text-emerald-300 underline underline-offset-4">{link[1]}</a>;
    }
    return part;
  });
}

export default function DocsDocument({source}: {source: string}) {
  const lines = source.split("\n"), blocks: ReactNode[] = [];
  let i = 0;
  const cells = (line: string) => line.trim().replace(/^\||\|$/g, "").split("|").map(v=>v.trim());
  while (i < lines.length) {
    const line = lines[i], key = i;
    if (!line.trim()) { i++; continue; }
    if (line.startsWith("```")) {
      const language = line.slice(3).trim(), code: string[] = [];
      i++;
      while (i < lines.length && !lines[i].startsWith("```")) code.push(lines[i++]);
      i++;
      blocks.push(<DocsCode key={key} code={code.join("\n")} language={language} />);
    } else if (/^#{1,3} /.test(line)) {
      const level = line.indexOf(" "), text = line.slice(level+1), id = headingId(text);
      blocks.push(level === 1 ? <h1 key={key} id={id} className="mb-6 scroll-mt-6 text-3xl font-semibold tracking-tight text-white">{inline(text)}</h1> : level === 2 ? <h2 key={key} id={id} className="mb-4 mt-10 scroll-mt-6 border-t border-slate-800 pt-6 text-xl font-semibold text-white">{inline(text)}</h2> : <h3 key={key} id={id} className="mb-3 mt-6 scroll-mt-6 text-lg font-semibold text-white">{inline(text)}</h3>);
      i++;
    } else if (line.startsWith("|") && /^\|[\s:|\-]+\|$/.test(lines[i+1] || "")) {
      const headers = cells(line), rows: string[][] = [];
      i += 2;
      while (i < lines.length && lines[i].startsWith("|")) rows.push(cells(lines[i++]));
      blocks.push(<div key={key} className="my-5 max-w-full overflow-x-auto rounded-xl border border-slate-700"><table className="w-full min-w-[480px] border-collapse text-left text-sm"><thead className="bg-slate-800/60"><tr>{headers.map((cell,n)=><th key={n} className="px-4 py-3 font-semibold text-white">{inline(cell)}</th>)}</tr></thead><tbody>{rows.map((row,n)=><tr key={n} className="border-t border-slate-800">{row.map((cell,k)=><td key={k} className="px-4 py-3 align-top leading-6">{inline(cell)}</td>)}</tr>)}</tbody></table></div>);
    } else if (line.startsWith("- ")) {
      const items: string[]=[];
      while (i < lines.length && lines[i].startsWith("- ")) items.push(lines[i++].slice(2));
      blocks.push(<ul key={key} className="my-4 list-disc space-y-2 pl-6">{items.map((item,n)=><li key={n}>{inline(item)}</li>)}</ul>);
    } else {
      const paragraph: string[]=[];
      while (i < lines.length && lines[i].trim() && !/^(#|```|\||- )/.test(lines[i])) paragraph.push(lines[i++]);
      if (!paragraph.length) { paragraph.push(lines[i]); i++; }
      blocks.push(<p key={key} className="my-4 leading-7">{inline(paragraph.join(" "))}</p>);
    }
  }
  return <article className="min-w-0 text-slate-300 [overflow-wrap:anywhere]">{blocks}</article>;
}
