"use client";

import { createElement, useEffect, useState, type ReactNode } from "react";

export type ArticleContent = { text: string; html?: string | null };

const styles: Record<string, string> = {
  p: "my-4", div: "my-2", section: "my-4", article: "my-4",
  h1: "my-5 text-2xl font-semibold", h2: "my-4 text-xl font-semibold",
  h3: "my-3 text-lg font-semibold", h4: "my-3 font-semibold",
  h5: "my-3 font-semibold", h6: "my-3 font-semibold",
  ul: "my-4 list-disc pl-6", ol: "my-4 list-decimal pl-6", li: "my-1",
  blockquote: "my-4 border-l-2 border-slate-500 pl-4 text-slate-400",
  pre: "my-4 overflow-x-auto whitespace-pre-wrap rounded bg-[#0c141e] p-3 text-sm",
  code: "font-mono text-sm", table: "w-full border-collapse text-sm tabular-nums",
  th: "border border-slate-700 p-2 text-left font-semibold",
  td: "border border-slate-700 p-2 align-top", caption: "my-2 text-left text-slate-400",
  a: "text-emerald-300 underline underline-offset-2 hover:text-emerald-200",
  strong: "font-semibold", b: "font-semibold", em: "italic", i: "italic",
  u: "underline", s: "line-through", hr: "my-5 border-slate-700",
};
const allowed = new Set([...Object.keys(styles), "br", "span", "thead", "tbody", "tfoot", "tr", "sup", "sub", "dl", "dt", "dd"]);
const discarded = new Set(["script", "style", "iframe", "object", "embed", "img", "svg", "math", "form", "input", "button", "select", "textarea", "link", "meta", "base", "head", "title", "template", "noscript", "audio", "video", "source"]);

function renderMarkup(markup: string): ReactNode[] {
  // Template content is inert. Only explicitly allowed elements are recreated;
  // provider nodes, CSS, event handlers and other attributes never enter the page.
  const template = document.createElement("template");
  template.innerHTML = markup.slice(0, 128 * 1024);
  let count = 0;
  function render(node: Node, key: string, depth: number): ReactNode {
    if (++count > 10000 || depth > 64) return null;
    if (node.nodeType === Node.TEXT_NODE) return node.textContent;
    if (node.nodeType !== Node.ELEMENT_NODE) return null;
    const element = node as Element;
    const tag = element.tagName.toLowerCase();
    if (discarded.has(tag)) return null;
    const children = Array.from(node.childNodes, (child, index) => render(child, `${key}.${index}`, depth + 1));
    if (!allowed.has(tag)) return children;
    const props: Record<string, unknown> = { key, className: styles[tag] };
    if (tag === "a") {
      const href = element.getAttribute("href")?.trim() ?? "";
      try {
        const url = new URL(href);
        if (url.protocol !== "http:" && url.protocol !== "https:") return children;
        props.href = url.href;
        props.target = "_blank";
        props.rel = "noopener noreferrer";
      } catch { return children; }
    }
    if (tag === "td" || tag === "th") {
      for (const [attribute, property] of [["colspan", "colSpan"], ["rowspan", "rowSpan"]]) {
        const value = Number(element.getAttribute(attribute));
        if (Number.isInteger(value) && value >= 1 && value <= 100) props[property] = value;
      }
    }
    if (tag === "br" || tag === "hr") return createElement(tag, props);
    const rendered = createElement(tag, props, ...children);
    return tag === "table" ? <div key={key} className="my-4 max-w-full overflow-x-auto">{rendered}</div> : rendered;
  }
  return Array.from(template.content.childNodes, (node, index) => render(node, String(index), 0));
}

export default function ArticleBody({ text, html }: ArticleContent) {
  const [rendered, setRendered] = useState<{html: string; nodes: ReactNode[]} | null>(null);
  useEffect(() => {
    setRendered(html ? { html, nodes: renderMarkup(html) } : null);
  }, [html]);
  return <div className="min-w-0 max-w-full text-sm leading-7 text-slate-300 [overflow-wrap:anywhere]" data-article-body>
    {html ? (rendered?.html === html ? rendered.nodes : <p role="status">Formatting article…</p>) : <p className="whitespace-pre-wrap">{text || "The provider returned an empty article."}</p>}
  </div>;
}
