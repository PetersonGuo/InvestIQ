import { notFound } from "next/navigation";
import DocsView from "@/components/docs-view";
import { guides, type Guide } from "@/lib/docs";
export function generateStaticParams() { return Object.keys(guides).map(guide=>({guide})); }
export default async function GuidePage({params}: {params: Promise<{guide: string}>}) {
  const {guide} = await params;
  if (!Object.hasOwn(guides, guide)) notFound();
  return <DocsView guide={guide as Guide} />;
}
