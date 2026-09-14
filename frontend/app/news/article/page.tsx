import ArticleReader from "@/components/article-reader";
export default async function ArticlePage({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const params = await searchParams;
  const value = (key: string) => typeof params[key] === "string" ? params[key] as string : "";
  return <ArticleReader provider={value("provider")} article={value("article")} title={value("title")} publisher={value("publisher")} />;
}
