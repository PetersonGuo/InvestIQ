import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

async function proxy(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> },
) {
  const { path } = await context.params;
  const host = request.headers.get("host") ?? "";
  if (!/^(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$/.test(host)) {
    return NextResponse.json(
      { detail: "StockAssist API is available on localhost only." },
      { status: 403 },
    );
  }
  if (
    ![
      "health",
      "market",
      "search",
      "stocks",
      "alerts",
      "portfolio",
      "order",
      "scanner",
      "strategies",
      "backtests",
      "pairs",
    ].includes(path[0])
  ) {
    return NextResponse.json({ detail: "Not found" }, { status: 404 });
  }
  const origin = request.headers.get("origin");
  if (
    request.method !== "GET" &&
    origin &&
    origin !== `${request.nextUrl.protocol}//${request.headers.get("host")}`
  ) {
    return NextResponse.json(
      { detail: "Cross-origin write rejected" },
      { status: 403 },
    );
  }
  const base = process.env.STOCKASSIST_API_URL || "http://127.0.0.1:8000";
  const prefix = path[0] === "health" ? "/" : "/api/";
  const url = `${base}${prefix}${path.map(encodeURIComponent).join("/")}${request.nextUrl.search}`;
  try {
    const response = await fetch(url, {
      method: request.method,
      headers: { "Content-Type": "application/json" },
      body: ["GET", "HEAD"].includes(request.method)
        ? undefined
        : await request.text(),
      cache: "no-store",
      signal: AbortSignal.timeout(path[0] === "pairs" ? 40000 : 20000),
    });
    return new NextResponse(
      response.status === 204 ? null : await response.text(),
      {
        status: response.status,
        headers: { "Content-Type": "application/json" },
      },
    );
  } catch {
    return NextResponse.json(
      {
        detail:
          "StockAssist API is unavailable. Start the Python backend and retry.",
      },
      { status: 503 },
    );
  }
}
export { proxy as GET, proxy as POST, proxy as PUT, proxy as DELETE };
