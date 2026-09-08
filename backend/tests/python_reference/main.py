import asyncio
import logging
from contextlib import asynccontextmanager, suppress
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import JSONResponse
from core.config import DATA_MODE
from core.database import initialize
from api.trading import router
from services.alerts import check_alerts
from services import backtests, pairs
from api.research import router as research_router
from api.pairs import router as pair_router


async def alert_loop():
    while True:
        for checker in (check_alerts, pairs.check_pair_alerts):
            try:
                await asyncio.to_thread(checker)
            except Exception:
                logging.getLogger(__name__).exception(
                    "Alert worker check failed; retrying on the next cycle"
                )
        await asyncio.sleep(30)


@asynccontextmanager
async def lifespan(app):
    initialize()
    backtests.initialize()
    pairs.initialize()
    task = asyncio.create_task(alert_loop())
    yield
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task
    if DATA_MODE == "ibkr":
        from services.ibkr import provider

        await asyncio.to_thread(provider.close)


app = FastAPI(title="StockAssist", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type"],
)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"],
)


@app.middleware("http")
async def protect_local_writes(request, call_next):
    if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        origin = request.headers.get("origin")
        if origin and origin not in {"http://127.0.0.1:3000", "http://localhost:3000"}:
            return JSONResponse(
                {"detail": "Cross-origin write rejected."}, status_code=403
            )
        if (
            request.method != "DELETE"
            and request.headers.get("content-type", "").split(";")[0].strip()
            != "application/json"
        ):
            return JSONResponse(
                {"detail": "Content-Type must be application/json."}, status_code=415
            )
    return await call_next(request)


app.include_router(router)
app.include_router(research_router)
app.include_router(pair_router)


@app.get("/health")
@app.get("/")
def health():
    return {
        "status": "ok",
        "data_mode": DATA_MODE,
        "trading_mode": "paper",
        "account_mode": "local-single-user",
    }


@app.get("/api/market/status")
def market_status():
    if DATA_MODE == "ibkr":
        from services.ibkr import provider

        return provider.status()
    return {
        "provider": DATA_MODE,
        "connected": None,
        "price_type": "completed_daily_close",
    }
