from functools import lru_cache
import requests
import os
import numexpr

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")


@lru_cache(maxsize=128)
def find_matching_tickers(ticker, market="stocks", limit=100):
    """Find a matching ticker symbol."""
    response = requests.get(
        f"https://api.polygon.io/v3/reference/tickers?search={ticker}&market={market}&limit={limit}&active=true&order=asc&sort=ticker",
        headers={"Authorization": f"Bearer {POLYGON_API_KEY}"},
    )
    if response.status_code != 200:
        raise Exception(
            f"Error fetching tickers: {response.status_code} - {response.text}"
        )
    return response.json() if response.status_code == 200 else []


def parse_query_str(query: str):
    """Parse a query string into a dictionary."""
    return dict(item.split("=") for item in query.split("&"))


def check_query_condition(parsed_query: dict) -> bool:
    """Check if the parsed query meets certain conditions."""
    result = numexpr.evaluate(
        "rsi < 30 and volume > 1000000", local_dict=parsed_query
    ).item()
    return result
