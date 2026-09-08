def test_company_fundamentals_and_news(client):
    fundamentals = client.get("/api/stocks/AAPL/fundamentals")
    assert fundamentals.status_code == 200
    data = fundamentals.json()
    assert data["ticker"] == "AAPL" and data["source"] == "demo"
    assert data["metrics"][0]["period_end"]
    assert data["metrics"][0]["filed"]
    assert "Synthetic" in data["notes"][0]
    news = client.get("/api/stocks/MSFT/news").json()
    assert news["source"] == "demo" and news["ticker"] == "MSFT"
    assert all("synthetic" in row["title"] for row in news["articles"])
    assert client.get("/api/stocks/UNKNOWN/fundamentals").status_code == 404
    assert client.get("/api/stocks/UNKNOWN/news").status_code == 404
    assert client.get("/api/news/article?provider=x&article=y").status_code == 422
