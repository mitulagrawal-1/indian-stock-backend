import os
import asyncio
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import yfinance as yf
from supabase import create_client, Client

app = FastAPI(title="Indian Stock Sector Sentiment API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_last_trading_day() -> date:
    today = date.today()
    # Monday=0, Sunday=6
    if today.weekday() == 5:  # Saturday
        return today - timedelta(days=1)
    elif today.weekday() == 6:  # Sunday
        return today - timedelta(days=2)
    else:  # Weekday, get previous day
        return today - timedelta(days=1)


SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
HF_SPACE_URL = os.getenv("HF_SPACE_URL")

supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

class SectorRequest(BaseModel):
    sector_name: str 

SECTOR_TICKERS = {
    "Banking": "^NSEBANK",
    "IT": "^CNXIT",
    "Auto": "^CNXAUTO",
    "Pharma": "^CNXPHARMA",
    "FMCG": "^CNXFMCG"
}

SECTOR_QUERIES = {
    "Banking": "Nifty Bank OR HDFC Bank OR SBI OR ICICI Bank business news",
    "IT": "Nifty IT OR TCS OR Infosys OR Wipro stock news",
    "Auto": "Nifty Auto OR Tata Motors OR Maruti Suzuki OR Mahindra stock news",
    "Pharma": "Nifty Pharma OR Sun Pharma OR Dr Reddys stock news",
    "FMCG": "Nifty FMCG OR ITC share OR Hindustan Unilever news"
}

@app.post("/api/analyze-sector")
async def analyze_sector(payload: SectorRequest):
    # Ensure database is configured
    if not supabase:
        raise HTTPException(status_code=500, detail="Database credentials missing on Render.")
    
    name = payload.sector_name
    if name not in SECTOR_TICKERS:
        raise HTTPException(status_code=400, detail=f"Sector '{name}' is not supported.")
        
    ticker_symbol = SECTOR_TICKERS[name]
    query_string = SECTOR_QUERIES[name]
    
    # Initialize all variables at the very top of the function scope
    close_price = 0.0
    pct_change = 0.0
    headlines = []
    avg_sentiment = 0.0
    sentiment_label = "Neutral"
    rss_url = f"https://news.google.com/rss/search?q={query_string}&hl=en-IN&gl=IN&ceid=IN:en"

    fetch_date = get_last_trading_day()
    print(f"DEBUG: Fetching data for {fetch_date.isoformat()}")

    try:
        print(f"DEBUG: Starting pipeline for {name}")

        # 1. Fetch Indian Market Data
        ticker = yf.Ticker(ticker_symbol)

        # Fetch enough history to ensure we get the last trading day and its previous trading day
        # Max period of 5 days should cover most weekend/holiday scenarios
        hist = ticker.history(start=fetch_date - timedelta(days=5), end=fetch_date + timedelta(days=1))

        if hist.empty or len(hist) < 2:
            raise HTTPException(status_code=500, detail=f"Insufficient historical data for {ticker_symbol} around {fetch_date.isoformat()}")
        else:
            # Filter for the actual fetch_date and its immediate preceding trading day
            actual_trading_days = hist[hist.index <= str(fetch_date)].tail(2)

            if len(actual_trading_days) < 2:
                raise HTTPException(status_code=500, detail=f"Insufficient historical data for {ticker_symbol} after filtering around {fetch_date.isoformat()}")            else:
                close_price = round(actual_trading_days['Close'].iloc[-1], 2)
                prev_close = actual_trading_days['Close'].iloc[-2]
                pct_change = round(((close_price - prev_close) / prev_close) * 100, 2)
            
        # 2. Scrape News
        print(f"DEBUG: Fetching news from {rss_url}")
        async with httpx.AsyncClient() as client:
            rss_response = await client.get(rss_url, timeout=10.0)
            if rss_response.status_code == 200:
                root = ET.fromstring(rss_response.content)
                for item in root.findall('.//item')[:5]:
                    title = item.find('title').text
                    if " - " in title:
                        title = title.rsplit(" - ", 1)[0]
                    headlines.append(title)
        
        if not headlines:
            headlines = [f"Tracking general indices for the Indian {name} sector."]

        # 3. Sentiment Analysis
        if HF_SPACE_URL and headlines:
            print("DEBUG: Sending to HF Space")
            async with httpx.AsyncClient() as client:
                tasks = [client.post(f"{HF_SPACE_URL.rstrip('/')}/analyze", json={"text": t}, timeout=15.0) for t in headlines]
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                
                total_score, valid_count = 0.0, 0
                for res in responses:
                    if isinstance(res, httpx.Response) and res.status_code == 200:
                        data = res.json()
                        conf = data.get("confidence", 1.0)
                        if data.get("sentiment", "neutral").lower() == "positive": total_score += conf
                        elif data.get("sentiment", "neutral").lower() == "negative": total_score -= conf
                        valid_count += 1
                
                if valid_count > 0:
                    avg_sentiment = round(total_score / valid_count, 2)
                    if avg_sentiment > 0.15: sentiment_label = "Positive"
                    elif avg_sentiment < -0.15: sentiment_label = "Negative"

        # 4. Save to Supabase
        print("DEBUG: Saving to Supabase")
        data_to_insert = {
            "sector_name": name, "ticker": ticker_symbol, "close_price": close_price,
            "pct_change": pct_change, "avg_sentiment_score": avg_sentiment,
            "sentiment_label": sentiment_label, "headlines": headlines 
        }
        supabase.table("sector_analyses").insert(data_to_insert).execute()
        
        return {"status": "success", "data": data_to_insert}
        
    except Exception as e:
        print(f"DEBUG: CRITICAL ERROR - {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))