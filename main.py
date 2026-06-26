import os
import asyncio
import xml.etree.ElementTree as ET
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

# Targeted search strings designed for Indian financial media outlets
SECTOR_QUERIES = {
    "Banking": "Nifty Bank OR HDFC Bank OR SBI OR ICICI Bank business news",
    "IT": "Nifty IT OR TCS OR Infosys OR Wipro stock news",
    "Auto": "Nifty Auto OR Tata Motors OR Maruti Suzuki OR Mahindra stock news",
    "Pharma": "Nifty Pharma OR Sun Pharma OR Dr Reddys stock news",
    "FMCG": "Nifty FMCG OR ITC share OR Hindustan Unilever news"
}

@app.get("/health")
def health_check():
    return {"status": "alive"}

@app.post("/api/analyze-sector")
async def analyze_sector(payload: SectorRequest):
    if not supabase:
        raise HTTPException(status_code=500, detail="Database credentials missing on Render.")
    
    name = payload.sector_name
    if name not in SECTOR_TICKERS:
        raise HTTPException(status_code=400, detail=f"Sector '{name}' is not supported.")
        
    ticker_symbol = SECTOR_TICKERS[name]
    query_string = SECTOR_QUERIES[name]
    
    try:
        # 1. Fetch Indian Market Data via yfinance
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period="2d")
        
        if len(hist) < 2:
            raise HTTPException(status_code=500, detail="Insufficient market data returned from yfinance.")
            
        close_price = round(hist['Close'].iloc[-1], 2)
        prev_close = hist['Close'].iloc[-2]
        pct_change = round(((close_price - prev_close) / prev_close) * 100, 2)
        
        # 2. Scrape Live Indian Financial News via Google News RSS Feed
        rss_url = f"https://news.google.com/rss/search?q={query_string}&hl=en-IN&gl=IN&ceid=IN:en"
        headlines = []
        
        async with httpx.AsyncClient() as client:
            rss_response = await client.get(rss_url)
            if rss_response.status_code == 200:
                root = ET.fromstring(rss_response.text)
                # Parse and extract the top 5 relevant headlines
                for item in root.findall('.//item')[:5]:
                    title = item.find('title').text
                    # Clean the source tracking suffix out of the headline string
                    if " - " in title:
                        title = title.rsplit(" - ", 1)[0]
                    headlines.append(title)
                    
        if not headlines:
            headlines = [f"Tracking general indices for the Indian {name} sector."]

        # 3. Concurrent Evaluation via Hugging Face Space (FinBERT)
        avg_sentiment = 0.0
        sentiment_label = "Neutral"
        
        if HF_SPACE_URL and headlines:
            async with httpx.AsyncClient() as client:
                tasks = []
                for text in headlines:
                    url = f"{HF_SPACE_URL.rstrip('/')}/analyze"
                    tasks.append(client.post(url, json={"text": text}, timeout=15.0))
                
                # Fire all HTTP requests to your Space simultaneously for low latency
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                
                total_score = 0.0
                valid_responses = 0
                
                for res in responses:
                    if isinstance(res, httpx.Response) and res.status_code == 200:
                        data = res.json()
                        label = data.get("sentiment", "neutral").lower()
                        confidence = data.get("confidence", 1.0)
                        
                        # Map text categories to numerical vector values
                        if label == "positive":
                            total_score += (1.0 * confidence)
                        elif label == "negative":
                            total_score += (-1.0 * confidence)
                        
                        valid_responses += 1
                
                if valid_responses > 0:
                    avg_sentiment = round(total_score / valid_responses, 2)
            
            # Categorize the final composite score
            if avg_sentiment > 0.15:
                sentiment_label = "Positive"
            elif avg_sentiment < -0.15:
                sentiment_label = "Negative"

        # 4. Save Final Real-Time Output directly to Supabase
        data_to_insert = {
            "sector_name": name,
            "ticker": ticker_symbol,
            "close_price": close_price,
            "pct_change": pct_change,
            "avg_sentiment_score": avg_sentiment,
            "sentiment_label": sentiment_label,
            "headlines": headlines 
        }
        
        supabase.table("sector_analyses").insert(data_to_insert).execute()
        
        return {
            "status": "success",
            "data": data_to_insert
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))