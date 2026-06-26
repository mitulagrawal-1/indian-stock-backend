import os
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

# Initialize Supabase
supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

class SectorRequest(BaseModel):
    sector_name: str  # e.g., "IT", "Banking"

# Map frontend readable names to Yahoo Finance NSE Tickers
SECTOR_TICKERS = {
    "Banking": "^NSEBANK",
    "IT": "^CNXIT",
    "Auto": "^CNXAUTO",
    "Pharma": "^CNXPHARMA",
    "FMCG": "^CNXFMCG"
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
    
    try:
        # 1. Fetch Current Market Data from yfinance
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period="2d") # Get today and yesterday to find percentage change
        
        if len(hist) < 2:
            raise HTTPException(status_code=500, detail="Not enough market data returned from yfinance.")
            
        close_price = round(hist['Close'].iloc[-1], 2)
        prev_close = hist['Close'].iloc[-2]
        pct_change = round(((close_price - prev_close) / prev_close) * 100, 2)
        
        # 2. Phase 1 Placeholder for News & Hugging Face Sentiment
        # (We will replace this mock data with the Google News scraper next)
        mock_headlines = [
            f"Indian {name} companies show robust growth projections this quarter.",
            f"Market analysts remain cautious over global impacts on NSE {name} index."
        ]
        
        # Mocking evaluation scores until we hook up the live HF async call
        avg_sentiment = 0.45 
        sentiment_label = "Positive"

        # 3. Save the results directly into Supabase
        data_to_insert = {
            "sector_name": name,
            "ticker": ticker_symbol,
            "close_price": close_price,
            "pct_change": pct_change,
            "avg_sentiment_score": avg_sentiment,
            "sentiment_label": sentiment_label,
            "headlines": mock_headlines
        }
        
        response = supabase.table("sector_analyses").insert(data_to_insert).execute()
        
        return {
            "status": "success",
            "saved_data": data_to_insert
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))