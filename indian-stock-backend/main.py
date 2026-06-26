import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import yfinance as yf
from supabase import create_client, Client

app = FastAPI(title="Indian Stock Sector Sentiment API")

# Enable CORS so your Netlify frontend can talk to your Render backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your specific Netlify URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Supabase Client using environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
HF_SPACE_URL = os.getenv("HF_SPACE_URL")

supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

class SectorRequest(BaseModel):
    sector_name: str  # e.g., "banking", "it", "auto"

@app.get("/")
def read_root():
    return {"message": "Welcome to the Indian Stock Sentiment API"}

@app.get("/health")
def health_check():
    """
    Critical endpoint.
    An external cron job will ping this route every 14 minutes 
    to keep the Render free tier from going to sleep.
    """
    return {"status": "alive"}

@app.post("/api/analyze-sector")
async def analyze_sector(payload: SectorRequest):
    """
    Skeleton route where your core logic will live.
    It will handle pulling data, sending it to Hugging Face, 
    and saving the results to Supabase.
    """
    if not supabase:
        raise HTTPException(status_code=500, detail="Database connection not configured.")
        
    return {
        "sector": payload.sector_name,
        "status": "Pipeline skeleton ready. Next step: integrate data fetching."
    }

if __name__ == "__main__":
    import uvicorn
    # Local development run command
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)