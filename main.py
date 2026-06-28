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

        market_movement = "Neutral"
        if pct_change > 0.01:  # Define a threshold for 'Up'
            market_movement = "Up"
        elif pct_change < -0.01: # Define a threshold for 'Down'
            market_movement = "Down"

        data_to_insert = {
            "sector_name": name, "ticker": ticker_symbol, "close_price": close_price,
            "pct_change": pct_change, "avg_sentiment_score": avg_sentiment,
            "sentiment_label": sentiment_label, "headlines": headlines,
            "market_movement": market_movement # New field
        }
        supabase.table("sector_analyses").insert(data_to_insert).execute()
        
        return {"status": "success", "data": data_to_insert}

@app.get("/api/historical-analyses")
async def get_historical_analyses(limit: int = 30):
    if not supabase:
        raise HTTPException(status_code=500, detail="Database credentials missing on Render.")

    try:
        # Fetch recent analyses, ordered by creation time descending
        response = supabase.table("sector_analyses")\
                           .select("*")\
                           .order("created_at", desc=True)\
                           .limit(limit)\
                           .execute()

    

@app.post("/api/predict-movement")
async def predict_movement(payload: dict):
    if not supabase:
        raise HTTPException(status_code=500, detail="Database credentials missing on Render.")

    sector_name = payload.get("sector_name")
    start_date_str = payload.get("start_date")
    end_date_str = payload.get("end_date")

    if not all([sector_name, start_date_str, end_date_str]):
        raise HTTPException(status_code=400, detail="Missing required fields: sector_name, start_date, end_date")

    try:
        from joblib import load
        import numpy as np
        from datetime import datetime

        # Load the trained model
        model_filename = "sentiment_model.pkl"
        try:
            model = load(model_filename)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Sentiment model not found. Please train the model first.")

        # Convert date strings to date objects
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()

        # Fetch historical data for the prediction period
        # We need to query Supabase directly here to get data for specific dates
        # Using a date range query on 'created_at' or a similar date field if available
        # For now, assume sector_analyses table has a date column we can filter on, or we use fetched data
        # If 'created_at' is not a date, we might need to add one or rely on fetched data that already has dates.
        # Let's assume we can query based on 'created_at' which is usually a timestamp, and we extract date from it.

        # Fetching analyses for the prediction period.
        # Assuming created_at is a timestamp that can be compared.
        # Adjusting query to fetch data for the specific date range
        # NOTE: The existing historical_analyses endpoint fetches data in reverse chronological order.
        # We need data that aligns with specific dates for prediction.
        # A more robust solution would involve a date column in sector_analyses and direct DB query.
        # For now, let's fetch a larger set and filter by date manually.
        # Fetching analyses for the sector, covering a slightly larger range to be safe.
        sector_analyses_response = supabase.table("sector_analyses")\
                                       .select("*")\
                                       .eq("sector_name", sector_name)

        # To filter by date, we need to compare dates. Assuming sector_analyses has a date field
        # If not, we might need to parse it from created_at or add a date column.
        # For now, let's fetch all for the sector and filter client-side or assume created_at can be filtered
        # Using .gt() and .lt() for date filtering. This assumes created_at is a date/timestamp.
        results = sector_analyses_response.gt("created_at", start_date.strftime('%Y-%m-%d') + 'T00:00:00Z')\
                                       .lt("created_at", (end_date + timedelta(days=1)).strftime('%Y-%m-%d') + 'T00:00:00Z')\
                                       .order("created_at", asc=True)
                                       .execute()

        prediction_data = results.data

        if not prediction_data:
            raise HTTPException(status_code=404, detail=f"No historical data found for sector '{sector_name}' between {start_date_str} and {end_date_str}")

        # Prepare data for prediction
        features_for_prediction = []
        dates_in_period = []
        actual_movements = []

        # Map target labels to numerical values for consistency if needed, though model expects numerical output
        target_map_inv = {1: "Up", -1: "Down", 0: "Neutral"}

        # Temporary storage to align predictions with actual data
        predictions = []
        actual_data_by_date = {}

        # Process fetched data to prepare for prediction and store actuals
        for record in prediction_data:
            # Assuming 'created_at' can be parsed to get the date
            record_date_str = record.get('created_at')
            if record_date_str:
                try:
                    record_date = datetime.fromisoformat(record_str.replace('Z', '+00:00')).date() # Handle Z for UTC
                except ValueError:
                    print(f"DEBUG: Could not parse date from created_at: {record_date_str}")
                    continue

                # Only consider records within the requested date range
                if start_date <= record_date <= end_date:
                    sentiment_score = record.get('avg_sentiment_score')
                    actual_movement = record.get('market_movement')

                    if sentiment_score is not None:
                        # Ensure feature is in the correct format (list of lists)
                        features_for_prediction.append([sentiment_score])
                        dates_in_period.append(record_date.isoformat())
                        actual_movements.append(actual_movement)
                        actual_data_by_date[record_date.isoformat()] = {
                            "close_price": record.get('close_price'),
                            "pct_change": record.get('pct_change'),
                            "actual_movement": actual_movement,
                            "sentiment_label": record.get('sentiment_label'),
                            "headlines": record.get('headlines', [])
                        }
                    else:
                         print(f"DEBUG: Skipping record for {record_date.isoformat()} due to missing sentiment score.")
            else:
                print(f"DEBUG: Skipping record due to missing 'created_at' field: {record}")

        if not features_for_prediction:
            raise HTTPException(status_code=404, detail=f"No valid data points found for prediction within the date range {start_date_str} to {end_date_str} for sector '{sector_name}'.")

        # Make predictions
        predicted_numerical = model.predict(np.array(features_for_prediction))

        # Convert numerical predictions back to labels
        predicted_labels = [target_map_inv.get(p, "Unknown") for p in predicted_numerical]

        # Combine predictions with actual data and dates
        results_list = []
        for i in range(len(dates_in_period)):
            prediction_date = dates_in_period[i]
            actual_record = actual_data_by_date.get(prediction_date)

            if actual_record: # Ensure we have actual data for this date
                results_list.append({
                    "date": prediction_date,
                    "predicted_movement": predicted_labels[i],
                    "predicted_sentiment_score": features_for_prediction[i][0],
                    "actual_data": actual_record
                })
            else:
                print(f"DEBUG: Skipping prediction for {prediction_date} as no corresponding actual data was found.")

        return {"status": "success", "predictions": results_list}

    except HTTPException as http_e:
        raise http_e
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Sentiment model file not found. Please train the model first.")
    except Exception as e:
        print(f"DEBUG: ERROR during prediction - {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/train-model")
async def train_model():
    if not supabase:
        raise HTTPException(status_code=500, detail="Database credentials missing on Render.")

    try:
        # 1. Fetch Historical Data (last 30 days)
        # Use limit of 30 for approximately one month of data
        historical_data_response = await get_historical_analyses(limit=30)
        data = historical_data_response.get("data", [])

        if not data or len(data) < 2: # Need at least two data points to train
            raise HTTPException(status_code=400, detail="Not enough historical data (need at least 2 days) to train the model.")

        # 2. Data Preprocessing
        # For simplicity, we'll use avg_sentiment_score as a feature and market_movement as the target
        features = []
        targets = []

        for record in data:
            # Ensure all necessary fields are present and valid
            if (
                'avg_sentiment_score' in record and record['avg_sentiment_score'] is not None and
                'market_movement' in record and record['market_movement'] in ['Up', 'Down', 'Neutral']
            ):
                features.append([record['avg_sentiment_score']]) # Feature is a list of lists for scikit-learn
                target_map = {"Up": 1, "Down": -1, "Neutral": 0}
                targets.append(target_map[record['market_movement']])
            else:
                print(f"DEBUG: Skipping record due to missing or invalid data: {record}")

        if not features or len(features) < 2:
            raise HTTPException(status_code=400, detail="Not enough valid historical data points for training after processing.")

        # 3. Model Training
        from sklearn.linear_model import LogisticRegression
        from joblib import dump
        import numpy as np

        model = LogisticRegression()
        model.fit(np.array(features), np.array(targets))

        # 4. Model Saving
        model_filename = "sentiment_model.pkl"
        dump(model, model_filename)

        # Store model path or model itself if needed globally (for simplicity, we assume it's accessible)
        # In a real app, you'd load this model in prediction endpoint

        return {"status": "success", "message": "Model trained and saved successfully.", "model_filename": model_filename}

    except HTTPException as http_e:
        # Re-raise HTTPExceptions to propagate them
        raise http_e
    except Exception as e:
        print(f"DEBUG: ERROR during model training - {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        print(f"DEBUG: CRITICAL ERROR - {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        print(f"DEBUG: CRITICAL ERROR - {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))