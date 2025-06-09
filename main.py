from fastapi import FastAPI
from candle_schema import Candle
from datetime import date, datetime
import httpx, joblib, pandas as pd

app = FastAPI()
MODEL = None
METRICS = {
    "accuracy": None,
    "feature_importance": None,
    "last_trained": None
}

@app.on_event("startup")
def load_model():
    global MODEL
    try:
        MODEL = joblib.load("model.pkl")
    except Exception as e:
        print(f"⚠️ Could not load model on startup: {e}")
        MODEL = None

@app.get("/train")
async def train(symbol: str, from_date: date, to_date: date):
    url = f"http://localhost:7070/api/history?symbol={symbol}&from={from_date}&to={to_date}"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        candles = response.json()

    print("📥 Fetched candles:", len(candles))
    from train_model import train_model
    result = train_model(candles)

    # Reload model and update metrics
    global MODEL, METRICS
    MODEL = joblib.load("model.pkl")
    METRICS["accuracy"] = result["accuracy"]
    METRICS["feature_importance"] = result["feature_importance"]
    METRICS["last_trained"] = datetime.now().isoformat()

    return {"message": "Model trained successfully."}

@app.post("/predict")
def predict(candle: Candle):
    global MODEL
    if not MODEL:
        return {"error": "Model not loaded"}
    df = pd.DataFrame([candle.dict()])
    X = df[["open", "high", "low", "close", "volume"]]
    prediction = MODEL.predict(X)[0]
    return {"prediction": int(prediction)}

@app.get("/metrics")
def get_metrics():
    global MODEL, METRICS
    return {
        "model_loaded": MODEL is not None,
        **METRICS
    }
