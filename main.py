from fastapi import FastAPI
from candle_schema import Candle
from datetime import date
import httpx, joblib, pandas as pd

app = FastAPI()
MODEL = None

@app.on_event("startup")
def load_model():
    global MODEL
    try:
        MODEL = joblib.load("model.pkl")
    except:
        MODEL = None

@app.get("/train")
async def train(symbol: str, from_date: date, to_date: date):
    url = f"http://localhost:7070/api/history?symbol={symbol}&from={from_date}&to={to_date}"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        candles = response.json()

    from train_model import train_model
    train_model(candles)

    # 🔁 Load model after training
    global MODEL
    MODEL = joblib.load("model.pkl")

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
