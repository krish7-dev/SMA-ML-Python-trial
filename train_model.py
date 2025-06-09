import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier

def train_model(data):
    df = pd.DataFrame(data)
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df.dropna(inplace=True)

    features = df[["open", "high", "low", "close", "volume"]]
    target = df["target"]

    model = DecisionTreeClassifier()
    model.fit(features, target)

    joblib.dump(model, "model.pkl")
    print("✅ Model trained and saved.")
