import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_model(data):
    df = pd.DataFrame(data)

    # Create binary target: 1 if next close is higher, else 0
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df.dropna(inplace=True)

    features = df[["open", "high", "low", "close", "volume"]]
    target = df["target"]

    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(features, target)

    # Accuracy
    predictions = model.predict(features)
    accuracy = accuracy_score(target, predictions)
    print(f"✅ Model trained. Training Accuracy: {accuracy:.2%}")

    # Feature importance
    importances = model.feature_importances_
    feature_names = ["open", "high", "low", "close", "volume"]
    for name, imp in zip(feature_names, importances):
        print(f"  {name}: {imp:.4f}")

    # Save model
    joblib.dump(model, "model.pkl")

    # Return metrics
    return {
        "accuracy": float(accuracy),
        "feature_importance": dict(zip(feature_names, importances))
    }
