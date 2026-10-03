import joblib
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="AI Text Classifier")

MODEL_PATH = Path("model/classifier.joblib")
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


class TextInput(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "AI Text Classifier API is running"}


@app.post("/predict")
def predict(data: TextInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not trained yet. Run: python -m app.train")
    if not data.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    prediction = model.predict([data.text])[0]
    confidence = float(max(model.predict_proba([data.text])[0]))
    return {"text": data.text, "prediction": prediction, "confidence": round(confidence, 3)}
