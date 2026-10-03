# AI Text Classifier

A simple sentiment classifier built with scikit-learn (TF-IDF + Logistic Regression) and served with FastAPI.

## Setup
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m app.train
uvicorn app.main:app --reload
```

## Usage
Open http://127.0.0.1:8000/docs and try `POST /predict`:
```json
{ "text": "I really love this service" }
```

## Push to GitHub
```bash
git init
git add .
git commit -m "feat: add AI text classifier with scikit-learn and FastAPI"
git branch -M main
git remote add origin https://github.com/<your-username>/ai-text-classifier.git
git push -u origin main
```

## What I learned
- Turning text into numbers with TF-IDF
- Training and saving a model with joblib
- Serving ML predictions through a REST API
