import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# Small demo dataset: replace with your own data later
texts = [
    "I love this product, it works great",
    "Amazing service and very friendly staff",
    "This is the best day ever",
    "Really happy with the quality",
    "Excellent experience, highly recommend",
    "Terrible service, I am very disappointed",
    "This is the worst product I have bought",
    "I hate waiting so long, awful experience",
    "Very bad quality and a waste of money",
    "Horrible, never coming back again",
]
labels = ["positive"] * 5 + ["negative"] * 5

model = make_pipeline(TfidfVectorizer(), LogisticRegression())
model.fit(texts, labels)

Path("model").mkdir(exist_ok=True)
joblib.dump(model, "model/classifier.joblib")
print("Model trained and saved to model/classifier.joblib")
