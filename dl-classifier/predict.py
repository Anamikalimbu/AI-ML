import joblib
import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer

# Same architecture as in train.py
model = nn.Sequential(
    nn.Linear(30, 64), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(32, 1),
)
state = torch.load("model.pt", map_location="cpu")
model.load_state_dict({k.replace("net.", ""): v for k, v in state.items()})
model.eval()
scaler = joblib.load("scaler.joblib")

# Demo: predict the first 5 samples
X, y = load_breast_cancer(return_X_y=True)
x = torch.tensor(scaler.transform(X[:5]), dtype=torch.float32)
with torch.no_grad():
    probs = torch.sigmoid(model(x)).squeeze(1)

for p, true in zip(probs, y[:5]):
    label = "benign" if p > 0.5 else "malignant"
    print(f"predicted: {label} (p_benign={p:.3f}) | actual: {'benign' if true else 'malignant'}")
