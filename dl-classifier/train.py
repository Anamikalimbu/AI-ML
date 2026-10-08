import copy
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ---------- Config ----------
SEED = 42
EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-3
PATIENCE = 10  # early stopping

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- 1. Data ----------
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=SEED, stratify=y_train
)

scaler = StandardScaler().fit(X_train)  # fit on train only


def to_loader(X_, y_, shuffle):
    ds = TensorDataset(
        torch.tensor(scaler.transform(X_), dtype=torch.float32),
        torch.tensor(y_, dtype=torch.float32).unsqueeze(1),
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


train_loader = to_loader(X_train, y_train, True)
val_loader = to_loader(X_val, y_val, False)
test_loader = to_loader(X_test, y_test, False)

# ---------- 2. Model ----------
class MLP(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1),  # raw logit
        )

    def forward(self, x):
        return self.net(x)


model = MLP(X.shape[1]).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


def run_epoch(loader, train):
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(xb)
            correct += ((logits > 0).float() == yb).sum().item()
            n += len(xb)
    return total_loss / n, correct / n


# ---------- 3. Train with early stopping ----------
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best_val, best_state, bad_epochs = float("inf"), None, 0

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = run_epoch(train_loader, True)
    va_loss, va_acc = run_epoch(val_loader, False)
    for k, v in zip(history, (tr_loss, va_loss, tr_acc, va_acc)):
        history[k].append(v)
    print(f"Epoch {epoch:3d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} "
          f"| val loss {va_loss:.4f} acc {va_acc:.3f}")

    if va_loss < best_val:
        best_val, best_state, bad_epochs = va_loss, copy.deepcopy(model.state_dict()), 0
    else:
        bad_epochs += 1
        if bad_epochs >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

model.load_state_dict(best_state)

# ---------- 4. Test evaluation ----------
model.eval()
preds = []
with torch.no_grad():
    for xb, _ in test_loader:
        preds.append((model(xb.to(device)) > 0).int().cpu().numpy().ravel())
preds = np.concatenate(preds)

print("\nTest results")
print(classification_report(y_test, preds, target_names=["malignant", "benign"]))

# ---------- 5. Save artifacts ----------
torch.save(model.state_dict(), "model.pt")
joblib.dump(scaler, "scaler.joblib")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(history["train_loss"], label="train")
ax[0].plot(history["val_loss"], label="val")
ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
ax[1].plot(history["train_acc"], label="train")
ax[1].plot(history["val_acc"], label="val")
ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
fig.savefig("training_curves.png", dpi=150, bbox_inches="tight")

ConfusionMatrixDisplay.from_predictions(
    y_test, preds, display_labels=["malignant", "benign"]
)
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
