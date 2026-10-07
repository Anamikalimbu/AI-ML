import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 1. Load data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Candidate models
models = {
    "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

# 3. Train and evaluate
best_name, best_model, best_acc = None, None, 0
for name, model in models.items():
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{name}: accuracy = {acc:.4f}")
    if acc > best_acc:
        best_name, best_model, best_acc = name, model, acc

# 4. Report for the best model
print(f"\nBest model: {best_name}")
print(classification_report(y_test, best_model.predict(X_test)))

# 5. Save the model and confusion matrix
joblib.dump(best_model, "model.joblib")
ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test)
plt.title(best_name)
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
