import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

# 1. Load data (built into scikit-learn, no download needed)
iris = load_iris(as_frame=True)
X, y = iris.data, iris.target
target_names = iris.target_names
print("Shape:", X.shape)
print(X.describe().round(2))
print("\nClass counts:\n", y.value_counts().sort_index())

# 2. Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Compare several models using 5-fold cross-validation
models = {
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=200)),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5)
    results[name] = scores.mean()
    print(f"{name:20s} CV accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

# 4. Fit the best model and evaluate on the test set
best_name = max(results, key=results.get)
best_model = models[best_name]
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print(f"\nBest model: {best_name}")
print("Test accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("\nClassification report:\n",
      classification_report(y_test, y_pred, target_names=target_names))

# 5. Visualizations
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

axes[0].bar(results.keys(), results.values(), color="teal")
axes[0].set_ylim(0.8, 1.0)
axes[0].set_title("Cross-validation Accuracy")
axes[0].tick_params(axis="x", rotation=25)

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=target_names, ax=axes[1], cmap="Blues", colorbar=False
)
axes[1].set_title(f"Confusion Matrix ({best_name})")

for label, name in enumerate(target_names):
    subset = X[y == label]
    axes[2].scatter(subset.iloc[:, 2], subset.iloc[:, 3], label=name, alpha=0.7)
axes[2].set_xlabel("Petal length (cm)")
axes[2].set_ylabel("Petal width (cm)")
axes[2].set_title("Petal Size by Species")
axes[2].legend()

plt.tight_layout()
plt.savefig("classification_results.png", dpi=150)
plt.show()
