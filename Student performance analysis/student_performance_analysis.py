import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Create sample data
np.random.seed(42)
n = 200
df = pd.DataFrame({
    "study_hours": np.random.uniform(1, 10, n).round(1),
    "attendance": np.random.randint(50, 100, n),
    "sleep_hours": np.random.uniform(4, 9, n).round(1),
})
df["score"] = (
    df["study_hours"] * 5
    + df["attendance"] * 0.3
    + np.random.normal(0, 5, n)
).clip(0, 100).round(1)

# 2. Basic exploration
print(df.head())
print("\nSummary statistics:\n", df.describe())
print("\nMissing values:\n", df.isnull().sum())

# 3. Correlation
corr = df.corr()
print("\nCorrelation with score:\n", corr["score"].sort_values(ascending=False))

# 4. Group analysis
df["study_level"] = pd.cut(df["study_hours"], bins=[0, 4, 7, 10],
                           labels=["Low", "Medium", "High"])
print("\nAverage score by study level:\n",
      df.groupby("study_level", observed=True)["score"].mean().round(2))

# 5. Visualizations
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(df["score"], bins=20, color="skyblue", edgecolor="black")
axes[0].set_title("Score Distribution")
axes[0].set_xlabel("Score")

axes[1].scatter(df["study_hours"], df["score"], alpha=0.6, color="teal")
m, b = np.polyfit(df["study_hours"], df["score"], 1)
axes[1].plot(df["study_hours"], m * df["study_hours"] + b, color="red")
axes[1].set_title("Study Hours vs Score")
axes[1].set_xlabel("Study Hours")
axes[1].set_ylabel("Score")

im = axes[2].imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
axes[2].set_xticks(range(len(corr)))
axes[2].set_xticklabels(corr.columns, rotation=45, ha="right")
axes[2].set_yticks(range(len(corr)))
axes[2].set_yticklabels(corr.columns)
axes[2].set_title("Correlation Heatmap")
fig.colorbar(im, ax=axes[2])

plt.tight_layout()
plt.savefig("analysis_results.png", dpi=150)
plt.show()
