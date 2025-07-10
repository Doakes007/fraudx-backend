import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, precision_recall_curve,
    auc, precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split
from imblearn.ensemble import BalancedRandomForestClassifier

# === Step 1: Load dataset ===
print("📥 Loading preprocessed dataset...")
PARQUET_PATH = "../dataset/paysim_preprocessed.parquet"
CSV_FALLBACK = "../dataset/paysim_preprocessed.csv"

if os.path.exists(PARQUET_PATH):
    full_pp = pd.read_parquet(PARQUET_PATH)
elif os.path.exists(CSV_FALLBACK):
    full_pp = pd.read_csv(CSV_FALLBACK)
    print("⚠️ Parquet not found, loaded CSV instead.")
else:
    raise FileNotFoundError("❌ Could not find paysim_preprocessed.parquet or paysim_preprocessed.csv")

# Optional: Save CSV version for review
full_pp.to_csv("../dataset/paysim_preprocessed.csv", index=False)

# === Step 2: Drop unwanted features ===
DROP_COLS = ["step", "hour", "isFlaggedFraud"]
print(f"🧼 Dropping columns: {DROP_COLS}")
full_pp = full_pp.drop(columns=DROP_COLS, errors="ignore")

# === Step 3: Split chronologically (80% train / 20% test) ===
cutoff_index = int(0.8 * len(full_pp))
train_df = full_pp.iloc[:cutoff_index].copy()
test_df = full_pp.iloc[cutoff_index:].copy()

X_train = train_df.drop("isFraud", axis=1)
y_train = train_df["isFraud"]
X_test = test_df.drop("isFraud", axis=1)
y_test = test_df["isFraud"]

# === Step 4: Train model ===
print("🌲 Training BalancedRandomForestClassifier...")
model = BalancedRandomForestClassifier(n_estimators=300, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# === Step 5: Evaluate ===
print("🔍 Evaluating model...")
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n✅ Classification Report:")
print(classification_report(y_test, y_pred, digits=4))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# === Step 6: Visualizations ===

# Confusion Matrix
plt.figure(figsize=(4, 3))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Precision-Recall Curve
prec, rec, _ = precision_recall_curve(y_test, y_proba)
pr_auc = auc(rec, prec)
plt.figure()
plt.plot(rec, prec, label=f"PR AUC = {pr_auc:.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Feature Importances
importances = pd.Series(model.feature_importances_, index=X_train.columns)
importances.nlargest(15).sort_values().plot(kind='barh', figsize=(7, 6))
plt.title("Top 15 Feature Importances — Random Forest")
plt.tight_layout()
plt.show()

# === Step 7: Save model and expected features ===
print("💾 Saving model and feature columns...")
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/random_forest_model.pkl")
joblib.dump(list(X_train.columns), "models/model_columns.pkl")
print("✅ Model & columns saved to /models/")

