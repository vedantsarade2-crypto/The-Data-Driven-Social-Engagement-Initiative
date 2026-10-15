import pandas as pd
import numpy as np
import os

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ==================================================
# Configuration
# ==================================================

TRAIN_PATH = "data/processed/train_data.csv"
TEST_PATH = "data/processed/test_data.csv"

REPORT_PATH = "reports/model_baseline.txt"

os.makedirs("reports", exist_ok=True)

# ==================================================
# Load Data
# ==================================================

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("Training shape:", train_df.shape)
print("Testing shape:", test_df.shape)

# ==================================================
# Separate X and y
# ==================================================

X_train = train_df.drop(columns=["viral"])
y_train = train_df["viral"]

X_test = test_df.drop(columns=["viral"])
y_test = test_df["viral"]

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())

# ==================================================
# Train Logistic Regression
# ==================================================

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# ==================================================
# Predictions
# ==================================================

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]

# ==================================================
# Evaluation Metrics
# ==================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

# ROC-AUC requires both classes in test set
if len(np.unique(y_test)) == 2:
    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )
else:
    roc_auc = np.nan

cm = confusion_matrix(
    y_test,
    y_pred
)

# ==================================================
# Print Results
# ==================================================

print("\n========== LOGISTIC REGRESSION ==========")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\n========== CONFUSION MATRIX ==========")
print(cm)

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Non-Viral",
            "Viral"
        ],
        zero_division=0
    )
)

# ==================================================
# Save Model Report
# ==================================================

with open(REPORT_PATH, "w") as file:

    file.write("LOGISTIC REGRESSION BASELINE\n")
    file.write("============================\n\n")

    file.write(f"Accuracy : {accuracy:.4f}\n")
    file.write(f"Precision: {precision:.4f}\n")
    file.write(f"Recall   : {recall:.4f}\n")
    file.write(f"F1 Score : {f1:.4f}\n")
    file.write(f"ROC-AUC  : {roc_auc:.4f}\n\n")

    file.write("Confusion Matrix:\n")
    file.write(str(cm))
    file.write("\n\n")

    file.write("Classification Report:\n")
    file.write(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Non-Viral",
                "Viral"
            ],
            zero_division=0
        )
    )

print("\nBaseline model report saved to:")
print(REPORT_PATH)

print("\nBaseline model training completed.")