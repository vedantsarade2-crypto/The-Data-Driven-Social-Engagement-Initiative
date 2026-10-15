import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# LOAD TRAINING AND TEST DATA
# ============================================================

train_path = "data/processed/train_data.csv"
test_path = "data/processed/test_data.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

X_train = train_df.drop(columns=["viral"])
y_train = train_df["viral"]

X_test = test_df.drop(columns=["viral"])
y_test = test_df["viral"]


print("Training shape:", X_train.shape)
print("Testing shape :", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())


# ============================================================
# RANDOM FOREST MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# ============================================================
# PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


# ============================================================
# EVALUATION
# ============================================================

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

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

pr_auc = average_precision_score(
    y_test,
    y_prob
)


print("\n========== RANDOM FOREST ==========")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")
print(f"PR-AUC   : {pr_auc:.4f}")


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("\n========== CONFUSION MATRIX ==========")
print(cm)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Non-Viral", "Viral"],
        zero_division=0
    )
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

print("\n========== TOP FEATURE IMPORTANCE ==========")
print(feature_importance.head(15))


# ============================================================
# SAVE FEATURE IMPORTANCE
# ============================================================

feature_importance.to_csv(
    "reports/random_forest_feature_importance.csv",
    index=False
)


# ============================================================
# SAVE MODEL REPORT
# ============================================================

with open("reports/model_random_forest.txt", "w") as f:

    f.write("RANDOM FOREST MODEL REPORT\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Accuracy : {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall   : {recall:.4f}\n")
    f.write(f"F1 Score : {f1:.4f}\n")
    f.write(f"ROC-AUC  : {roc_auc:.4f}\n")
    f.write(f"PR-AUC   : {pr_auc:.4f}\n\n")

    f.write("CONFUSION MATRIX\n")
    f.write(str(cm))
    f.write("\n\n")

    f.write("CLASSIFICATION REPORT\n")
    f.write(
        classification_report(
            y_test,
            y_pred,
            target_names=["Non-Viral", "Viral"],
            zero_division=0
        )
    )

    f.write("\n\nFEATURE IMPORTANCE\n")
    f.write(feature_importance.to_string(index=False))


print("\nRandom Forest report saved to:")
print("reports/model_random_forest.txt")

print("\nFeature importance saved to:")
print("reports/random_forest_feature_importance.csv")

print("\nRandom Forest training completed.")