import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# LOAD DATA
# ============================================================

train_df = pd.read_csv("data/processed/train_data.csv")
test_df = pd.read_csv("data/processed/test_data.csv")

# Combine training and testing data only for cross-validation
# This is acceptable because CV will create fresh validation folds.
full_df = pd.concat(
    [train_df, test_df],
    ignore_index=True
)

X = full_df.drop(columns=["viral"])
y = full_df["viral"]

print("Dataset shape:", X.shape)

print("\nTarget distribution:")
print(y.value_counts())


# ============================================================
# DEFINE MODELS
# ============================================================

logistic_model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "model",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        )
    )
])


random_forest_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


# ============================================================
# STRATIFIED K-FOLD
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# EVALUATION METRICS
# ============================================================

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision"
}


# ============================================================
# LOGISTIC REGRESSION CV
# ============================================================

print("\n========================================")
print("LOGISTIC REGRESSION CROSS-VALIDATION")
print("========================================")

logistic_cv = cross_validate(
    logistic_model,
    X,
    y,
    cv=cv,
    scoring=scoring,
    n_jobs=-1
)


# ============================================================
# RANDOM FOREST CV
# ============================================================

print("\n========================================")
print("RANDOM FOREST CROSS-VALIDATION")
print("========================================")

rf_cv = cross_validate(
    random_forest_model,
    X,
    y,
    cv=cv,
    scoring=scoring,
    n_jobs=-1
)


# ============================================================
# FUNCTION TO SUMMARIZE RESULTS
# ============================================================

def summarize_cv(name, results):

    print(f"\n========== {name} ==========")

    summary = {}

    for metric in scoring.keys():

        values = results[f"test_{metric}"]

        mean_value = np.mean(values)
        std_value = np.std(values)

        summary[metric] = {
            "mean": mean_value,
            "std": std_value
        }

        print(
            f"{metric.upper():10s}: "
            f"{mean_value:.4f} ± {std_value:.4f}"
        )

    return summary


# ============================================================
# SUMMARIZE BOTH MODELS
# ============================================================

logistic_summary = summarize_cv(
    "LOGISTIC REGRESSION",
    logistic_cv
)

rf_summary = summarize_cv(
    "RANDOM FOREST",
    rf_cv
)


# ============================================================
# CREATE COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "Random Forest"
    ],

    "Accuracy Mean": [
        logistic_summary["accuracy"]["mean"],
        rf_summary["accuracy"]["mean"]
    ],

    "Accuracy Std": [
        logistic_summary["accuracy"]["std"],
        rf_summary["accuracy"]["std"]
    ],

    "Precision Mean": [
        logistic_summary["precision"]["mean"],
        rf_summary["precision"]["mean"]
    ],

    "Precision Std": [
        logistic_summary["precision"]["std"],
        rf_summary["precision"]["std"]
    ],

    "Recall Mean": [
        logistic_summary["recall"]["mean"],
        rf_summary["recall"]["mean"]
    ],

    "Recall Std": [
        logistic_summary["recall"]["std"],
        rf_summary["recall"]["std"]
    ],

    "F1 Mean": [
        logistic_summary["f1"]["mean"],
        rf_summary["f1"]["mean"]
    ],

    "F1 Std": [
        logistic_summary["f1"]["std"],
        rf_summary["f1"]["std"]
    ],

    "ROC-AUC Mean": [
        logistic_summary["roc_auc"]["mean"],
        rf_summary["roc_auc"]["mean"]
    ],

    "ROC-AUC Std": [
        logistic_summary["roc_auc"]["std"],
        rf_summary["roc_auc"]["std"]
    ],

    "PR-AUC Mean": [
        logistic_summary["pr_auc"]["mean"],
        rf_summary["pr_auc"]["mean"]
    ],

    "PR-AUC Std": [
        logistic_summary["pr_auc"]["std"],
        rf_summary["pr_auc"]["std"]
    ]
})


# ============================================================
# PRINT COMPARISON
# ============================================================

print("\n========================================")
print("CROSS-VALIDATION MODEL COMPARISON")
print("========================================")

print(
    comparison.to_string(index=False)
)


# ============================================================
# SAVE RESULTS
# ============================================================

comparison.to_csv(
    "reports/cross_validation_comparison.csv",
    index=False
)

print("\nCross-validation results saved to:")
print("reports/cross_validation_comparison.csv")