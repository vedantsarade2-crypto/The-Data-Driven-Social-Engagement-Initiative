import pandas as pd

# ============================================================
# MODEL PERFORMANCE COMPARISON
# ============================================================

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest"
    ],
    "Accuracy": [
        0.6750,
        0.8500
    ],
    "Precision": [
        0.1538,
        0.0000
    ],
    "Recall": [
        0.5000,
        0.0000
    ],
    "F1 Score": [
        0.2353,
        0.0000
    ],
    "ROC-AUC": [
        0.5417,
        0.4236
    ],
    "PR-AUC": [
        None,
        0.1058
    ]
})

print("\n========== MODEL COMPARISON ==========")

print(
    results.to_string(index=False)
)


# ============================================================
# BEST MODEL BY IMPORTANT METRICS
# ============================================================

print("\n========== BEST MODELS ==========")

print(
    "Best Accuracy:",
    results.loc[
        results["Accuracy"].idxmax(),
        "Model"
    ]
)

print(
    "Best Precision:",
    results.loc[
        results["Precision"].idxmax(),
        "Model"
    ]
)

print(
    "Best Recall:",
    results.loc[
        results["Recall"].idxmax(),
        "Model"
    ]
)

print(
    "Best F1 Score:",
    results.loc[
        results["F1 Score"].idxmax(),
        "Model"
    ]
)

print(
    "Best ROC-AUC:",
    results.loc[
        results["ROC-AUC"].idxmax(),
        "Model"
    ]
)


# ============================================================
# SAVE COMPARISON
# ============================================================

results.to_csv(
    "reports/model_comparison.csv",
    index=False
)

print("\nModel comparison saved to:")
print("reports/model_comparison.csv")