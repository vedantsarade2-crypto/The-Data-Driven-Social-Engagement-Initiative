import pandas as pd
import numpy as np
from scipy import stats


# ==========================================
# 1. Load dataset
# ==========================================

INPUT_PATH = "data/processed/video_nlp_dataset.csv"

df = pd.read_csv(INPUT_PATH)

print("Dataset loaded:")
print(df.shape)


# ==========================================
# 2. Basic statistics
# ==========================================

print("\n========== BASIC STATISTICS ==========")

print(
    df[
        [
            "views",
            "likes",
            "comments_count",
            "engagement_rate",
            "average_sentiment",
            "relatability_rate",
            "problem_awareness_rate"
        ]
    ].describe()
)


# ==========================================
# 3. Correlation analysis
# ==========================================

correlation_columns = [
    "views",
    "likes",
    "comments_count",
    "engagement_rate",
    "average_sentiment",
    "relatability_rate",
    "problem_awareness_rate"
]

correlation_matrix = (
    df[correlation_columns]
    .corr(method="spearman")
)


print("\n========== SPEARMAN CORRELATION ==========")

print(
    correlation_matrix.round(3)
)


# ==========================================
# 4. Spearman tests
# ==========================================

variables = [
    "relatability_rate",
    "problem_awareness_rate",
    "average_sentiment",
    "engagement_rate"
]

print("\n========== RELATIONSHIP WITH VIEWS ==========")

for variable in variables:

    valid_data = df[
        [
            "views",
            variable
        ]
    ].dropna()

    correlation, p_value = stats.spearmanr(
        valid_data["views"],
        valid_data[variable]
    )

    print(
        f"\n{variable}"
    )

    print(
        f"Correlation: {correlation:.4f}"
    )

    print(
        f"P-value: {p_value:.4f}"
    )


# ==========================================
# 5. Statistical significance
# ==========================================

print("\n========== SIGNIFICANCE ==========")

print(
    "We use alpha = 0.05"
)

print(
    "p < 0.05 → statistically significant"
)

print(
    "p >= 0.05 → not statistically significant"
)


# ==========================================
# 6. Save correlation matrix
# ==========================================

correlation_matrix.to_csv(
    "reports/eda/nlp_correlation_matrix.csv"
)


print(
    "\nCorrelation matrix saved."
)


# ==========================================
# 7. Completion
# ==========================================

print("\n==========================================")
print("STATISTICAL ANALYSIS COMPLETED")
print("==========================================")