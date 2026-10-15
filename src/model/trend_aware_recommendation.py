import os
import pandas as pd
import numpy as np


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

TOPIC_FILE = "data/processed/topic_performance.csv"
PREDICTION_FILE = "data/processed/viral_predictions.csv"
TREND_FILE = "data/processed/rising_topics.csv"

OUTPUT_DIR = "data/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------
# 1. Load datasets
# ---------------------------------------------------------

topic_df = pd.read_csv(TOPIC_FILE)
prediction_df = pd.read_csv(PREDICTION_FILE)
trend_df = pd.read_csv(TREND_FILE)

print("Topic performance:", topic_df.shape)
print("ML predictions:", prediction_df.shape)
print("Trend data:", trend_df.shape)


# ---------------------------------------------------------
# 2. Prepare ML prediction data
# ---------------------------------------------------------

prediction_summary = (
    prediction_df
    .groupby("topic")
    .agg(
        average_ml_probability=(
            "viral_probability",
            "mean"
        )
    )
    .reset_index()
)


# ---------------------------------------------------------
# 3. Prepare historical topic performance
# ---------------------------------------------------------

historical = topic_df[
    [
        "topic",
        "median_views",
        "viral_rate",
        "recommendation_score"
    ]
].copy()


# ---------------------------------------------------------
# 4. Prepare trend data
# ---------------------------------------------------------

trend = trend_df[
    [
        "keyword",
        "average_search_growth",
        "average_hashtag_growth",
        "latest_trend_score",
        "rising_score",
        "rank"
    ]
].copy()


trend = trend.rename(
    columns={
        "keyword": "topic"
    }
)


# ---------------------------------------------------------
# 5. Merge datasets
# ---------------------------------------------------------

combined = historical.merge(
    prediction_summary,
    on="topic",
    how="left"
)

combined = combined.merge(
    trend,
    on="topic",
    how="left"
)


# ---------------------------------------------------------
# 6. Handle topics without trend information
# ---------------------------------------------------------

trend_columns = [
    "average_search_growth",
    "average_hashtag_growth",
    "latest_trend_score",
    "rising_score"
]

for column in trend_columns:
    combined[column] = combined[column].fillna(0)


combined["rank"] = combined["rank"].fillna(
    len(combined) + 1
)


# ---------------------------------------------------------
# 7. Normalize components
# ---------------------------------------------------------

def normalize(series):

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return pd.Series(
            np.ones(len(series)),
            index=series.index
        )

    return (
        (series - minimum)
        / (maximum - minimum)
    )


combined["historical_score"] = normalize(
    combined["recommendation_score"]
)

combined["ml_score"] = normalize(
    combined["average_ml_probability"]
)

combined["trend_score_normalized"] = normalize(
    combined["rising_score"]
)


# ---------------------------------------------------------
# 8. Calculate final trend-aware score
# ---------------------------------------------------------

combined["final_recommendation_score"] = (
    0.40 * combined["historical_score"]
    + 0.30 * combined["ml_score"]
    + 0.30 * combined["trend_score_normalized"]
)


# ---------------------------------------------------------
# 9. Rank topics
# ---------------------------------------------------------

combined = combined.sort_values(
    "final_recommendation_score",
    ascending=False
).reset_index(drop=True)

combined["final_rank"] = (
    combined.index + 1
)


# ---------------------------------------------------------
# 10. Recommendation reasoning
# ---------------------------------------------------------

def generate_reason(row):

    reasons = []

    if row["historical_score"] >= 0.7:
        reasons.append(
            "strong historical content performance"
        )

    if row["ml_score"] >= 0.7:
        reasons.append(
            "high predicted viral potential"
        )

    if row["trend_score_normalized"] >= 0.7:
        reasons.append(
            "strong current trend momentum"
        )

    if not reasons:
        reasons.append(
            "moderate historical, ML, and trend signals"
        )

    return "; ".join(reasons)


combined["reasoning"] = combined.apply(
    generate_reason,
    axis=1
)


# ---------------------------------------------------------
# 11. Save results
# ---------------------------------------------------------

output_file = os.path.join(
    OUTPUT_DIR,
    "trend_aware_recommendations.csv"
)

combined.to_csv(
    output_file,
    index=False
)


# ---------------------------------------------------------
# 12. Display recommendations
# ---------------------------------------------------------

display_columns = [
    "final_rank",
    "topic",
    "final_recommendation_score",
    "historical_score",
    "ml_score",
    "trend_score_normalized",
    "average_search_growth",
    "average_hashtag_growth",
    "reasoning"
]

print("\nFinal Trend-Aware Recommendations:\n")

print(
    combined[
        display_columns
    ].to_string(index=False)
)


print(
    f"\nSaved to: {output_file}"
)