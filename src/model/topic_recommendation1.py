import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

NLP_PATH = "data/processed/video_nlp_dataset.csv"
PREDICTION_PATH = "data/processed/viral_predictions.csv"

OUTPUT_PATH = "data/processed/topic_performance.csv"


# ============================================================
# LOAD DATA
# ============================================================

nlp_df = pd.read_csv(NLP_PATH)
prediction_df = pd.read_csv(PREDICTION_PATH)

print("NLP dataset shape:", nlp_df.shape)
print("Prediction dataset shape:", prediction_df.shape)


# ============================================================
# DEFINE VIRAL THRESHOLD
# ============================================================

viral_threshold = nlp_df["views"].quantile(0.90)

print("\nViral views threshold:")
print(f"{viral_threshold:,.0f}")


# ============================================================
# CREATE VIRAL FLAG
# ============================================================

nlp_df["viral"] = (
    nlp_df["views"] >= viral_threshold
).astype(int)


# ============================================================
# TOPIC PERFORMANCE
# ============================================================

topic_performance = (
    nlp_df
    .groupby("topic")
    .agg(
        videos=("video_id", "count"),
        average_views=("views", "mean"),
        median_views=("views", "median"),
        average_engagement=("engagement_rate", "mean"),
        average_sentiment=("average_sentiment", "mean"),
        average_relatability=("relatability_rate", "mean"),
        average_problem_awareness=("problem_awareness_rate", "mean"),
        viral_count=("viral", "sum")
    )
    .reset_index()
)


# ============================================================
# VIRAL RATE
# ============================================================

topic_performance["viral_rate"] = (
    topic_performance["viral_count"]
    / topic_performance["videos"]
)


# ============================================================
# AVERAGE ML VIRAL PROBABILITY BY TOPIC
# ============================================================

prediction_topic = (
    prediction_df
    .groupby("topic")
    .agg(
        average_ml_probability=(
            "viral_probability",
            "mean"
        ),
        high_potential_count=(
            "viral_potential",
            lambda x: (x == "High").sum()
        )
    )
    .reset_index()
)


# ============================================================
# MERGE PERFORMANCE + ML RESULTS
# ============================================================

topic_performance = topic_performance.merge(
    prediction_topic,
    on="topic",
    how="left"
)


# ============================================================
# NORMALIZATION FUNCTION
# ============================================================

def min_max_normalize(series):

    min_value = series.min()
    max_value = series.max()

    if max_value == min_value:
        return pd.Series(
            1.0,
            index=series.index
        )

    return (
        (series - min_value)
        / (max_value - min_value)
    )


# ============================================================
# NORMALIZE IMPORTANT METRICS
# ============================================================

topic_performance["views_score"] = (
    min_max_normalize(
        topic_performance["median_views"]
    )
)

topic_performance["viral_score"] = (
    min_max_normalize(
        topic_performance["viral_rate"]
    )
)

topic_performance["relatability_score"] = (
    min_max_normalize(
        topic_performance["average_relatability"]
    )
)

topic_performance["problem_awareness_score"] = (
    min_max_normalize(
        topic_performance["average_problem_awareness"]
    )
)

topic_performance["ml_score"] = (
    topic_performance["average_ml_probability"]
)


# ============================================================
# RECOMMENDATION SCORE
# ============================================================

topic_performance["recommendation_score"] = (

    0.30 * topic_performance["views_score"]

    + 0.25 * topic_performance["viral_score"]

    + 0.20 * topic_performance["ml_score"]

    + 0.15 * topic_performance["relatability_score"]

    + 0.10 * topic_performance["problem_awareness_score"]
)


# ============================================================
# SORT TOPICS
# ============================================================

topic_performance = topic_performance.sort_values(
    by="recommendation_score",
    ascending=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n========================================")
print("TOPIC RECOMMENDATION ANALYSIS")
print("========================================")

display_columns = [
    "topic",
    "videos",
    "median_views",
    "viral_count",
    "viral_rate",
    "average_relatability",
    "average_problem_awareness",
    "average_ml_probability",
    "recommendation_score"
]

print(
    topic_performance[
        display_columns
    ].to_string(index=False)
)


# ============================================================
# TOP RECOMMENDED TOPICS
# ============================================================

print("\n========================================")
print("TOP RECOMMENDED TOPICS")
print("========================================")

print(
    topic_performance[
        [
            "topic",
            "recommendation_score"
        ]
    ]
    .head(5)
    .to_string(index=False)
)


# ============================================================
# SAVE RESULTS
# ============================================================

topic_performance.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nTopic performance saved to:")
print(OUTPUT_PATH)