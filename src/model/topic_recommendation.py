import pandas as pd
import numpy as np


# ============================================================
# LOAD DATASETS
# ============================================================

nlp_path = "data/processed/video_nlp_dataset.csv"
prediction_path = "data/processed/viral_predictions.csv"

nlp_df = pd.read_csv(nlp_path)
prediction_df = pd.read_csv(prediction_path)


print("NLP dataset shape:", nlp_df.shape)
print("Prediction dataset shape:", prediction_df.shape)


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
        average_problem_awareness=("problem_awareness_rate", "mean")
    )
    .reset_index()
)


# ============================================================
# VIRAL COUNT BY TOPIC
# ============================================================

viral_counts = (
    nlp_df
    .assign(
        viral=nlp_df["views"] >= nlp_df["views"].quantile(0.90)
    )
    .groupby("topic")["viral"]
    .sum()
    .reset_index(name="viral_count")
)