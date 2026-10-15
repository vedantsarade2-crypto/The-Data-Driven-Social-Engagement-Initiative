import pandas as pd


# ==========================================
# 1. Load datasets
# ==========================================

comments = pd.read_csv(
    "data/processed/nlp_comments.csv"
)

videos = pd.read_csv(
    "data/processed/video_features.csv"
)


# ==========================================
# 2. Aggregate NLP metrics by video
# ==========================================

nlp_summary = (
    comments
    .groupby("video_id")
    .agg(
        total_comments=("video_id", "count"),

        average_sentiment=(
            "sentiment_score",
            "mean"
        ),

        relatable_comments=(
            "relatability_score",
            "sum"
        ),

        relatability_rate=(
            "relatability_score",
            "mean"
        ),

        problem_aware_comments=(
            "problem_awareness_score",
            "sum"
        ),

        problem_awareness_rate=(
            "problem_awareness_score",
            "mean"
        )
    )
    .reset_index()
)


# ==========================================
# 3. Convert rates to percentages
# ==========================================

nlp_summary["relatability_rate"] = (
    nlp_summary["relatability_rate"] * 100
)

nlp_summary["problem_awareness_rate"] = (
    nlp_summary["problem_awareness_rate"] * 100
)


# ==========================================
# 4. Merge with video performance
# ==========================================

video_nlp = pd.merge(
    videos,
    nlp_summary,
    on="video_id",
    how="left"
)


# ==========================================
# 5. Fill missing NLP values
# ==========================================

numeric_columns = [
    "total_comments",
    "average_sentiment",
    "relatable_comments",
    "relatability_rate",
    "problem_aware_comments",
    "problem_awareness_rate"
]

for column in numeric_columns:

    video_nlp[column] = (
        video_nlp[column]
        .fillna(0)
    )


# ==========================================
# 6. Save
# ==========================================

OUTPUT_PATH = (
    "data/processed/video_nlp_dataset.csv"
)

video_nlp.to_csv(
    OUTPUT_PATH,
    index=False
)


# ==========================================
# 7. Display
# ==========================================

print("\n========== VIDEO NLP DATASET ==========")

print(
    video_nlp.shape
)

print("\nColumns:")
print(
    video_nlp.columns.tolist()
)

print("\nTop videos by relatability rate:")

print(
    video_nlp[
        [
            "video_id",
            "topic",
            "views",
            "relatability_rate",
            "problem_awareness_rate",
            "average_sentiment"
        ]
    ]
    .sort_values(
        "relatability_rate",
        ascending=False
    )
    .head(10)
    .to_string(index=False)
)


print("\nSaved to:")
print(OUTPUT_PATH)