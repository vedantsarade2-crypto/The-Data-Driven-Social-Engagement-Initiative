import pandas as pd
import numpy as np
import os


# ==========================================
# 1. Load dataset
# ==========================================

INPUT_PATH = "data/processed/video_nlp_dataset.csv"

df = pd.read_csv(INPUT_PATH)

print("Dataset shape:", df.shape)


# ==========================================
# 2. Calculate viral threshold
# ==========================================

viral_threshold = df["views"].quantile(0.90)

print("\n========== VIRAL THRESHOLD ==========")

print(
    f"90th percentile views: {viral_threshold:,.0f}"
)


# ==========================================
# 3. Create viral label
# ==========================================

df["viral"] = (
    df["views"] >= viral_threshold
).astype(int)


df["viral_label"] = df["viral"].map(
    {
        0: "Non-Viral",
        1: "Viral"
    }
)


# ==========================================
# 4. Count viral videos
# ==========================================

viral_counts = (
    df["viral_label"]
    .value_counts()
)


print("\n========== VIRAL DISTRIBUTION ==========")

print(viral_counts)


# ==========================================
# 5. Compare viral vs non-viral
# ==========================================

comparison = (
    df.groupby("viral_label")
    [
        [
            "views",
            "likes",
            "comments_count",
            "engagement_rate",
            "average_sentiment",
            "relatability_rate",
            "problem_awareness_rate"
        ]
    ]
    .mean()
)


print("\n========== VIRAL VS NON-VIRAL ==========")

print(
    comparison.round(3)
)


# ==========================================
# 6. Median comparison
# ==========================================

median_comparison = (
    df.groupby("viral_label")
    [
        [
            "views",
            "likes",
            "comments_count",
            "engagement_rate",
            "average_sentiment",
            "relatability_rate",
            "problem_awareness_rate"
        ]
    ]
    .median()
)


print("\n========== MEDIAN COMPARISON ==========")

print(
    median_comparison.round(3)
)


# ==========================================
# 7. Viral topic distribution
# ==========================================

viral_topics = (
    df[df["viral"] == 1]["topic"]
    .value_counts()
)


print("\n========== VIRAL VIDEOS BY TOPIC ==========")

print(viral_topics)


# ==========================================
# 8. Save dataset
# ==========================================

os.makedirs(
    "data/processed",
    exist_ok=True
)

df.to_csv(
    "data/processed/video_viral_dataset.csv",
    index=False
)


# ==========================================
# 9. Save summary
# ==========================================

comparison.to_csv(
    "reports/eda/viral_comparison.csv"
)


print("\nViral analysis completed.")

print(
    "\nViral threshold:",
    f"{viral_threshold:,.0f} views"
)

print(
    "Viral videos:",
    df["viral"].sum()
)

print(
    "Saved to:",
    "data/processed/video_viral_dataset.csv"
)