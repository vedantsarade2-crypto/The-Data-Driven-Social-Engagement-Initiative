import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. Load Dataset
# ==========================================

INPUT_PATH = "data/processed/video_features.csv"

df = pd.read_csv(INPUT_PATH)

print("Dataset loaded successfully.")

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ==========================================
# 2. Basic Information
# ==========================================

print("\n========== DATA TYPES ==========")
print(df.dtypes)


print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())


print("\n========== DESCRIPTIVE STATISTICS ==========")
print(df.describe())


# ==========================================
# 3. Create EDA Output Directory
# ==========================================

EDA_DIR = "reports/eda"

os.makedirs(
    EDA_DIR,
    exist_ok=True
)


# ==========================================
# 4. Topic Distribution
# ==========================================

topic_counts = df["topic"].value_counts()

print("\n========== VIDEOS BY TOPIC ==========")
print(topic_counts)


plt.figure(figsize=(10, 6))

topic_counts.plot(
    kind="bar"
)

plt.title("Number of Videos by Topic")
plt.xlabel("Topic")
plt.ylabel("Number of Videos")

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    f"{EDA_DIR}/videos_by_topic.png",
    dpi=300
)

plt.close()


# ==========================================
# 5. Views Distribution
# ==========================================

plt.figure(figsize=(10, 6))

sns.histplot(
    df["views"],
    bins=30,
    kde=True
)

plt.title("Distribution of YouTube Views")
plt.xlabel("Views")
plt.ylabel("Number of Videos")

plt.tight_layout()

plt.savefig(
    f"{EDA_DIR}/views_distribution.png",
    dpi=300
)

plt.close()


# ==========================================
# 6. Log Views Distribution
# ==========================================

plt.figure(figsize=(10, 6))

sns.histplot(
    df["log_views"],
    bins=30,
    kde=True
)

plt.title("Distribution of Log-Transformed Views")
plt.xlabel("Log(Views + 1)")
plt.ylabel("Number of Videos")

plt.tight_layout()

plt.savefig(
    f"{EDA_DIR}/log_views_distribution.png",
    dpi=300
)

plt.close()


# ==========================================
# 7. Engagement Rate Distribution
# ==========================================

plt.figure(figsize=(10, 6))

sns.histplot(
    df["engagement_rate"],
    bins=30,
    kde=True
)

plt.title("Distribution of Engagement Rate")
plt.xlabel("Engagement Rate (%)")
plt.ylabel("Number of Videos")

plt.tight_layout()

plt.savefig(
    f"{EDA_DIR}/engagement_rate_distribution.png",
    dpi=300
)

plt.close()


# ==========================================
# 8. Views by Topic
# ==========================================

topic_views = (
    df.groupby("topic")["views"]
    .mean()
    .sort_values(
        ascending=False
    )
)

print("\n========== AVERAGE VIEWS BY TOPIC ==========")
print(topic_views)


plt.figure(figsize=(10, 6))

topic_views.plot(
    kind="bar"
)

plt.title("Average Views by Topic")
plt.xlabel("Topic")
plt.ylabel("Average Views")

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    f"{EDA_DIR}/average_views_by_topic.png",
    dpi=300
)

plt.close()


# ==========================================
# 9. Engagement by Topic
# ==========================================

topic_engagement = (
    df.groupby("topic")["engagement_rate"]
    .mean()
    .sort_values(
        ascending=False
    )
)

print("\n========== AVERAGE ENGAGEMENT BY TOPIC ==========")
print(topic_engagement)


plt.figure(figsize=(10, 6))

topic_engagement.plot(
    kind="bar"
)

plt.title("Average Engagement Rate by Topic")
plt.xlabel("Topic")
plt.ylabel("Average Engagement Rate (%)")

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    f"{EDA_DIR}/engagement_by_topic.png",
    dpi=300
)

plt.close()


# ==========================================
# 10. Likes vs Views
# ==========================================

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="views",
    y="likes"
)

plt.title("Likes vs Views")
plt.xlabel("Views")
plt.ylabel("Likes")

plt.tight_layout()

plt.savefig(
    f"{EDA_DIR}/likes_vs_views.png",
    dpi=300
)

plt.close()


# ==========================================
# 11. Comments vs Views
# ==========================================

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="views",
    y="comments_count"
)

plt.title("Comments vs Views")
plt.xlabel("Views")
plt.ylabel("Comments")

plt.tight_layout()

plt.savefig(
    f"{EDA_DIR}/comments_vs_views.png",
    dpi=300
)

plt.close()


# ==========================================
# 12. Engagement vs Views
# ==========================================

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="log_views",
    y="engagement_rate"
)

plt.title("Engagement Rate vs Log Views")
plt.xlabel("Log(Views + 1)")
plt.ylabel("Engagement Rate (%)")

plt.tight_layout()

plt.savefig(
    f"{EDA_DIR}/engagement_vs_views.png",
    dpi=300
)

plt.close()


# ==========================================
# 13. Top 10 Videos by Views
# ==========================================

top_videos = (
    df.sort_values(
        "views",
        ascending=False
    )
    .head(10)
)

print("\n========== TOP 10 VIDEOS BY VIEWS ==========")

print(
    top_videos[
        [
            "video_id",
            "topic",
            "title",
            "views",
            "likes",
            "comments_count",
            "engagement_rate"
        ]
    ].to_string(index=False)
)


# ==========================================
# 14. Top 10 Videos by Engagement
# ==========================================

top_engagement = (
    df.sort_values(
        "engagement_rate",
        ascending=False
    )
    .head(10)
)

print("\n========== TOP 10 VIDEOS BY ENGAGEMENT ==========")

print(
    top_engagement[
        [
            "video_id",
            "topic",
            "title",
            "views",
            "likes",
            "comments_count",
            "engagement_rate"
        ]
    ].to_string(index=False)
)


# ==========================================
# 15. Correlation Analysis
# ==========================================

correlation_columns = [
    "views",
    "likes",
    "comments_count",
    "engagement_rate",
    "like_rate",
    "comment_rate",
    "log_views"
]

correlation_matrix = (
    df[correlation_columns]
    .corr()
)


print("\n========== CORRELATION MATRIX ==========")
print(correlation_matrix)


plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f"
)

plt.title("Correlation Matrix")

plt.tight_layout()

plt.savefig(
    f"{EDA_DIR}/correlation_matrix.png",
    dpi=300
)

plt.close()


# ==========================================
# 16. Save Topic Summary
# ==========================================

topic_summary = (
    df.groupby("topic")
    .agg(
        video_count=("video_id", "count"),
        average_views=("views", "mean"),
        median_views=("views", "median"),
        average_likes=("likes", "mean"),
        average_comments=("comments_count", "mean"),
        average_engagement=("engagement_rate", "mean"),
        average_like_rate=("like_rate", "mean"),
        average_comment_rate=("comment_rate", "mean")
    )
    .sort_values(
        "average_views",
        ascending=False
    )
)


topic_summary.to_csv(
    f"{EDA_DIR}/topic_summary.csv"
)


# ==========================================
# 17. Completion Message
# ==========================================

print("\n==========================================")
print("EDA COMPLETED SUCCESSFULLY")
print("==========================================")

print("\nEDA files saved in:")
print(EDA_DIR)