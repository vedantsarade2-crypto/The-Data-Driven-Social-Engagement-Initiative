import pandas as pd
import os


# ---------------------------------------
# File paths
# ---------------------------------------

VIDEO_SEARCH_PATH = "data/raw/youtube/video_search_results.csv"
VIDEO_STATS_PATH = "data/raw/youtube/video_statistics.csv"
COMMENTS_PATH = "data/raw/comments/youtube_comments.csv"


# ---------------------------------------
# Load datasets
# ---------------------------------------

print("Loading datasets...")

video_search = pd.read_csv(VIDEO_SEARCH_PATH)
video_stats = pd.read_csv(VIDEO_STATS_PATH)
comments = pd.read_csv(COMMENTS_PATH)


# ---------------------------------------
# Basic information
# ---------------------------------------

print("\n========== VIDEO SEARCH DATA ==========")
print(video_search.shape)
print(video_search.columns.tolist())

print("\n========== VIDEO STATISTICS ==========")
print(video_stats.shape)
print(video_stats.columns.tolist())

print("\n========== COMMENTS DATA ==========")
print(comments.shape)
print(comments.columns.tolist())


# ---------------------------------------
# Missing values
# ---------------------------------------

print("\n========== MISSING VALUES ==========")

print("\nVideo Search:")
print(video_search.isnull().sum())

print("\nVideo Statistics:")
print(video_stats.isnull().sum())

print("\nComments:")
print(comments.isnull().sum())


# ---------------------------------------
# Duplicate checks
# ---------------------------------------

print("\n========== DUPLICATES ==========")

print(
    "Duplicate videos:",
    video_search["video_id"].duplicated().sum()
)

print(
    "Duplicate statistics:",
    video_stats["video_id"].duplicated().sum()
)

print(
    "Duplicate comments:",
    comments.duplicated().sum()
)


# ---------------------------------------
# Unique video IDs
# ---------------------------------------

print("\n========== UNIQUE VIDEO IDS ==========")

print(
    "Search videos:",
    video_search["video_id"].nunique()
)

print(
    "Statistics videos:",
    video_stats["video_id"].nunique()
)

print(
    "Comment videos:",
    comments["video_id"].nunique()
)


# ---------------------------------------
# Check negative values
# ---------------------------------------

numeric_columns = [
    "views",
    "likes",
    "comments_count"
]

print("\n========== NEGATIVE VALUES ==========")

for column in numeric_columns:

    if column in video_stats.columns:

        negative_count = (
            video_stats[column] < 0
        ).sum()

        print(
            f"{column}: {negative_count}"
        )


# ---------------------------------------
# Check comments
# ---------------------------------------

if "comment_text" in comments.columns:

    empty_comments = (
        comments["comment_text"]
        .fillna("")
        .str.strip()
        .eq("")
        .sum()
    )

    print(
        "\nEmpty comments:",
        empty_comments
    )


print("\n========== VALIDATION COMPLETE ==========")