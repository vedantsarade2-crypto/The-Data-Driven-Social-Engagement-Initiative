import pandas as pd


INPUT_PATH = "data/processed/master_video_dataset.csv"
OUTPUT_PATH = "data/processed/video_features.csv"


# ---------------------------------------
# Load master dataset
# ---------------------------------------

df = pd.read_csv(INPUT_PATH)


# ---------------------------------------
# Handle missing numeric values
# ---------------------------------------

numeric_columns = [
    "views",
    "likes",
    "comments_count"
]

for column in numeric_columns:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        df[column] = df[column].fillna(0)


# ---------------------------------------
# Engagement Rate
# ---------------------------------------

df["engagement_rate"] = 0.0

valid_views = df["views"] > 0

df.loc[valid_views, "engagement_rate"] = (
    (
        df.loc[valid_views, "likes"]
        + df.loc[valid_views, "comments_count"]
    )
    / df.loc[valid_views, "views"]
) * 100


# ---------------------------------------
# Like Rate
# ---------------------------------------

df["like_rate"] = 0.0

df.loc[valid_views, "like_rate"] = (
    df.loc[valid_views, "likes"]
    / df.loc[valid_views, "views"]
) * 100


# ---------------------------------------
# Comment Rate
# ---------------------------------------

df["comment_rate"] = 0.0

df.loc[valid_views, "comment_rate"] = (
    df.loc[valid_views, "comments_count"]
    / df.loc[valid_views, "views"]
) * 100


# ---------------------------------------
# Log Views
# ---------------------------------------

import numpy as np

df["log_views"] = np.log1p(
    df["views"]
)


# ---------------------------------------
# Save
# ---------------------------------------

df.to_csv(
    OUTPUT_PATH,
    index=False
)


print("Feature engineering completed.")

print("\nNew columns:")
print([
    "engagement_rate",
    "like_rate",
    "comment_rate",
    "log_views"
])

print("\nDataset shape:")
print(df.shape)

print("\nSaved to:")
print(OUTPUT_PATH)