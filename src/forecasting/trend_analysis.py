import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------

INPUT_FILE = "data/raw/tremds/trend_data.csv"

OUTPUT_DIR = "reports/forecasting"
PROCESSED_DIR = "data/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


# ---------------------------------------------------------
# 2. Load data
# ---------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("Trend dataset loaded successfully.")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())


# ---------------------------------------------------------
# 3. Data validation
# ---------------------------------------------------------

required_columns = [
    "trend_id",
    "trend_date",
    "keyword",
    "category",
    "search_volume",
    "hashtag_volume"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# Convert date
df["trend_date"] = pd.to_datetime(df["trend_date"])


# Convert numeric columns
df["search_volume"] = pd.to_numeric(
    df["search_volume"],
    errors="coerce"
)

df["hashtag_volume"] = pd.to_numeric(
    df["hashtag_volume"],
    errors="coerce"
)


# Remove invalid rows
df = df.dropna(
    subset=[
        "trend_date",
        "keyword",
        "search_volume",
        "hashtag_volume"
    ]
)


# Sort
df = df.sort_values(
    ["keyword", "trend_date"]
).reset_index(drop=True)


# ---------------------------------------------------------
# 4. Calculate Trend Score
# ---------------------------------------------------------

# Normalize search and hashtag volume
search_max = df["search_volume"].max()
hashtag_max = df["hashtag_volume"].max()

if search_max > 0:
    df["search_score"] = (
        df["search_volume"] / search_max
    )
else:
    df["search_score"] = 0


if hashtag_max > 0:
    df["hashtag_score"] = (
        df["hashtag_volume"] / hashtag_max
    )
else:
    df["hashtag_score"] = 0


# Weighted trend score
df["trend_score"] = (
    0.5 * df["search_score"] +
    0.5 * df["hashtag_score"]
)


# ---------------------------------------------------------
# 5. Calculate growth rates
# ---------------------------------------------------------

df["search_growth_pct"] = (
    df.groupby("keyword")["search_volume"]
    .pct_change() * 100
)

df["hashtag_growth_pct"] = (
    df.groupby("keyword")["hashtag_volume"]
    .pct_change() * 100
)


# Overall trend growth
df["trend_score_growth_pct"] = (
    df.groupby("keyword")["trend_score"]
    .pct_change() * 100
)


# ---------------------------------------------------------
# 6. Rising Topic Analysis
# ---------------------------------------------------------

latest_date = df["trend_date"].max()

latest = df[
    df["trend_date"] == latest_date
].copy()


# Calculate average growth
growth_summary = (
    df.groupby("keyword")
    .agg(
        average_search_growth=(
            "search_growth_pct",
            "mean"
        ),
        average_hashtag_growth=(
            "hashtag_growth_pct",
            "mean"
        ),
        latest_search_volume=(
            "search_volume",
            "last"
        ),
        latest_hashtag_volume=(
            "hashtag_volume",
            "last"
        ),
        latest_trend_score=(
            "trend_score",
            "last"
        )
    )
    .reset_index()
)


# Rising score
growth_summary["rising_score"] = (
    0.4 * growth_summary["average_search_growth"].fillna(0)
    + 0.4 * growth_summary["average_hashtag_growth"].fillna(0)
    + 0.2 * growth_summary["latest_trend_score"]
)


growth_summary = growth_summary.sort_values(
    "rising_score",
    ascending=False
).reset_index(drop=True)


# Rank
growth_summary["rank"] = (
    growth_summary.index + 1
)


# ---------------------------------------------------------
# 7. Save processed trend data
# ---------------------------------------------------------

trend_output = os.path.join(
    PROCESSED_DIR,
    "trend_analysis.csv"
)

df.to_csv(
    trend_output,
    index=False
)


rising_output = os.path.join(
    PROCESSED_DIR,
    "rising_topics.csv"
)

growth_summary.to_csv(
    rising_output,
    index=False
)


# ---------------------------------------------------------
# 8. Print results
# ---------------------------------------------------------

print("\nTrend Analysis:")
print(
    df[
        [
            "trend_date",
            "keyword",
            "search_volume",
            "hashtag_volume",
            "trend_score",
            "search_growth_pct",
            "hashtag_growth_pct"
        ]
    ].to_string(index=False)
)


print("\nRising Topics:")
print(
    growth_summary.to_string(index=False)
)


# ---------------------------------------------------------
# 9. Visualization — Search Volume
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

for keyword, group in df.groupby("keyword"):

    plt.plot(
        group["trend_date"],
        group["search_volume"],
        marker="o",
        label=keyword
    )

plt.title("Search Volume Trend")
plt.xlabel("Date")
plt.ylabel("Search Volume")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "search_volume_trend.png"
    ),
    dpi=300
)

plt.close()


# ---------------------------------------------------------
# 10. Visualization — Hashtag Volume
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

for keyword, group in df.groupby("keyword"):

    plt.plot(
        group["trend_date"],
        group["hashtag_volume"],
        marker="o",
        label=keyword
    )

plt.title("Hashtag Volume Trend")
plt.xlabel("Date")
plt.ylabel("Hashtag Volume")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "hashtag_volume_trend.png"
    ),
    dpi=300
)

plt.close()


# ---------------------------------------------------------
# 11. Rising Topic Chart
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    growth_summary["keyword"],
    growth_summary["rising_score"]
)

plt.title("Rising Topic Score")
plt.xlabel("Keyword")
plt.ylabel("Rising Score")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "rising_topic_score.png"
    ),
    dpi=300
)

plt.close()


print("\nTrend analysis completed successfully.")

print("\nGenerated files:")
print(trend_output)
print(rising_output)

print(
    os.path.join(
        OUTPUT_DIR,
        "search_volume_trend.png"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "hashtag_volume_trend.png"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "rising_topic_score.png"
    )
)