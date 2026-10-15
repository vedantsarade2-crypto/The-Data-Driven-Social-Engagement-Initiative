import pandas as pd
import os

INPUT_PATH = "data/raw/content_series.csv"
OUTPUT_DIR = "reports/statistics"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_PATH)

print("========== A/B TESTING DATASET ==========")
print("Shape:", df.shape)

# --------------------------------------------------
# Experiment 1: Short vs Long
# --------------------------------------------------

print("\n========== EXPERIMENT 1: SHORT VS LONG ==========")

short_group = df[df["content_format"] == "Short"]
long_group = df[df["content_format"] == "Long"]

print("Short videos:", len(short_group))
print("Long videos:", len(long_group))

print("\nShort content:")
print(
    short_group[
        ["content_id", "topic", "video_length_sec"]
    ].to_string(index=False)
)

print("\nLong content:")
print(
    long_group[
        ["content_id", "topic", "video_length_sec"]
    ].to_string(index=False)
)


# --------------------------------------------------
# Experiment 2: Visual vs Text Hook
# --------------------------------------------------

print("\n========== EXPERIMENT 2: VISUAL VS TEXT HOOK ==========")

visual_group = df[df["hook_type"] == "Visual"]
text_group = df[df["hook_type"] == "Text"]

print("Visual hook:", len(visual_group))
print("Text hook:", len(text_group))

print("\nVisual hook content:")
print(
    visual_group[
        ["content_id", "topic", "hook_type"]
    ].to_string(index=False)
)

print("\nText hook content:")
print(
    text_group[
        ["content_id", "topic", "hook_type"]
    ].to_string(index=False)
)


# --------------------------------------------------
# Experiment 3: Evening vs Night
# --------------------------------------------------

print("\n========== EXPERIMENT 3: EVENING VS NIGHT ==========")

evening_group = df[df["posting_period"] == "Evening"]
night_group = df[df["posting_period"] == "Night"]

print("Evening:", len(evening_group))
print("Night:", len(night_group))

print("\nEvening content:")
print(
    evening_group[
        ["content_id", "topic", "posting_period"]
    ].to_string(index=False)
)

print("\nNight content:")
print(
    night_group[
        ["content_id", "topic", "posting_period"]
    ].to_string(index=False)
)


# --------------------------------------------------
# Check for performance metrics
# --------------------------------------------------

possible_metrics = [
    "views",
    "likes",
    "comments_count",
    "shares",
    "saves",
    "engagement_rate",
    "retention_rate"
]

available_metrics = [
    column for column in possible_metrics
    if column in df.columns
]

print("\n========== PERFORMANCE METRICS ==========")

if available_metrics:
    print("Available metrics:")
    for metric in available_metrics:
        print("-", metric)
else:
    print("No performance metrics are available.")

    print("\nA/B testing cannot calculate statistical")
    print("significance until performance data is available.")


# --------------------------------------------------
# Save experiment design
# --------------------------------------------------

experiment_design = pd.DataFrame({
    "experiment": [
        "Video Length",
        "Hook Type",
        "Posting Period"
    ],
    "group_a": [
        "Short",
        "Visual",
        "Evening"
    ],
    "group_b": [
        "Long",
        "Text",
        "Night"
    ],
    "group_a_count": [
        len(short_group),
        len(visual_group),
        len(evening_group)
    ],
    "group_b_count": [
        len(long_group),
        len(text_group),
        len(night_group)
    ]
})

experiment_design.to_csv(
    f"{OUTPUT_DIR}/ab_experiment_design.csv",
    index=False
)

print("\nExperiment design saved to:")
print(f"{OUTPUT_DIR}/ab_experiment_design.csv")

print("\nA/B experiment design completed.")