import pandas as pd
import numpy as np
import os
import re

# ==================================================
# Configuration
# ==================================================

INPUT_PATH = "data/processed/video_viral_dataset.csv"
OUTPUT_PATH = "data/processed/ml_features.csv"

os.makedirs("data/processed", exist_ok=True)

# ==================================================
# Load Dataset
# ==================================================

df = pd.read_csv(INPUT_PATH)

print("Original dataset shape:", df.shape)

# ==================================================
# Basic Date Processing
# ==================================================

df["published_at"] = pd.to_datetime(
    df["published_at"],
    errors="coerce"
)

# Publishing hour
df["publish_hour"] = df["published_at"].dt.hour

# Day of week
df["publish_day"] = df["published_at"].dt.day_name()

# Day number
df["publish_day_num"] = df["published_at"].dt.dayofweek

# Weekend indicator
df["is_weekend"] = (
    df["publish_day_num"] >= 5
).astype(int)

# ==================================================
# Title Features
# ==================================================

df["title"] = df["title"].fillna("").astype(str)

# Character count
df["title_length"] = df["title"].str.len()

# Word count
df["title_word_count"] = (
    df["title"]
    .str.split()
    .str.len()
)

# Number of question marks
df["question_mark_count"] = (
    df["title"].str.count(r"\?")
)

# Number of exclamation marks
df["exclamation_mark_count"] = (
    df["title"].str.count(r"!")
)

# Number of digits
df["digit_count"] = (
    df["title"].str.count(r"\d")
)

# Uppercase character count
df["uppercase_count"] = df["title"].apply(
    lambda x: sum(c.isupper() for c in x)
)

# ==================================================
# Title Style Features
# ==================================================

df["has_question"] = (
    df["question_mark_count"] > 0
).astype(int)

df["has_exclamation"] = (
    df["exclamation_mark_count"] > 0
).astype(int)

df["has_number"] = (
    df["digit_count"] > 0
).astype(int)

# ==================================================
# Topic Cleaning
# ==================================================

df["topic"] = (
    df["topic"]
    .fillna("unknown")
    .astype(str)
    .str.lower()
    .str.strip()
)

# ==================================================
# Create ML Dataset
# ==================================================

feature_columns = [
    "video_id",
    "topic",
    "title",
    "publish_hour",
    "publish_day",
    "publish_day_num",
    "is_weekend",
    "title_length",
    "title_word_count",
    "question_mark_count",
    "exclamation_mark_count",
    "digit_count",
    "uppercase_count",
    "has_question",
    "has_exclamation",
    "has_number",
    "viral"
]

ml_df = df[feature_columns].copy()

# ==================================================
# Missing Value Handling
# ==================================================

numeric_columns = [
    "publish_hour",
    "publish_day_num",
    "is_weekend",
    "title_length",
    "title_word_count",
    "question_mark_count",
    "exclamation_mark_count",
    "digit_count",
    "uppercase_count",
    "has_question",
    "has_exclamation",
    "has_number"
]

for column in numeric_columns:
    ml_df[column] = ml_df[column].fillna(0)

ml_df["topic"] = ml_df["topic"].fillna("unknown")

ml_df["publish_day"] = ml_df["publish_day"].fillna("Unknown")

# ==================================================
# Save
# ==================================================

ml_df.to_csv(
    OUTPUT_PATH,
    index=False
)

# ==================================================
# Summary
# ==================================================

print("\n========== ML FEATURE DATASET ==========")

print("Shape:", ml_df.shape)

print("\nColumns:")
for column in ml_df.columns:
    print("-", column)

print("\nTarget distribution:")
print(
    ml_df["viral"]
    .value_counts()
)

print("\nMissing values:")
print(
    ml_df.isnull().sum()
)

print("\nSample:")
print(
    ml_df.head().to_string(index=False)
)

print("\nSaved to:")
print(OUTPUT_PATH)

print("\nML feature engineering completed.")