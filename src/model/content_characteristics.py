import pandas as pd
import os


# ============================================================
# LOAD DATA
# ============================================================

ML_PATH = "data/processed/ml_features.csv"
NLP_PATH = "data/processed/video_nlp_dataset.csv"

ml_df = pd.read_csv(ML_PATH)
nlp_df = pd.read_csv(NLP_PATH)

print("ML dataset shape:", ml_df.shape)
print("NLP dataset shape:", nlp_df.shape)


# ============================================================
# DEFINE VIRAL THRESHOLD
# ============================================================

viral_threshold = nlp_df["views"].quantile(0.90)

print("\nViral threshold:")
print(f"{viral_threshold:,.0f}")


# ============================================================
# CREATE VIRAL LABEL
# ============================================================

nlp_df["viral"] = (
    nlp_df["views"] >= viral_threshold
).astype(int)


# ============================================================
# REMOVE DUPLICATE VIRAL COLUMN FROM ML DATASET
# ============================================================

if "viral" in ml_df.columns:
    ml_df = ml_df.drop(columns=["viral"])


# ============================================================
# MERGE ML FEATURES + NLP/PERFORMANCE DATA
# ============================================================

analysis_df = ml_df.merge(
    nlp_df,
    on="video_id",
    how="inner"
)

print("\nMerged dataset shape:", analysis_df.shape)

print("\nColumns:")
print(analysis_df.columns.tolist())


# ============================================================
# VIRAL CONTENT
# ============================================================

viral_df = analysis_df[
    analysis_df["viral"] == 1
].copy()

nonviral_df = analysis_df[
    analysis_df["viral"] == 0
].copy()

print("\nViral content count:", len(viral_df))
print("Non-viral content count:", len(nonviral_df))


# ============================================================
# POSTING HOUR ANALYSIS
# ============================================================

hour_analysis = (
    analysis_df
    .groupby("publish_hour")
    .agg(
        videos=("video_id", "count"),
        median_views=("views", "median"),
        average_views=("views", "mean"),
        viral_count=("viral", "sum")
    )
    .reset_index()
)

hour_analysis["viral_rate"] = (
    hour_analysis["viral_count"]
    / hour_analysis["videos"]
)

hour_analysis = hour_analysis.sort_values(
    by="median_views",
    ascending=False
)


print("\n========================================")
print("POSTING HOUR ANALYSIS")
print("========================================")

print(
    hour_analysis.to_string(index=False)
)


# ============================================================
# POSTING DAY ANALYSIS
# ============================================================

day_analysis = (
    analysis_df
    .groupby("publish_day")
    .agg(
        videos=("video_id", "count"),
        median_views=("views", "median"),
        average_views=("views", "mean"),
        viral_count=("viral", "sum")
    )
    .reset_index()
)

day_analysis["viral_rate"] = (
    day_analysis["viral_count"]
    / day_analysis["videos"]
)

day_analysis = day_analysis.sort_values(
    by="median_views",
    ascending=False
)


print("\n========================================")
print("POSTING DAY ANALYSIS")
print("========================================")

print(
    day_analysis.to_string(index=False)
)


# ============================================================
# TITLE CHARACTERISTICS
# ============================================================

title_metrics = {
    "viral_title_length_mean":
        viral_df["title_length"].mean(),

    "nonviral_title_length_mean":
        nonviral_df["title_length"].mean(),

    "viral_title_words_mean":
        viral_df["title_word_count"].mean(),

    "nonviral_title_words_mean":
        nonviral_df["title_word_count"].mean(),

    "viral_question_rate":
        viral_df["has_question"].mean(),

    "nonviral_question_rate":
        nonviral_df["has_question"].mean(),

    "viral_exclamation_rate":
        viral_df["has_exclamation"].mean(),

    "nonviral_exclamation_rate":
        nonviral_df["has_exclamation"].mean(),

    "viral_number_rate":
        viral_df["has_number"].mean(),

    "nonviral_number_rate":
        nonviral_df["has_number"].mean()
}


# ============================================================
# PRINT TITLE ANALYSIS
# ============================================================

print("\n========================================")
print("TITLE CHARACTERISTICS")
print("========================================")

print(
    f"Viral average title length: "
    f"{title_metrics['viral_title_length_mean']:.2f}"
)

print(
    f"Non-viral average title length: "
    f"{title_metrics['nonviral_title_length_mean']:.2f}"
)

print(
    f"Viral average title words: "
    f"{title_metrics['viral_title_words_mean']:.2f}"
)

print(
    f"Non-viral average title words: "
    f"{title_metrics['nonviral_title_words_mean']:.2f}"
)

print(
    f"Viral question-title rate: "
    f"{title_metrics['viral_question_rate']:.2%}"
)

print(
    f"Non-viral question-title rate: "
    f"{title_metrics['nonviral_question_rate']:.2%}"
)

print(
    f"Viral exclamation-title rate: "
    f"{title_metrics['viral_exclamation_rate']:.2%}"
)

print(
    f"Non-viral exclamation-title rate: "
    f"{title_metrics['nonviral_exclamation_rate']:.2%}"
)

print(
    f"Viral number-title rate: "
    f"{title_metrics['viral_number_rate']:.2%}"
)

print(
    f"Non-viral number-title rate: "
    f"{title_metrics['nonviral_number_rate']:.2%}"
)


# ============================================================
# RECOMMENDED POSTING HOUR
# ============================================================

recommended_hour = int(
    hour_analysis.iloc[0]["publish_hour"]
)


# ============================================================
# RECOMMENDED POSTING DAY
# ============================================================

recommended_day = day_analysis.iloc[0]["publish_day"]


# ============================================================
# RECOMMENDED TITLE CHARACTERISTICS
# ============================================================

recommended_title_length = round(
    viral_df["title_length"].median()
)

recommended_title_words = round(
    viral_df["title_word_count"].median()
)


# ============================================================
# RECOMMENDATION SUMMARY
# ============================================================

print("\n========================================")
print("CONTENT CHARACTERISTIC RECOMMENDATIONS")
print("========================================")

print(
    f"Recommended posting hour: "
    f"{recommended_hour}:00"
)

print(
    f"Recommended posting day: "
    f"{recommended_day}"
)

print(
    f"Recommended title length: "
    f"around {recommended_title_length} characters"
)

print(
    f"Recommended title word count: "
    f"around {recommended_title_words} words"
)


# ============================================================
# SAVE REPORTS
# ============================================================

os.makedirs("reports", exist_ok=True)

hour_analysis.to_csv(
    "reports/posting_hour_analysis.csv",
    index=False
)

day_analysis.to_csv(
    "reports/posting_day_analysis.csv",
    index=False
)

pd.DataFrame(
    [title_metrics]
).to_csv(
    "reports/title_characteristics_analysis.csv",
    index=False
)


print("\nAnalysis files saved:")
print("reports/posting_hour_analysis.csv")
print("reports/posting_day_analysis.csv")
print("reports/title_characteristics_analysis.csv")

print("\nContent characteristic analysis completed.")