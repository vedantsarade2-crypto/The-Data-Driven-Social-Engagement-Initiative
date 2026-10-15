from pathlib import Path
import pandas as pd


# ============================================================
# FINAL PROJECT DATA VALIDATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORT_DIR = PROJECT_ROOT / "reports" / "validation"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Expected files
# ------------------------------------------------------------

RAW_FILES = [
    RAW_DIR / "comments" / "youtube_comments.csv",
    RAW_DIR / "tremds" / "trend_data.csv",
    RAW_DIR / "youtube" / "video_search_results.csv",
    RAW_DIR / "youtube" / "video_statistics.csv",
    RAW_DIR / "youtube" / "youtube_combined.csv",
    RAW_DIR / "content_series.csv",
]

PROCESSED_FILES = [
    PROCESSED_DIR / "clean_comments.csv",
    PROCESSED_DIR / "sentiment_comments.csv",
    PROCESSED_DIR / "nlp_comments.csv",
    PROCESSED_DIR / "video_nlp_dataset.csv",
    PROCESSED_DIR / "ml_features.csv",
    PROCESSED_DIR / "train_data.csv",
    PROCESSED_DIR / "test_data.csv",
    PROCESSED_DIR / "viral_predictions.csv",
    PROCESSED_DIR / "topic_performance.csv",
    PROCESSED_DIR / "final_recommendations.csv",
    PROCESSED_DIR / "trend_analysis.csv",
    PROCESSED_DIR / "rising_topics.csv",
    PROCESSED_DIR / "trend_aware_recommendations.csv",
]


# ------------------------------------------------------------
# Important expected columns
# ------------------------------------------------------------

EXPECTED_COLUMNS = {
    "youtube_comments.csv": [
        "video_id",
        "comment_text",
    ],

    "trend_data.csv": [
        "trend_id",
        "trend_date",
        "keyword",
        "category",
        "search_volume",
        "hashtag_volume",
    ],

    "youtube_combined.csv": [
        "video_id",
        "topic",
        "title",
        "published_at",
        "channel_title",
        "views",
        "likes",
        "comments_count",
    ],

    "content_series.csv": [
        "content_id",
        "topic",
        "content_format",
        "hook_type",
        "video_length_sec",
        "posting_period",
        "caption_style",
    ],

    "clean_comments.csv": [
        "video_id",
        "comment_text",
        "clean_text",
    ],

    "sentiment_comments.csv": [
        "video_id",
        "comment_text",
        "sentiment",
        "sentiment_score",
    ],

    "nlp_comments.csv": [
        "video_id",
        "comment_text",
        "sentiment",
        "sentiment_score",
        "relatability",
        "relatability_score",
        "problem_awareness",
    ],

    "video_nlp_dataset.csv": [
        "video_id",
        "topic",
        "views",
        "likes",
        "comments_count",
        "engagement_rate",
        "average_sentiment",
        "relatability_rate",
        "problem_awareness_rate",
    ],

    "ml_features.csv": [
        "video_id",
        "topic",
        "viral",
    ],

    "viral_predictions.csv": [
        "video_id",
        "viral_probability",
    ],

    "topic_performance.csv": [
        "topic",
        "videos",
        "median_views",
        "viral_count",
        "viral_rate",
    ],

    "final_recommendations.csv": [
        "recommendation_id",
        "recommended_topic",
    ],

    "trend_analysis.csv": [
        "keyword",
        "search_volume",
        "hashtag_volume",
        "trend_score",
    ],

    "rising_topics.csv": [
        "keyword",
        "rising_score",
    ],

    "trend_aware_recommendations.csv": [
        "topic",
    ],
}


# ------------------------------------------------------------
# Validation helper
# ------------------------------------------------------------

validation_results = []


def validate_file(file_path):
    result = {
        "file": str(file_path.relative_to(PROJECT_ROOT)),
        "exists": False,
        "status": "FAIL",
        "rows": 0,
        "columns": 0,
        "missing_cells": 0,
        "duplicate_rows": 0,
        "missing_expected_columns": "",
        "error": "",
    }

    if not file_path.exists():
        result["error"] = "File does not exist"
        return result

    result["exists"] = True

    try:
        df = pd.read_csv(file_path)

        result["rows"] = len(df)
        result["columns"] = len(df.columns)
        result["missing_cells"] = int(df.isna().sum().sum())
        result["duplicate_rows"] = int(df.duplicated().sum())

        expected = EXPECTED_COLUMNS.get(file_path.name, [])

        missing_columns = [
            col for col in expected
            if col not in df.columns
        ]

        result["missing_expected_columns"] = ", ".join(
            missing_columns
        )

        if len(df) == 0:
            result["error"] = "Dataset is empty"
            return result

        if missing_columns:
            result["error"] = (
                "Missing expected columns: "
                + ", ".join(missing_columns)
            )
            return result

        result["status"] = "PASS"

    except Exception as e:
        result["error"] = str(e)

    return result


# ------------------------------------------------------------
# Run validation
# ------------------------------------------------------------

print("=" * 70)
print("FINAL PROJECT DATA VALIDATION")
print("=" * 70)

print("\nProject root:")
print(PROJECT_ROOT)

print("\nChecking RAW datasets...")
print("-" * 70)

for file_path in RAW_FILES:
    result = validate_file(file_path)
    validation_results.append(result)

    status = "PASS" if result["status"] == "PASS" else "FAIL"

    print(
        f"[{status}] "
        f"{result['file']} | "
        f"Rows: {result['rows']} | "
        f"Columns: {result['columns']}"
    )

    if result["error"]:
        print(f"      Error: {result['error']}")


print("\nChecking PROCESSED datasets...")
print("-" * 70)

for file_path in PROCESSED_FILES:
    result = validate_file(file_path)
    validation_results.append(result)

    status = "PASS" if result["status"] == "PASS" else "FAIL"

    print(
        f"[{status}] "
        f"{result['file']} | "
        f"Rows: {result['rows']} | "
        f"Columns: {result['columns']}"
    )

    if result["error"]:
        print(f"      Error: {result['error']}")


# ------------------------------------------------------------
# Save detailed validation report
# ------------------------------------------------------------

report_df = pd.DataFrame(validation_results)

report_path = REPORT_DIR / "final_data_validation.csv"

report_df.to_csv(
    report_path,
    index=False
)


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

total = len(report_df)
passed = int((report_df["status"] == "PASS").sum())
failed = total - passed

total_missing = int(report_df["missing_cells"].sum())
total_duplicates = int(report_df["duplicate_rows"].sum())

print("\n" + "=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)

print(f"Total files checked : {total}")
print(f"Passed              : {passed}")
print(f"Failed              : {failed}")
print(f"Missing cells       : {total_missing}")
print(f"Duplicate rows      : {total_duplicates}")

print(f"\nValidation report:")
print(report_path)


# ------------------------------------------------------------
# Failed files
# ------------------------------------------------------------

failed_df = report_df[
    report_df["status"] != "PASS"
]

if not failed_df.empty:

    print("\n" + "=" * 70)
    print("FILES REQUIRING ATTENTION")
    print("=" * 70)

    for _, row in failed_df.iterrows():

        print(f"\nFile: {row['file']}")

        if row["error"]:
            print(f"Reason: {row['error']}")

else:

    print("\n" + "=" * 70)
    print("ALL DATASETS PASSED VALIDATION")
    print("=" * 70)

print("\nValidation completed.")