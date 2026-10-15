import pandas as pd
import numpy as np
import os
from datetime import date


# ============================================================
# CONFIGURATION
# ============================================================

TOPIC_PATH = "data/processed/topic_performance.csv"
PREDICTION_PATH = "data/processed/viral_predictions.csv"

HOUR_PATH = "reports/posting_hour_analysis.csv"
DAY_PATH = "reports/posting_day_analysis.csv"
TITLE_PATH = "reports/title_characteristics_analysis.csv"

OUTPUT_PATH = "data/processed/final_recommendations.csv"

MIN_HOUR_VIDEOS = 5
MIN_DAY_VIDEOS = 5


# ============================================================
# LOAD DATA
# ============================================================

topic_df = pd.read_csv(TOPIC_PATH)
prediction_df = pd.read_csv(PREDICTION_PATH)
hour_df = pd.read_csv(HOUR_PATH)
day_df = pd.read_csv(DAY_PATH)
title_df = pd.read_csv(TITLE_PATH)

print("Topic performance shape:", topic_df.shape)
print("Prediction dataset shape:", prediction_df.shape)
print("Posting hour analysis shape:", hour_df.shape)
print("Posting day analysis shape:", day_df.shape)
print("Title analysis shape:", title_df.shape)


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_topic_columns = [
    "topic",
    "recommendation_score"
]

required_prediction_columns = [
    "video_id"
]

required_hour_columns = [
    "publish_hour",
    "videos",
    "median_views",
    "viral_rate"
]

required_day_columns = [
    "publish_day",
    "videos",
    "median_views",
    "viral_rate"
]

for column in required_topic_columns:
    if column not in topic_df.columns:
        raise ValueError(
            f"Missing column in topic performance: {column}"
        )

for column in required_prediction_columns:
    if column not in prediction_df.columns:
        raise ValueError(
            f"Missing column in predictions: {column}"
        )

for column in required_hour_columns:
    if column not in hour_df.columns:
        raise ValueError(
            f"Missing column in hour analysis: {column}"
        )

for column in required_day_columns:
    if column not in day_df.columns:
        raise ValueError(
            f"Missing column in day analysis: {column}"
        )


# ============================================================
# FIND BEST RELIABLE POSTING HOUR
# ============================================================

reliable_hours = hour_df[
    hour_df["videos"] >= MIN_HOUR_VIDEOS
].copy()

if reliable_hours.empty:
    raise ValueError(
        "No posting hour has enough observations."
    )

reliable_hours = reliable_hours.sort_values(
    by=["median_views", "viral_rate"],
    ascending=False
)

recommended_hour = int(
    reliable_hours.iloc[0]["publish_hour"]
)

recommended_hour_median_views = (
    reliable_hours.iloc[0]["median_views"]
)

recommended_hour_viral_rate = (
    reliable_hours.iloc[0]["viral_rate"]
)

recommended_hour_videos = (
    reliable_hours.iloc[0]["videos"]
)


# ============================================================
# FIND BEST RELIABLE POSTING DAY
# ============================================================

reliable_days = day_df[
    day_df["videos"] >= MIN_DAY_VIDEOS
].copy()

if reliable_days.empty:
    raise ValueError(
        "No posting day has enough observations."
    )

reliable_days = reliable_days.sort_values(
    by=["median_views", "viral_rate"],
    ascending=False
)

recommended_day = (
    reliable_days.iloc[0]["publish_day"]
)

recommended_day_median_views = (
    reliable_days.iloc[0]["median_views"]
)

recommended_day_viral_rate = (
    reliable_days.iloc[0]["viral_rate"]
)

recommended_day_videos = (
    reliable_days.iloc[0]["videos"]
)


# ============================================================
# TITLE RECOMMENDATIONS
# ============================================================

title_row = title_df.iloc[0]

viral_title_length = (
    title_row["viral_title_length_mean"]
)

viral_title_words = (
    title_row["viral_title_words_mean"]
)

viral_question_rate = (
    title_row["viral_question_rate"]
)

viral_number_rate = (
    title_row["viral_number_rate"]
)


# ============================================================
# TOP TOPICS
# ============================================================

top_topics = topic_df.sort_values(
    by="recommendation_score",
    ascending=False
).head(5)


# ============================================================
# HELPER: GET MODEL SCORE COLUMN
# ============================================================

possible_probability_columns = [
    "viral_probability",
    "predicted_probability",
    "viral_prob",
    "probability",
    "ml_probability"
]

probability_column = None

for column in possible_probability_columns:
    if column in prediction_df.columns:
        probability_column = column
        break


# ============================================================
# CREATE RECOMMENDATIONS
# ============================================================

recommendations = []


for rank, (_, topic_row) in enumerate(
    top_topics.iterrows(),
    start=1
):

    topic = topic_row["topic"]

    recommendation_score = (
        topic_row["recommendation_score"]
    )

    # --------------------------------------------------------
    # MODEL SCORE
    # --------------------------------------------------------

    if probability_column is not None:

        topic_predictions = prediction_df[
            prediction_df["topic"] == topic
        ]

        if len(topic_predictions) > 0:

            model_score = (
                topic_predictions[
                    probability_column
                ].mean()
            )

        else:

            model_score = recommendation_score

    else:

        model_score = recommendation_score


    # --------------------------------------------------------
    # TITLE STYLE
    # --------------------------------------------------------

    if viral_question_rate >= 0.15:

        hook_style = "Question-based"

    else:

        hook_style = "Direct statement"


    # --------------------------------------------------------
    # RECOMMENDED LENGTH
    # --------------------------------------------------------

    recommended_length = (
        f"{round(viral_title_length)} characters / "
        f"{round(viral_title_words)} words"
    )


    # --------------------------------------------------------
    # REASONING
    # --------------------------------------------------------

    reasoning = (
        f"Topic '{topic}' ranked #{rank} based on its "
        f"historical recommendation score. "
        f"The topic showed an estimated model score of "
        f"{model_score:.3f}. "
        f"Use a concise title of approximately "
        f"{round(viral_title_length)} characters "
        f"({round(viral_title_words)} words). "
        f"The reliable posting analysis favors "
        f"{recommended_day} around {recommended_hour}:00. "
        f"Number-based titles appeared in "
        f"{viral_number_rate:.1%} of viral videos."
    )


    # --------------------------------------------------------
    # ADD RECORD
    # --------------------------------------------------------

    recommendations.append({

        "recommendation_id":
            f"REC{rank:03d}",

        "recommendation_date":
            date.today().isoformat(),

        "recommended_topic":
            topic,

        "recommended_format":
            "Not determined",

        "recommended_length":
            recommended_length,

        "recommended_hook":
            hook_style,

        "recommended_caption_style":
            "Not determined",

        "recommended_posting_period":
            f"{recommended_day} {recommended_hour:02d}:00",

        "expected_viral_probability":
            round(float(model_score), 4),

        "reasoning":
            reasoning
    })


# ============================================================
# CREATE DATAFRAME
# ============================================================

recommendation_df = pd.DataFrame(
    recommendations
)


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

recommendation_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n========================================")
print("FINAL CONTENT RECOMMENDATIONS")
print("========================================")

print(
    recommendation_df.to_string(index=False)
)


print("\n========================================")
print("RECOMMENDATION SUMMARY")
print("========================================")

print(
    f"Reliable recommended posting day: "
    f"{recommended_day}"
)

print(
    f"Reliable recommended posting hour: "
    f"{recommended_hour}:00"
)

print(
    f"Videos considered for hour recommendation: "
    f"{recommended_hour_videos}"
)

print(
    f"Videos considered for day recommendation: "
    f"{recommended_day_videos}"
)

print(
    f"Recommended viral title length: "
    f"~{round(viral_title_length)} characters"
)

print(
    f"Recommended viral title words: "
    f"~{round(viral_title_words)} words"
)


print("\nOutput saved to:")
print(OUTPUT_PATH)

print("\nFinal recommendation engine completed.")