import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# ============================================================
# LOAD COMPLETE DATASET
# ============================================================

data_path = "data/processed/ml_features.csv"

df = pd.read_csv(data_path)

print("Dataset shape:", df.shape)

print("\nTarget distribution:")
print(df["viral"].value_counts())


# ============================================================
# SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(
    columns=[
        "viral",
        "video_id",
        "title"
    ]
)

y = df["viral"]


# ============================================================
# IDENTIFY CATEGORICAL FEATURES
# ============================================================

categorical_features = [
    "topic",
    "publish_day"
]

numeric_features = [
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


# ============================================================
# PREPARE DATA
# ============================================================

X = pd.get_dummies(
    X,
    columns=categorical_features,
    dtype=int
)


# ============================================================
# TRAIN FINAL LOGISTIC REGRESSION
# ============================================================

model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "logistic",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        )
    )
])


model.fit(X, y)


print("\nFinal Logistic Regression model trained successfully.")


# ============================================================
# VIRAL PROBABILITY
# ============================================================

viral_probability = model.predict_proba(X)[:, 1]

predicted_class = (
    viral_probability >= 0.50
).astype(int)


# ============================================================
# CREATE RESULTS DATASET
# ============================================================

results = df[
    [
        "video_id",
        "topic",
        "title",
        "viral"
    ]
].copy()

results["viral_probability"] = viral_probability

results["predicted_viral"] = predicted_class


# ============================================================
# VIRAL POTENTIAL CATEGORY
# ============================================================

def classify_probability(probability):

    if probability >= 0.70:
        return "High"

    elif probability >= 0.40:
        return "Medium"

    else:
        return "Low"


results["viral_potential"] = results[
    "viral_probability"
].apply(classify_probability)


# ============================================================
# SORT BY VIRAL PROBABILITY
# ============================================================

results = results.sort_values(
    by="viral_probability",
    ascending=False
)


# ============================================================
# DISPLAY TOP CONTENT
# ============================================================

print("\n========== TOP VIRAL POTENTIAL CONTENT ==========")

print(
    results[
        [
            "video_id",
            "topic",
            "title",
            "viral_probability",
            "viral_potential"
        ]
    ].head(20).to_string(index=False)
)


# ============================================================
# CATEGORY DISTRIBUTION
# ============================================================

print("\n========== VIRAL POTENTIAL DISTRIBUTION ==========")

print(
    results["viral_potential"].value_counts()
)


# ============================================================
# SAVE RESULTS
# ============================================================

output_path = "data/processed/viral_predictions.csv"

results.to_csv(
    output_path,
    index=False
)

print("\nPredictions saved to:")
print(output_path)


# ============================================================
# SAVE MODEL COEFFICIENTS
# ============================================================

coefficients = pd.DataFrame({
    "feature": X.columns,
    "coefficient": model.named_steps[
        "logistic"
    ].coef_[0]
})

coefficients["absolute_coefficient"] = (
    coefficients["coefficient"].abs()
)

coefficients = coefficients.sort_values(
    by="absolute_coefficient",
    ascending=False
)

coefficients.to_csv(
    "reports/logistic_feature_coefficients.csv",
    index=False
)

print("\nFeature coefficients saved to:")
print("reports/logistic_feature_coefficients.csv")

print("\nFinal model pipeline completed.")