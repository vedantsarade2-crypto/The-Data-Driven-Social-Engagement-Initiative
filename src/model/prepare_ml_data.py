import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ==================================================
# Configuration
# ==================================================

INPUT_PATH = "data/processed/ml_features.csv"

TRAIN_PATH = "data/processed/train_data.csv"
TEST_PATH = "data/processed/test_data.csv"

os.makedirs("data/processed", exist_ok=True)

# ==================================================
# Load Dataset
# ==================================================

df = pd.read_csv(INPUT_PATH)

print("Dataset shape:", df.shape)

# ==================================================
# Separate Features and Target
# ==================================================

TARGET = "viral"

# Remove identifiers and target
X = df.drop(
    columns=[
        "video_id",
        "title",
        TARGET
    ]
)

y = df[TARGET]

print("\n========== FEATURES ==========")
print(X.columns.tolist())

print("\n========== TARGET DISTRIBUTION ==========")
print(y.value_counts())

print("\nTarget percentage:")
print(
    y.value_counts(normalize=True).mul(100).round(2)
)

# ==================================================
# Identify Feature Types
# ==================================================

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

print("\nCategorical features:")
print(categorical_features)

print("\nNumeric features:")
print(numeric_features)

# ==================================================
# Train-Test Split
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========== TRAIN / TEST SPLIT ==========")

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())

# ==================================================
# Encoding
# ==================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            SimpleImputer(strategy="median"),
            numeric_features
        ),
        (
            "categorical",
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent")
                    ),
                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False
                        )
                    )
                ]
            ),
            categorical_features
        )
    ]
)

# Fit ONLY on training data
X_train_processed = preprocessor.fit_transform(X_train)

# Transform test data
X_test_processed = preprocessor.transform(X_test)

print("\n========== PROCESSED FEATURES ==========")

print(
    "Training feature shape:",
    X_train_processed.shape
)

print(
    "Testing feature shape:",
    X_test_processed.shape
)

# ==================================================
# Get Feature Names
# ==================================================

feature_names = preprocessor.get_feature_names_out()

print("\nNumber of final features:", len(feature_names))

print("\nFeature names:")

for feature in feature_names:
    print("-", feature)

# ==================================================
# Convert to DataFrames
# ==================================================

X_train_processed = pd.DataFrame(
    X_train_processed,
    columns=feature_names,
    index=X_train.index
)

X_test_processed = pd.DataFrame(
    X_test_processed,
    columns=feature_names,
    index=X_test.index
)

# Add target
X_train_processed[TARGET] = y_train.values
X_test_processed[TARGET] = y_test.values

# ==================================================
# Save
# ==================================================

X_train_processed.to_csv(
    TRAIN_PATH,
    index=False
)

X_test_processed.to_csv(
    TEST_PATH,
    index=False
)

print("\n========== FILES SAVED ==========")

print("Training data:")
print(TRAIN_PATH)

print("Testing data:")
print(TEST_PATH)

print("\nML data preparation completed.")