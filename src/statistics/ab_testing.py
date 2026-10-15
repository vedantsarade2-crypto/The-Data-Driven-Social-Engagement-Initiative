# import pandas as pd

# INPUT_PATH = "data/processed/video_viral_dataset.csv"

# df = pd.read_csv(INPUT_PATH)

# print("Dataset shape:", df.shape)

# print("\n========== AVAILABLE COLUMNS ==========")
# print(df.columns.tolist())

# # --------------------------------------------------
# # Video length
# # --------------------------------------------------

# if "video_length_sec" in df.columns:
#     print("\n========== VIDEO LENGTH ==========")
#     print(df["video_length_sec"].describe())

# # --------------------------------------------------
# # Hook type
# # --------------------------------------------------

# if "hook_type" in df.columns:
#     print("\n========== HOOK TYPE ==========")
#     print(df["hook_type"].value_counts(dropna=False))

# # --------------------------------------------------
# # Posting period
# # --------------------------------------------------

# if "posting_period" in df.columns:
#     print("\n========== POSTING PERIOD ==========")
#     print(df["posting_period"].value_counts(dropna=False))

# # --------------------------------------------------
# # Content format
# # --------------------------------------------------

# if "content_format" in df.columns:
#     print("\n========== CONTENT FORMAT ==========")
#     print(df["content_format"].value_counts(dropna=False))

# # --------------------------------------------------
# # Caption style
# # --------------------------------------------------

# if "caption_style" in df.columns:
#     print("\n========== CAPTION STYLE ==========")
#     print(df["caption_style"].value_counts(dropna=False))

# print("\nInspection completed.")


import pandas as pd

INPUT_PATH = "data/raw/content_series.csv"

df = pd.read_csv(INPUT_PATH)

print("Dataset shape:", df.shape)

print("\n========== COLUMNS ==========")
print(df.columns.tolist())

print("\n========== FIRST 10 ROWS ==========")
print(df.head(10).to_string())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# Print unique values for categorical columns
categorical_columns = [
    "topic",
    "content_format",
    "hook_type",
    "posting_period",
    "caption_style"
]

for column in categorical_columns:
    if column in df.columns:
        print(f"\n========== {column.upper()} ==========")
        print(df[column].value_counts(dropna=False))

# Print numeric columns
print("\n========== NUMERIC SUMMARY ==========")
print(df.describe(include="all").transpose())

print("\nInspection completed.")