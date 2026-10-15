import pandas as pd
import re
import os


# ==========================================
# 1. File paths
# ==========================================

INPUT_PATH = "data/raw/comments/youtube_comments.csv"

OUTPUT_DIR = "data/processed"
OUTPUT_PATH = "data/processed/clean_comments.csv"


# ==========================================
# 2. Create output directory
# ==========================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ==========================================
# 3. Load comments
# ==========================================

df = pd.read_csv(INPUT_PATH)

print("Original dataset shape:")
print(df.shape)


# ==========================================
# 4. Remove missing comments
# ==========================================

df = df.dropna(
    subset=["comment_text"]
)


# ==========================================
# 5. Convert comments to string
# ==========================================

df["comment_text"] = (
    df["comment_text"]
    .astype(str)
)


# ==========================================
# 6. Remove duplicate comments
# ==========================================

df = df.drop_duplicates(
    subset=["video_id", "comment_text"]
)


# ==========================================
# 7. Text cleaning function
# ==========================================

def clean_text(text):

    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    # Remove mentions
    text = re.sub(
        r"@\w+",
        "",
        text
    )

    # Remove hashtags symbol but keep word
    text = re.sub(
        r"#",
        "",
        text
    )

    # Remove HTML tags
    text = re.sub(
        r"<.*?>",
        "",
        text
    )

    # Keep letters and numbers
    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ==========================================
# 8. Apply cleaning
# ==========================================

df["clean_text"] = (
    df["comment_text"]
    .apply(clean_text)
)


# ==========================================
# 9. Remove empty cleaned comments
# ==========================================

df = df[
    df["clean_text"].str.len() > 0
]


# ==========================================
# 10. Comment length
# ==========================================

df["word_count"] = (
    df["clean_text"]
    .str.split()
    .str.len()
)


# ==========================================
# 11. Save cleaned dataset
# ==========================================

df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ==========================================
# 12. Summary
# ==========================================

print("\n========== CLEANING COMPLETE ==========")

print(
    "Original comments:",
    15682
)

print(
    "Clean comments:",
    len(df)
)

print(
    "Removed comments:",
    15682 - len(df)
)

print("\nColumns:")
print(df.columns.tolist())

print("\nSample comments:")

print(
    df[
        [
            "video_id",
            "comment_text",
            "clean_text",
            "word_count"
        ]
    ].head(10)
)

print("\nSaved to:")
print(OUTPUT_PATH)