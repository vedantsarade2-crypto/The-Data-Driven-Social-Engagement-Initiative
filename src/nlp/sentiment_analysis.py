import pandas as pd
import os
from textblob import TextBlob


# ==========================================
# 1. Paths
# ==========================================

INPUT_PATH = "data/processed/clean_comments.csv"

OUTPUT_PATH = "data/processed/sentiment_comments.csv"


# ==========================================
# 2. Load data
# ==========================================

df = pd.read_csv(INPUT_PATH)

print("Comments loaded:", len(df))


# ==========================================
# 3. Sentiment function
# ==========================================

def analyze_sentiment(text):

    analysis = TextBlob(text)

    polarity = analysis.sentiment.polarity

    if polarity > 0.05:

        sentiment = "Positive"

    elif polarity < -0.05:

        sentiment = "Negative"

    else:

        sentiment = "Neutral"

    return sentiment, polarity


# ==========================================
# 4. Apply sentiment analysis
# ==========================================

results = df["clean_text"].apply(
    analyze_sentiment
)


df["sentiment"] = results.apply(
    lambda x: x[0]
)

df["sentiment_score"] = results.apply(
    lambda x: x[1]
)


# ==========================================
# 5. Sentiment distribution
# ==========================================

print("\n========== SENTIMENT DISTRIBUTION ==========")

print(
    df["sentiment"]
    .value_counts()
)


# ==========================================
# 6. Average sentiment
# ==========================================

print("\nAverage sentiment score:")

print(
    df["sentiment_score"].mean()
)


# ==========================================
# 7. Save
# ==========================================

df.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\nSentiment analysis completed.")

print("Saved to:")
print(OUTPUT_PATH)