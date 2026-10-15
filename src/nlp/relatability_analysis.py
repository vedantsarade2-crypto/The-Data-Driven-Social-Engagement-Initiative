import pandas as pd


# ==========================================
# 1. Paths
# ==========================================

INPUT_PATH = "data/processed/sentiment_comments.csv"

OUTPUT_PATH = "data/processed/nlp_comments.csv"


# ==========================================
# 2. Load dataset
# ==========================================

df = pd.read_csv(INPUT_PATH)


# ==========================================
# 3. Relatability keywords
# ==========================================

relatable_phrases = [
    "same",
    "so relatable",
    "relatable",
    "this is me",
    "literally me",
    "this is literally me",
    "i feel this",
    "i feel the same",
    "i can relate",
    "can relate",
    "this describes me",
    "describes my life",
    "this is my life",
    "going through this",
    "i am going through",
    "happens to me",
    "happened to me",
    "i thought i was the only one",
    "thought i was the only one",
    "me too",
    "this hit me",
    "this hits",
    "hits different",
    "exactly me"
]


# ==========================================
# 4. Problem awareness keywords
# ==========================================

problem_keywords = [
    "anxiety",
    "stress",
    "depressed",
    "depression",
    "lonely",
    "loneliness",
    "overthinking",
    "procrastination",
    "pressure",
    "fear",
    "scared",
    "insecure",
    "insecurity",
    "burnout",
    "tired",
    "exhausted",
    "sad",
    "alone",
    "confused",
    "struggle",
    "struggling",
    "problem",
    "pain",
    "worry",
    "worried",
    "academic",
    "career",
    "friendship",
    "relationship"
]


# ==========================================
# 5. Relatability function
# ==========================================

def classify_relatability(text):

    text = str(text).lower()

    for phrase in relatable_phrases:

        if phrase in text:

            return "Relatable"

    return "Neutral"


# ==========================================
# 6. Problem awareness function
# ==========================================

def detect_problem_awareness(text):

    text = str(text).lower()

    for keyword in problem_keywords:

        if keyword in text:

            return "Aware"

    return "Not_Aware"


# ==========================================
# 7. Apply classification
# ==========================================

df["relatability"] = (
    df["clean_text"]
    .apply(classify_relatability)
)


df["problem_awareness"] = (
    df["clean_text"]
    .apply(detect_problem_awareness)
)


# ==========================================
# 8. Create scores
# ==========================================

df["relatability_score"] = (
    df["relatability"]
    .map({
        "Relatable": 1,
        "Neutral": 0
    })
)


df["problem_awareness_score"] = (
    df["problem_awareness"]
    .map({
        "Aware": 1,
        "Not_Aware": 0
    })
)


# ==========================================
# 9. Print results
# ==========================================

print("\n========== RELATABILITY ==========")

print(
    df["relatability"]
    .value_counts()
)


print("\n========== PROBLEM AWARENESS ==========")

print(
    df["problem_awareness"]
    .value_counts()
)


# ==========================================
# 10. Save
# ==========================================

df.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\nNLP classification completed.")

print("Dataset shape:")
print(df.shape)

print("\nSaved to:")
print(OUTPUT_PATH)