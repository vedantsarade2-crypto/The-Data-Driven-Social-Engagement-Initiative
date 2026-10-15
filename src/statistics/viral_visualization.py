import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

INPUT_PATH = "data/processed/video_viral_dataset.csv"
OUTPUT_DIR = "reports/eda/viral"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_PATH)

# --------------------------------------------------
# 1. Viral vs Non-Viral Views
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="viral_label",
    y="views"
)

plt.yscale("log")

plt.title("Views: Viral vs Non-Viral Content")
plt.xlabel("Content Type")
plt.ylabel("Views (Log Scale)")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/viral_vs_nonviral_views.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 2. Viral Topic Distribution
# --------------------------------------------------

viral_df = df[df["viral"] == 1]

topic_counts = (
    viral_df["topic"]
    .value_counts()
    .sort_values(ascending=True)
)

plt.figure(figsize=(9, 6))

topic_counts.plot(kind="barh")

plt.title("Viral Videos by Topic")
plt.xlabel("Number of Viral Videos")
plt.ylabel("Topic")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/viral_topics.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 3. Relatability: Viral vs Non-Viral
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="viral_label",
    y="relatability_rate"
)

plt.title("Relatability Rate: Viral vs Non-Viral")
plt.xlabel("Content Type")
plt.ylabel("Relatability Rate (%)")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/viral_vs_nonviral_relatability.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 4. Problem Awareness: Viral vs Non-Viral
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="viral_label",
    y="problem_awareness_rate"
)

plt.title("Problem Awareness Rate: Viral vs Non-Viral")
plt.xlabel("Content Type")
plt.ylabel("Problem Awareness Rate (%)")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/viral_vs_nonviral_problem_awareness.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 5. Engagement Rate: Viral vs Non-Viral
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="viral_label",
    y="engagement_rate"
)

plt.title("Engagement Rate: Viral vs Non-Viral")
plt.xlabel("Content Type")
plt.ylabel("Engagement Rate (%)")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/viral_vs_nonviral_engagement.png",
    dpi=300
)

plt.close()


print("Viral visualization completed.")

print("\nGenerated files:")

for file in os.listdir(OUTPUT_DIR):
    print("-", file)