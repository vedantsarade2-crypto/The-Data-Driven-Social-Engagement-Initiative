import requests
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")

df = pd.read_csv(
    "data/raw/youtube/video_search_results.csv"
)

video_ids = df["video_id"].tolist()

records = []

for i in range(0, len(video_ids), 50):

    batch = video_ids[i:i+50]

    params = {
        "part": "statistics,snippet",
        "id": ",".join(batch),
        "key": API_KEY
    }

    response = requests.get(
        "https://www.googleapis.com/youtube/v3/videos",
        params=params
    )

    data = response.json()

    for video in data.get("items", []):

        statistics = video.get("statistics", {})

        records.append({
            "video_id": video["id"],
            "views": statistics.get("viewCount", 0),
            "likes": statistics.get("likeCount", 0),
            "comments_count": statistics.get("commentCount", 0)
        })

stats_df = pd.DataFrame(records)

stats_df.to_csv(
    "data/raw/youtube/video_statistics.csv",
    index=False
)

print(stats_df.head())


#It is just used for to merge the file

# import pandas as pd

# metadata = pd.read_csv(
#     "data/raw/youtube/video_search_results.csv"
# )

# stats = pd.read_csv(
#     "data/raw/youtube/video_statistics.csv"
# )

# youtube_df = metadata.merge(
#     stats,
#     on="video_id",
#     how="left"
# )

# youtube_df.to_csv(
#     "data/raw/youtube/youtube_combined.csv",
#     index=False
# )

# print(youtube_df.shape)
# print(youtube_df.head())