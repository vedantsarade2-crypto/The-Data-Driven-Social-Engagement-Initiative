from flask import Flask, render_template, jsonify
from pathlib import Path
import pandas as pd
import math

app = Flask(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

DATASETS = {
    "video_nlp": "video_nlp_dataset.csv",
    "viral": "viral_predictions.csv",
    "topics": "topic_performance.csv",
    "recommendations": "final_recommendations.csv",
    "trends": "trend_analysis.csv",
    "rising": "rising_topics.csv",
    "trend_aware": "trend_aware_recommendations.csv",
}

def load_csv(key):
    path = DATA_DIR / DATASETS[key]
    if not path.exists():
        print(f"[Missing] {path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        print(f"[Loaded] {DATASETS[key]} - {len(df)} rows")
        return df
    except Exception as exc:
        print(f"[Error] {DATASETS[key]} - {exc}")
        return pd.DataFrame()

def safe_num(value, default=0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default

def pct(value):
    return round(safe_num(value) * 100, 1)

def clean_records(df, limit=None):
    if df.empty:
        return []
    out = df.copy()
    out = out.replace({float("nan"): None})
    if limit:
        out = out.head(limit)
    return out.to_dict(orient="records")

def build_summary():
    videos = load_csv("video_nlp")
    viral = load_csv("viral")
    topics = load_csv("topics")
    rising = load_csv("rising")

    total_videos = len(videos)
    total_views = int(videos["views"].sum()) if "views" in videos else 0
    total_likes = int(videos["likes"].sum()) if "likes" in videos else 0
    total_comments = int(videos["comments_count"].sum()) if "comments_count" in videos else 0
    viral_videos = int((viral["viral"] == 1).sum()) if "viral" in viral else 0

    avg_engagement = safe_num(videos["engagement_rate"].mean()) if "engagement_rate" in videos else 0
    avg_sentiment = safe_num(videos["average_sentiment"].mean()) if "average_sentiment" in videos else 0
    avg_relatability = safe_num(videos["relatability_rate"].mean()) if "relatability_rate" in videos else 0
    avg_problem = safe_num(videos["problem_awareness_rate"].mean()) if "problem_awareness_rate" in videos else 0

    top_topic = "—"
    top_topic_score = 0
    if not topics.empty and "topic" in topics and "recommendation_score" in topics:
        row = topics.sort_values("recommendation_score", ascending=False).iloc[0]
        top_topic = str(row["topic"])
        top_topic_score = safe_num(row["recommendation_score"])

    rising_topic = "—"
    rising_score = 0
    if not rising.empty and "keyword" in rising:
        row = rising.sort_values("rising_score", ascending=False).iloc[0]
        rising_topic = str(row["keyword"])
        rising_score = safe_num(row.get("rising_score", 0))

    return {
        "total_videos": total_videos,
        "total_views": total_views,
        "total_likes": total_likes,
        "total_comments": total_comments,
        "viral_videos": viral_videos,
        "viral_rate": round((viral_videos / total_videos * 100), 1) if total_videos else 0,
        "avg_engagement": round(avg_engagement, 2),
        "avg_sentiment": round(avg_sentiment, 3),
        "avg_relatability": round(avg_relatability, 2),
        "avg_problem_awareness": round(avg_problem, 2),
        "top_topic": top_topic,
        "top_topic_score": round(top_topic_score, 3),
        "rising_topic": rising_topic,
        "rising_score": round(rising_score, 3),
    }

def chart_data():
    videos = load_csv("video_nlp")
    topics = load_csv("topics")
    trends = load_csv("trends")
    rising = load_csv("rising")
    viral = load_csv("viral")

    topic_chart = []
    if not topics.empty and {"topic", "median_views"}.issubset(topics.columns):
        t = topics.sort_values("median_views", ascending=False).head(8)
        for _, r in t.iterrows():
            topic_chart.append({
                "label": str(r["topic"]),
                "value": round(safe_num(r["median_views"])),
                "viral_rate": round(safe_num(r.get("viral_rate", 0)) * 100, 1)
            })

    sentiment_chart = []
    if not videos.empty and {"topic", "average_sentiment"}.issubset(videos.columns):
        s = videos.groupby("topic", as_index=False)["average_sentiment"].mean()
        s = s.sort_values("average_sentiment", ascending=False).head(8)
        sentiment_chart = [{"label": str(r["topic"]), "value": round(safe_num(r["average_sentiment"]), 3)} for _, r in s.iterrows()]

    trend_chart = []
    if not trends.empty and {"trend_date", "trend_score"}.issubset(trends.columns):
        tr = trends.sort_values("trend_date")
        for _, r in tr.iterrows():
            trend_chart.append({
                "label": str(r["trend_date"])[:10],
                "value": round(safe_num(r["trend_score"]), 3),
                "search": int(safe_num(r.get("search_volume", 0))),
                "hashtag": int(safe_num(r.get("hashtag_volume", 0)))
            })

    viral_chart = []
    if not viral.empty and {"viral_potential"}.issubset(viral.columns):
        counts = viral["viral_potential"].astype(str).value_counts()
        order = ["High", "Medium", "Low"]
        viral_chart = [{"label": x, "value": int(counts.get(x, 0))} for x in order]

    return {
        "topic": topic_chart,
        "sentiment": sentiment_chart,
        "trend": trend_chart,
        "viral": viral_chart,
    }

@app.route("/")
def index():
    summary = build_summary()
    topics = load_csv("topics")
    videos = load_csv("video_nlp")
    viral = load_csv("viral")

    top_topics = clean_records(
        topics.sort_values("recommendation_score", ascending=False)
        if not topics.empty and "recommendation_score" in topics.columns else topics,
        6
    )
    top_videos = clean_records(
        viral.sort_values("viral_probability", ascending=False)
        if not viral.empty and "viral_probability" in viral.columns else viral,
        8
    )

    return render_template(
        "index.html",
        summary=summary,
        charts=chart_data(),
        top_topics=top_topics,
        top_videos=top_videos,
        page="overview"
    )

@app.route("/content")
def content():
    videos = load_csv("video_nlp")
    viral = load_csv("viral")
    if not videos.empty and not viral.empty and "video_id" in videos and "video_id" in viral:
        keep = [c for c in ["video_id", "viral_probability", "viral_potential"] if c in viral.columns]
        videos = videos.merge(viral[keep], on="video_id", how="left")
    records = clean_records(videos.sort_values("views", ascending=False) if not videos.empty and "views" in videos.columns else videos)
    return render_template("content.html", records=records, page="content")

@app.route("/audience")
def audience():
    videos = load_csv("video_nlp")
    summary = build_summary()
    topic_rows = []
    if not videos.empty and "topic" in videos.columns:
        cols = [c for c in ["topic", "average_sentiment", "relatability_rate", "problem_awareness_rate", "total_comments"] if c in videos.columns]
        g = videos.groupby("topic", as_index=False)[cols[1:]].mean().sort_values("relatability_rate", ascending=False)
        topic_rows = clean_records(g)
    return render_template("audience.html", summary=summary, topic_rows=topic_rows, charts=chart_data(), page="audience")

@app.route("/trends")
def trends():
    trends_df = load_csv("trends")
    rising = load_csv("rising")
    aware = load_csv("trend_aware")
    trend_rows = clean_records(trends_df.sort_values("trend_date") if not trends_df.empty and "trend_date" in trends_df.columns else trends_df)
    rising_rows = clean_records(rising.sort_values("rank") if not rising.empty and "rank" in rising.columns else rising)
    rec_rows = clean_records(aware.sort_values("final_rank") if not aware.empty and "final_rank" in aware.columns else aware, 10)
    return render_template("trends.html", trends=trend_rows, rising=rising_rows, recommendations=rec_rows, charts=chart_data(), page="trends")

@app.route("/recommendations")
def recommendations():
    final = load_csv("recommendations")
    aware = load_csv("trend_aware")
    final_rows = clean_records(final)
    aware_rows = clean_records(aware.sort_values("final_rank") if not aware.empty and "final_rank" in aware.columns else aware, 10)
    return render_template("recommendations.html", recommendations=final_rows, aware=aware_rows, page="recommendations")

@app.route("/api/summary")
def api_summary():
    return jsonify({"summary": build_summary(), "charts": chart_data()})

if __name__ == "__main__":
    build_summary()
    print(f"[Data directory] {DATA_DIR}")
    app.run(debug=True)
