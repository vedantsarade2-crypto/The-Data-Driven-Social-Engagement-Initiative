By mistakenly I ran "rm -rf .git" in gitbash thats why all old commits got deleted

# The Data-Driven Social Engagement Initiative

An end-to-end data science and machine learning project designed to analyze social media content performance, understand audience behavior, identify viral content patterns, analyze audience sentiment and problem awareness, detect emerging trends, and generate data-driven content recommendations.

The project combines data collection, data preprocessing, exploratory data analysis, NLP, statistical analysis, machine learning, trend analysis, recommendation systems, MySQL database design, and an interactive Flask dashboard.

---

## 🚀 Live Demo

### Interactive Dashboard
**Vercel Deployment:**  
https://the-data-driven-social-engagement-i-nine.vercel.app/

### GitHub Repository
**Source Code:**  
https://github.com/omkarsatpute18/The-Data-Driven-Social-Engagement-Initiative

---

## 📌 Project Overview

Social media platforms generate large amounts of engagement data, but raw metrics alone do not explain why certain content performs better than others.

This project attempts to answer questions such as:

- Which topics perform better?
- What patterns are associated with viral content?
- How does audience sentiment vary across content?
- How relatable is the content to the audience?
- How frequently do comments indicate awareness of a problem?
- Can machine learning identify content with higher viral potential?
- Which topics should be prioritized for future content?
- Are there emerging trends that can influence content strategy?
- How can historical performance, audience response, ML predictions, and trends be combined into recommendations?

The final system provides an interactive dashboard for exploring these insights.

---

## 🎯 Objectives

The main objectives of the project are:

1. Collect social media content and engagement data.
2. Clean and validate the collected datasets.
3. Perform exploratory data analysis.
4. Analyze audience comments using NLP.
5. Measure sentiment, relatability, and problem awareness.
6. Identify patterns associated with high-performing content.
7. Define and analyze viral content.
8. Design A/B testing experiments for content characteristics.
9. Build machine learning models for viral-content prediction.
10. Develop a topic-based recommendation engine.
11. Analyze emerging trends.
12. Combine historical, ML, and trend signals into trend-aware recommendations.
13. Store structured data using MySQL.
14. Build and deploy an interactive Flask dashboard.

---

## 🏗️ System Architecture

```text
Social Media / YouTube Data
            │
            ▼
      Data Collection
            │
            ▼
     Raw CSV Datasets
            │
            ▼
   Data Cleaning & Validation
            │
            ▼
   Feature Engineering
            │
     ┌──────┼──────────┐
     │      │          │
     ▼      ▼          ▼
   EDA     NLP       Statistics
     │      │          │
     └──────┼──────────┘
            ▼
     Machine Learning
            │
            ▼
    Viral Predictions
            │
     ┌──────┼───────────┐
     │      │           │
     ▼      ▼           ▼
Recommendation  Trend   A/B Testing
   Engine     Analysis    Design
     │          │
     └──────┬───┘
            ▼
   Final Recommendations
            │
            ▼
      Flask Dashboard
            │
            ▼
       Vercel Deployment
```

---

## 🛠️ Technologies Used

### Programming
- Python
- SQL
- HTML
- CSS
- JavaScript

### Data Analysis
- Pandas
- NumPy
- SciPy
- Matplotlib
- Seaborn

### Machine Learning
- Scikit-learn
- Logistic Regression
- Random Forest
- Feature Engineering
- Stratified Train/Test Split
- Cross-Validation

### NLP
- NLTK
- TextBlob
- Rule-based text analysis

### Statistical Analysis
- Spearman correlation
- Descriptive statistics
- Percentile analysis
- A/B experiment design

### Database
- MySQL
- MySQL Connector/Python

### Web Application
- Flask
- HTML
- CSS
- JavaScript

### Deployment
- Vercel
- GitHub

---

## 📊 Data Collection

The project uses YouTube Data API-based data collection for social content analysis.

The initial content topics include:

- Overthinking
- Social Anxiety
- Academic Pressure
- Career Anxiety
- Loneliness
- Friendship Problems
- Dating Anxiety
- Self Confidence
- Procrastination
- Student Stress

The collected datasets include:

- Video search results
- Video statistics
- YouTube comments
- Trend data
- Content-series experiment data

### Data Collection Summary

- 200 video search records
- 196 unique videos
- 15,682 comments
- 181 videos with comments
- 902 duplicate comment rows identified
- 2 empty comments identified

Duplicate and invalid records were handled during preprocessing.

---

## 🧹 Data Processing

The project contains dedicated preprocessing and validation pipelines.

Major processing steps include:

1. Duplicate removal
2. Missing-value checking
3. Text cleaning
4. Comment preprocessing
5. Feature extraction
6. Dataset merging
7. NLP feature generation
8. ML feature engineering
9. Train/test preparation
10. Final dataset validation

### Final Validation

The project validates 19 major datasets.

```text
Total files checked: 19
Passed: 19
Failed: 0
```

All required datasets passed the final validation process.

---

## 🧠 NLP Analysis

The NLP pipeline analyzes audience comments to understand audience response.

### Sentiment Analysis

From 14,169 processed comments:

| Sentiment | Count | Percentage |
|---|---:|---:|
| Positive | 6,146 | 43.38% |
| Neutral | 5,662 | 39.96% |
| Negative | 2,361 | 16.67% |

Average sentiment score:

```text
0.0978
```

### Relatability Analysis

The analysis identified:

```text
Relatable comments: 607
Neutral comments: 13,562
Relatability rate: approximately 4.28%
```

### Problem Awareness

The analysis identified:

```text
Problem-aware comments: 4,235
Not-aware comments: 9,934
Problem-awareness rate: approximately 29.89%
```

The relatability and problem-awareness components use keyword and phrase-based rules and should therefore be interpreted as analytical indicators rather than definitive psychological measurements.

---

## 📈 Exploratory Data Analysis

The project analyzes:

- Views
- Likes
- Comments
- Engagement
- Sentiment
- Relatability
- Problem awareness
- Topics
- Publishing time
- Publishing day
- Title characteristics

The dataset contains substantial variation in views, with several highly popular videos creating a strongly right-skewed distribution.

---

## 🔥 Viral Content Analysis

Viral content was defined using the 90th percentile of video views.

```text
Viral threshold: 4,278,044 views
Total videos: 196
Viral videos: 20
Non-viral videos: 176
```

This corresponds to approximately 10.2% of the dataset being classified as viral.

The most represented topics among viral videos included:

- Self Confidence
- Loneliness
- Overthinking
- Friendship Problems
- Procrastination

These patterns are observational and should not be interpreted as causal relationships.

---

## 📊 Statistical Analysis

Spearman correlation was used to examine relationships between content performance and audience-response features.

Important observations include:

- Views and likes showed a strong positive association.
- Views and comments also showed a strong positive association.
- Relatability had a weak positive association with views.
- Problem awareness did not show a statistically significant relationship with views.
- Average sentiment did not show a statistically significant relationship with views.

The analysis is observational and does not establish causality.

---

## 🧪 A/B Testing

The project includes an A/B experiment design framework for evaluating content characteristics such as:

- Content length
- Content format
- Posting period
- Hook type
- Topic

A small content-series dataset was created for experiment design.

The current dataset is intended for **experiment planning**, not for claiming statistically significant A/B test results.

Because the available experiment data does not contain sufficient controlled performance outcomes, no artificial winners or p-values were generated.

Future experiments should use controlled groups with sufficient sample sizes and measurable performance outcomes.

---

## 🤖 Machine Learning

The project predicts whether content has a higher probability of becoming viral.

### Target Definition

```text
viral = 1
```

when views are greater than or equal to the 90th percentile threshold.

### Features

The model uses pre-publication characteristics such as:

- Topic
- Publishing hour
- Publishing day
- Weekend indicator
- Title length
- Title word count
- Question marks
- Exclamation marks
- Digits
- Uppercase characters
- Presence of questions
- Presence of numbers

Performance metrics such as views, likes, and comments were excluded from prediction features to reduce target leakage.

---

## 📊 Model Comparison

Two baseline models were evaluated.

### Logistic Regression

Test performance:

```text
Accuracy:  0.6750
Precision: 0.1538
Recall:    0.5000
F1 Score:  0.2353
ROC-AUC:   0.5417
```

### Random Forest

Test performance:

```text
Accuracy:  0.8500
Precision: 0
Recall:    0
F1 Score:  0
ROC-AUC:   0.4236
PR-AUC:    0.1058
```

Although Random Forest achieved higher accuracy, it failed to identify the minority viral class in the test set.

Logistic Regression was therefore selected as the final model because it provided better minority-class recall and overall classification trade-offs.

Due to the relatively small number of viral examples, model metrics should be interpreted cautiously.

---

## 🎯 Recommendation Engine

The recommendation system combines several signals to rank content topics.

The topic recommendation score uses:

```text
30% Median Views
25% Viral Rate
20% Average ML Probability
15% Relatability
10% Problem Awareness
```

The highest-ranked topics included:

1. Loneliness
2. Self Confidence
3. Overthinking
4. Friendship Problems
5. Academic Pressure

The recommendation engine is heuristic-based and is intended to support decision-making rather than automatically determine content strategy.

---

## 📈 Trend Analysis

Trend analysis was performed using search volume and hashtag volume.

The available trend dataset contains observations for the keyword:

```text
overthinking
```

Observed growth included increases in:

- Search volume
- Hashtag volume
- Combined trend score

The trend-aware recommendation engine combines:

```text
40% Historical Performance
30% ML Prediction
30% Trend Signal
```

Because the available trend dataset is small, the trend analysis should be considered a preliminary signal rather than a robust forecasting system.

---

## 🗄️ MySQL Database

The project includes a relational MySQL database designed to store:

- Content
- Performance metrics
- Comments
- Sentiment results
- Trend data
- Experiments
- Recommendations

Main database tables:

```text
content
performance_metrics
comments
sentiment_results
trend_data
experiments
recommendations
```

SQL schema:

```text
database/sql/social_engagement_db.sql
```

---

## 🖥️ Interactive Dashboard

The project includes a Flask-based interactive dashboard.

### Dashboard Sections

#### Overview
Provides:

- Key performance indicators
- Viral content overview
- Topic performance
- Overall project metrics

#### Content Analysis
Provides:

- Video performance
- Topic comparisons
- Engagement information
- Content-level analysis

#### Audience Analysis
Provides:

- Sentiment distribution
- Relatability
- Problem awareness
- Audience response metrics

#### Trend Analysis
Provides:

- Trend scores
- Search growth
- Hashtag growth
- Rising-topic information

#### Recommendations
Provides:

- Recommended topics
- Viral potential
- Recommended posting period
- Content strategy insights
- Reasoning behind recommendations

---

## 🌐 Deployment

The Flask dashboard has been deployed using Vercel.

### Live Application

https://the-data-driven-social-engagement-i-nine.vercel.app/

The application uses Flask as the backend and serves the processed project data through the dashboard.

Vercel's Flask deployment system supports Flask applications through its Python runtime and can use a custom entrypoint through `tool.vercel.entrypoint` when the Flask application is located in a non-default module path.

---

## 📁 Project Structure

```text
The-Data-Driven-Social-Engagement-Initiative/
│
├── dashboards/
│   ├── app.py
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css
│   │   └── js/
│   │       └── dashboard.js
│   └── templates/
│       ├── base.html
│       ├── index.html
│       ├── content.html
│       ├── audience.html
│       ├── trends.html
│       └── recommendations.html
│
├── data/
│   ├── raw/
│   │   ├── youtube/
│   │   ├── comments/
│   │   ├── tremds/
│   │   └── content_series.csv
│   │
│   ├── processed/
│   └── final/
│
├── database/
│   └── sql/
│       └── social_engagement_db.sql
│
├── notebooks/
│
├── reports/
│   ├── eda/
│   ├── forecasting/
│   ├── statistics/
│   ├── validation/
│   └── final_project_report.docx
│
├── src/
│   ├── data_collection/
│   ├── data_processing/
│   ├── forecasting/
│   ├── model/
│   ├── nlp/
│   ├── recommendations/
│   └── statistics/
│
├── .gitignore
├── .env
├── pyproject.toml
├── README.md
└── requirements.txt
```

---

## 🔐 Security

Sensitive credentials should not be committed to the repository.

The project uses environment variables for API credentials.

The `.env` file is excluded through `.gitignore`.

Example:

```text
.env
venv/
__pycache__/
*.pyc
```

API keys and other credentials should be stored securely as environment variables.

---

## ⚠️ Limitations

Several limitations should be considered when interpreting the results:

1. Public YouTube data does not expose all creator-only metrics such as saves, shares, follower growth, and retention.
2. Some recommendation components are heuristic-based.
3. The viral dataset contains relatively few positive examples.
4. ML performance is therefore sensitive to the small minority class.
5. Sentiment analysis may not fully understand sarcasm, context, or slang.
6. Relatability and problem-awareness analysis is based on keyword/phrase rules.
7. The trend dataset is currently small.
8. Trend analysis should therefore be considered preliminary.
9. A/B experiment results cannot currently establish statistical significance because controlled outcome data is limited.
10. Observational correlations should not be interpreted as causal relationships.

---

## 🚀 Future Scope

Possible improvements include:

- Collecting larger multi-platform datasets
- Adding Instagram, TikTok, and other social platforms
- Collecting creator-level metrics where permitted
- Increasing the number of controlled A/B experiments
- Using transformer-based NLP models
- Improving sarcasm and contextual sentiment detection
- Adding advanced time-series forecasting
- Using larger trend datasets
- Implementing automated data pipelines
- Adding real-time dashboard updates
- Improving model calibration
- Using advanced ensemble models
- Adding explainable AI techniques
- Connecting the dashboard directly to a production database

---

## ✅ Project Validation

The final project validation pipeline checks the major raw and processed datasets.

Final validation result:

```text
Total datasets checked: 19
Passed: 19
Failed: 0
```

The validation process also checks missing values and duplicate records to ensure data quality before analysis and modeling.

---

## 📚 Key Output Files

Important generated datasets include:

```text
data/processed/clean_comments.csv
data/processed/sentiment_comments.csv
data/processed/nlp_comments.csv
data/processed/video_nlp_dataset.csv
data/processed/ml_features.csv
data/processed/train_data.csv
data/processed/test_data.csv
data/processed/viral_predictions.csv
data/processed/topic_performance.csv
data/processed/final_recommendations.csv
data/processed/trend_analysis.csv
data/processed/rising_topics.csv
data/processed/trend_aware_recommendations.csv
```

---

## ▶️ Running the Dashboard Locally

### 1. Clone the repository

```bash
git clone https://github.com/omkarsatpute18/The-Data-Driven-Social-Engagement-Initiative.git
```

### 2. Navigate to the project

```bash
cd The-Data-Driven-Social-Engagement-Initiative
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment

Windows:

```powershell
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Start the Flask dashboard

```bash
python dashboards/app.py
```

### 7. Open the dashboard

```text
http://127.0.0.1:5000/
```

---

## 👨‍💻 Author

**Omkar Satpute**

Third-Year Electronics and Telecommunication Engineering Student

Interested in:

- Data Science
- Machine Learning
- Artificial Intelligence
- Data Analytics
- Python
- Applied Machine Learning

---

## ⭐ Project Links

**GitHub Repository:**  
https://github.com/omkarsatpute18/The-Data-Driven-Social-Engagement-Initiative

**Live Dashboard:**  
https://the-data-driven-social-engagement-i-nine.vercel.app/

---

## 📄 License

This project was developed for academic, learning, portfolio, and demonstration purposes.