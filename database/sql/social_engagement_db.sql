CREATE DATABASE social_engagement_db;
USE social_engagement_db;
SELECT DATABASE();

CREATE TABLE content (
    content_id VARCHAR(20) PRIMARY KEY,
    content_date DATE NOT NULL,
    platform VARCHAR(50) NOT NULL,
    topic VARCHAR(100) NOT NULL,
    content_format VARCHAR(20) NOT NULL,
    hook_type VARCHAR(20) NOT NULL,
    video_length_sec INT,
    posting_time TIME,
    posting_period VARCHAR(20),
    caption_style VARCHAR(50),
    caption_text TEXT,
    hashtags TEXT
);
DESCRIBE content;

CREATE TABLE performance_metrics (
    performance_id VARCHAR(20) PRIMARY KEY,
    content_id VARCHAR(20) NOT NULL,
    views INT DEFAULT 0,
    likes INT DEFAULT 0,
    comments_count INT DEFAULT 0,
    shares INT DEFAULT 0,
    saves INT DEFAULT 0,
    retention_rate DECIMAL(5,2),
    followers_before INT DEFAULT 0,
    followers_after INT DEFAULT 0,
    reach INT DEFAULT 0,

    FOREIGN KEY (content_id)
        REFERENCES content(content_id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);

CREATE TABLE comments (
    comment_id VARCHAR(20) PRIMARY KEY,
    content_id VARCHAR(20) NOT NULL,
    comment_date DATE,
    comment_text TEXT NOT NULL,
    platform VARCHAR(50),

    FOREIGN KEY (content_id)
        REFERENCES content(content_id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);

CREATE TABLE sentiment_results (
    result_id VARCHAR(20) PRIMARY KEY,
    comment_id VARCHAR(20) NOT NULL,
    sentiment VARCHAR(20),
    sentiment_score DECIMAL(6,4),
    relatability VARCHAR(20),
    relatability_score DECIMAL(6,4),
    problem_awareness VARCHAR(10),

    FOREIGN KEY (comment_id)
        REFERENCES comments(comment_id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);

CREATE TABLE trend_data (
    trend_id VARCHAR(20) PRIMARY KEY,
    trend_date DATE NOT NULL,
    keyword VARCHAR(100) NOT NULL,
    category VARCHAR(100),
    search_volume INT DEFAULT 0,
    hashtag_volume INT DEFAULT 0,
    trend_score DECIMAL(8,2)
);

CREATE TABLE experiments (
    experiment_id VARCHAR(20) PRIMARY KEY,
    content_id VARCHAR(20) NOT NULL,
    experiment_name VARCHAR(100),
    experiment_group VARCHAR(50),
    metric_name VARCHAR(50),
    metric_value DECIMAL(10,4),

    FOREIGN KEY (content_id)
        REFERENCES content(content_id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);

CREATE TABLE recommendations (
    recommendation_id VARCHAR(20) PRIMARY KEY,
    recommendation_date DATE NOT NULL,
    recommended_topic VARCHAR(100),
    recommended_format VARCHAR(50),
    recommended_length VARCHAR(50),
    recommended_hook VARCHAR(50),
    recommended_caption_style VARCHAR(50),
    recommended_posting_period VARCHAR(50),
    expected_viral_probability DECIMAL(6,4),
    reasoning TEXT
);

INSERT INTO content
(content_id, content_date, platform, topic, content_format,
 hook_type, video_length_sec, posting_time, posting_period,
 caption_style, caption_text, hashtags)
VALUES
('C001', '2026-09-01', 'Instagram', 'Overthinking',
 'Short', 'Text', 25, '19:30:00', 'Evening',
 'Question',
 'Do you ever replay conversations in your head?',
 '#overthinking #mentalhealth'),

('C002', '2026-09-02', 'Instagram', 'Social Anxiety',
 'Short', 'Visual', 30, '20:00:00', 'Evening',
 'Statement',
 'Sometimes you want to talk but do not know how.',
 '#socialanxiety #relatable'),

('C003', '2026-09-03', 'YouTube', 'Career Anxiety',
 'Long', 'Text', 70, '21:00:00', 'Night',
 'Story',
 'Everyone asks what you want to become.',
 '#career #studentlife');
 
 INSERT INTO performance_metrics
(performance_id, content_id, views, likes, comments_count,
 shares, saves, retention_rate,
 followers_before, followers_after, reach)
VALUES
('P001', 'C001', 12000, 950, 130, 420, 610, 72.50, 10000, 10250, 10500),

('P002', 'C002', 18000, 1500, 240, 620, 800, 81.40, 10250, 10600, 16000),

('P003', 'C003', 15000, 1100, 180, 250, 300, 58.20, 10600, 10750, 14000);



SELECT
    c.content_id,
    c.topic,
    c.content_format,
    c.hook_type,
    p.views,
    p.likes,
    p.comments_count,
    p.shares,
    p.saves,
    p.retention_rate
FROM content c
JOIN performance_metrics p
    ON c.content_id = p.content_id;
    
    INSERT INTO comments
(comment_id, content_id, comment_date, comment_text, platform)
VALUES
('CM001', 'C001', '2026-09-01',
 'This is literally me. I overthink everything.',
 'Instagram'),

('CM002', 'C001', '2026-09-01',
 'I thought I was the only one who does this.',
 'Instagram'),

('CM003', 'C001', '2026-09-01',
 'Nice video.',
 'Instagram'),

('CM004', 'C002', '2026-09-02',
 'This describes my life perfectly.',
 'Instagram'),

('CM005', 'C002', '2026-09-02',
 'Going through exactly this right now.',
 'Instagram');
 
 
 SELECT
    c.content_id,
    c.topic,
    cm.comment_id,
    cm.comment_text
FROM content c
JOIN comments cm
    ON c.content_id = cm.content_id;
    
    SELECT
    c.content_id,
    c.topic,
    p.views,
    p.likes,
    p.comments_count,
    p.shares,
    p.saves,

    ROUND(
        (p.likes + p.comments_count + p.shares + p.saves)
        / p.views * 100,
        2
    ) AS engagement_rate

FROM content c
JOIN performance_metrics p
    ON c.content_id = p.content_id;
    
    SELECT
    content_id,
    ROUND((shares / views) * 100, 2) AS share_rate
FROM performance_metrics;

SELECT
    content_id,
    ROUND((saves / views) * 100, 2) AS save_rate
FROM performance_metrics;

SELECT
    content_id,
    ROUND(saves / NULLIF(shares, 0), 2) AS save_to_share_ratio
FROM performance_metrics;

SELECT
    content_id,
    followers_before,
    followers_after,
    (followers_after - followers_before) AS follower_growth
FROM performance_metrics;

SELECT
    content_id,
    ROUND(
        (followers_after - followers_before)
        / NULLIF(followers_before, 0) * 100,
        2
    ) AS follower_growth_percentage
FROM performance_metrics;

USE social_engagement_db;

ALTER TABLE content
ADD COLUMN source_type VARCHAR(30),
ADD COLUMN source_url TEXT,
ADD COLUMN external_id VARCHAR(100);

ALTER TABLE performance_metrics
ADD COLUMN data_source VARCHAR(50);

ALTER TABLE comments
ADD COLUMN source_type VARCHAR(30);