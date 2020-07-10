## Problem statement
While every day should be Mothers' Day [’cuz they rock], we are all guilty of taking to social media to post mushy messages or images on the second Sunday of May for our mothers. However, the Human Resources team of your organization wishes to celebrate and honor all moms currently employed at the company with a special monthly event. For one of such events, they have reached out to your team to curate and categorize Mothers’ Day-related tweets from across the globe.

As a Machine Learning Engineer, your task is to build a sophisticated sentiment analysis-based NLP model that can classify sentiments of tweets into positive, negative, and neutral.

## Dataset

### Data description
This data set consists of six columns:

### Column Name	Description

| <b>Column</b> | <b>Description</b> | 
| :---:   | :-: |
|id|ID of tweet|
|original_text|Text of tweet|
|lang|Language of tweet|
|retweet_count|Number of times retweeted|
|original_author|Twitter handle of Author|
|sentiment_class|Sentiment of Tweet (Target)|

The data folder consists of two .csv files. The details are as follows:

train.csv: 3235 x 6
test.csv: 1387 x 5

## Evaluation metric

score={100*f1_score(actual_values,predicted_values,average='weighted')}

## More Details
https://www.hackerearth.com/challenges/competitive/hackerearth-machine-learning-challenge-mothers-day/problems/

## Outcome
- <b>Time Spent</b>: ~12 Hrs 
- <b>Current Rank</b>: 35
- <b>Leaderboard Rank</b>: 61
