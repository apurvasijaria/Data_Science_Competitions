## Problem statement

To build a sophisticated sentiment analysis-based NLP model that can classify sentiments of Mother's day tweets into positive, negative, and neutral.

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
```math
score={100*f1_score(actual_values,predicted_values,average='weighted')}
```
## More Details
https://www.hackerearth.com/challenges/competitive/hackerearth-machine-learning-challenge-mothers-day/problems/

## Outcome
- <b>Time Spent</b>: ~12 Hrs 
- <b>Current Rank</b>: 35
- <b>Leaderboard Rank</b>: 61
