## Problem statement

Develop a machine learning program to identify when an article might be fake news. Run by the UTK Machine Learning Club.

## Dataset

### Data description
A full training dataset with the following attributes:

### Column Name	Description

| <b>Column</b> | <b>Description</b> | 
| :---:   | :-: |
|id | unique id for a news article|
|title | the title of a news article |
|author| author of the news article |
|text| the text of the article; could be incomplete |
|label| a label that marks the article as potentially unreliable 1: unreliable 0: reliable |


## Evaluation metric

The evaluation metric for this competition is accuracy, a very straightforward metric.

```math
accuracy= correct predictions / (correct predictions + incorrect predictions)
```
Accuracy measures false positives and false negeatives equally, and really should only be used in simple cases and when classes are of (generally) equal class size

I have also used F1 Score
```math
score={100*f1_score(actual_values,predicted_values,average='weighted')}
```
<i>About F1 Score:</i> 
- F1 Score is the weighted average of Precision and Recall. 
- Therefore, this score takes both false positives and false negatives into account. 
- F1 is usually more useful than accuracy, especially if you have an uneven class distribution.

## More Details
https://www.kaggle.com/c/fake-news/

