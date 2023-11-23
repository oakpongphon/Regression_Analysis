This summary provides a high-level view of the various analytical techniques and models that I explored in this coursework, spanning regression, classification, and recommendation systems, with a specific focus on the gaming domain.

## Part 1: Regression Analysis
### Overview
- Data Source: Video game reviews data from _Steam_.
- Primary Focus: Analyzing the relationship between review length and hours played, using different models and transformations.
### Key Sections
- Simple Linear Regression: Modeling review length as a function of hours played, reporting coefficient and MSE.
- Multiple Regression with Transforms: Adding transformed variables of hours played, and reporting MSE.
- Model with Binary Indicators: Using a sequence of binary indicators for hours played, reporting MSE.
- Predicting Hours from Review Length: Reversing the model to predict hours based on review length, discussing MSE and MAE.
- Transformed Hours Prediction: Focusing on log-transformed hours prediction, calculating MSE in both transformed and original scales.
- Validation Pipeline: Implementing train/validation/test splits and fitting a regularized model, reporting the best regularization parameter, and MSE on validation and test sets.

## Part 2: Classification
### Overview
- Objective: Classify whether the transformed number of hours played is above or below the median, using review length.
### Key Sections
- Data Insights: Computing median of transformed hours and interactions with less than one hour played.
- Logistic Regression: Building a classifier based on review length, reporting classification metrics and BER.
- Precision@k Calculation: Computing precision at various k levels, considering ties in classifier scores.
- Alternative Classifier: Exploring a threshold on a regression model from previous questions to achieve a lower BER.
## Part 3: Recommendation
### Overview
- Goal: Recommend games that users are likely to play for a long time, using median playtime as a benchmark.
### Key Sections
- Median Playtime Analysis: Calculating per-user and per-item median values, using training set data.
- Trivial Model: Developing a basic recommendation model based on median playtimes, reporting accuracy on the test set.
- Jaccard Similarity: Finding items similar to the first item based on Jaccard similarity.
- Cosine Similarity with Binary Labels: Computing cosine similarities using binary labels for playtime above or below median.
- Cosine Similarity with Hours Transformed: Repeating the cosine similarity analysis, but using the hours transformed values.
