# kaggle-nlp-disasters

## Overview
This project focuses on completing the [Kaggle Competition on NLP classification of disaster tweets.](https://www.kaggle.com/competitions/nlp-getting-started/overview) This is a binary classification problem where tweets are either a disaster or not a disaster. This attempt is for my CSCA 5642 assignment submission and is focused on learning rather than the best outcome. The goal of this project is to explore the data, build a feature dataset and then design and implement a model architecture that will be compared to a more basic baseline. My approach will focus mostly on how we design a feature set from text and then specifically use a model from the RNN family of neural networks. 

## Tech Stack
* Language: Python 3.12
* Core Libs:
    - pandas, numpy
    - scikit-learn (TF-IDF, logreg, metrics, train_test_split)
    - matplotlib, seaborn
    - nltk, gensim
    - tensorflow, keras
    - sentence-transformers (optional: if time allows)
## Plan

**Focus is on 3 capabilities of this model**:
1. Text --> Vectors (Representations)

* **Two** main representations of text features:
    * TF-IDF bag-of-words
        - strong baseline, sparse and works fast for linear models
    * Pre-trained embeddings (GloVe, fastText or compare both if time allows)
        - This captures semantic similarity for RNN models, lets us do cosine similarity analysis.
    * If time allows, try a contextual component like sentence-transformers

2. Text Classification / Scoring Models
* Simple baseline using logistic regression
* RNN model (LSTM or GRU, if time allows test both)

3. Comparison & Evaluation Framework
* Select evaluation frameworks
    - Kaggle submission uses F1 scores, so we should track that
    - Similarity score, cosine similarity on embeddings
    - Simple error analysis

## Text Representation
**Conduct EDA on things like:**
1. EDA Steps:
    common words
    - length of text
    - TF-IDF example
    - Embedding example
2. Two main representations:
    - TF-IDF: strong baseline, easy to understand, sparse, fast for linear model
    - GloVe: captures semantic similarity for RNN

## Basline + RNN Model
**Implement two model families for comparison:**
1. Baseline: TF-IDF + Logistic Regression
- Considering this the baseline where we use the TF-IDF features and the labels with logistic regression.
- Output: F1 on validation set
2. RNN Model:
- Tokenizer --> integer sequences --> padded to max length
- Embedding layer (random vs GloVe-initialized)
- LSTM / GRU (considering BiLSTM too) --> Dense (sigmoid)
- Output: F1 on validation set + Kaggle submission CSV


## Comparison & Evaluation Framework
**Implement a standarized evaluation framework so our experiments comparing different models and hyperparameters are fair comparisons:**
1. Data Splits:
    - Use Train / Test provided by Kaggle. This creates a "frozen" training set for consistency and comparability.
    - No external / other data points allowed, only what is provided.
    - Consideration: We could holdout a small portion to have a small test-like "holdout" for cross-validation.
2. Metrics:
    - Primary: F1 (binary) on validation
    - Secondary: accuracy, confusion matrix for error analysis
3. Error analysis (why are the things that are wrong, wrong?)
    - Inspect top 10 false positives & false negatives
    - For a few errors, find nearest neighbors in embedding space (cosine similarity) and investigate.
