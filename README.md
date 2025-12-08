# kaggle-nlp-disasters

## Plan

**Focus is on 3 capabilities of this model**:
1. Text --> Vectors (Representations)
* Bag-of-words / TF-IDF
* Pre-trained embeddings (GloVe, fastText or compare both if time allows)
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
- common words
- length of text
- TF-IDF example
- Embedding example

## Basline + RNN Model
**Implement two model families for comparison:**
1. Baseline: TF-IDF + Logistic Regression
- Considering this the baseline where we use the TF-IDF features and the labels with logistic regression.
2. RNN Model:
- Embedding layer
- LSTM / GRU (considering BiLSTM too)
- Dense layer -->  sigmoid output
- Tune hyperparameters via experiments:
    - hidden size
    - dropout rate
    - pretrained embeddings

## Comparison & Evaluation Framework
**Implement a standarized evaluation framework so our experiments comparing different models and hyperparameters are fair comparisons:**
1. Create a "frozen" training set so we are training models on the same data (same for test).
2. No external / other data points allowed, only what is provided.
3. F1 will be our "what is best for Kaggle" metric, since that is the evaluation used in the competition.
4. Error analysis (why are the things that are wrong, wrong?)
    - Leverage something like similarity scores on the embeddings to see if the errors are similar or not.