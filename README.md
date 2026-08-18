![banner](./assets/images/banner.jpg)

# 📰 **news-category**

Classifying [HuffPost news headlines](https://www.kaggle.com/datasets/rmisra/news-category-dataset/data) into 33 topic categories, comparing four models of increasing sophistication: **fine-tuned DistilBERT**, a **BiLSTM** over GloVe embeddings, **Logistic Regression**, and **Naive Bayes**. A fine-tuned DistilBERT reaches **0.762 macro-F1** (0.820 accuracy) on the held-out test set, while the BiLSTM over static embeddings underperforms both classical baselines.


## 🏆 **Results**

Best macro-F1 each model achieves, with the field combination that produces it:

![Best Macro-F1 per Model](./assets/images/best_by_model.png)

| Rank | Model               | Macro-F1  | Accuracy  | Best Field Combination                 |
|:----:|:------------------- |:---------:|:---------:|:-------------------------------------- |
|  1   | **DistilBERT**      | **0.762** | **0.820** | headline + short_description + authors |
|  2   | Logistic Regression |   0.713   |   0.778   | headline + short_description + authors |
|  3   | Naive Bayes         |   0.695   |   0.764   | headline + short_description + authors |
|  4   | BiLSTM (GloVe)      |   0.670   |   0.740   | headline + authors                     |


## 📚 **Table of Contents**

- [🗺️ Overview](#%EF%B8%8F-overview)
  - [🧠 Models](#-models)
  - [🔀 Field Combinations](#-field-combinations)
  - [📏 Metric](#-metric)

- [🗂️ Dataset](#%EF%B8%8F-dataset)

- [🧹 Preprocessing](#-preprocessing)
  - [🌳 Taxonomy Merge](#-taxonomy-merge)
  - [🔗 Field Assembly](#-field-assembly)
  - [⚖️ Class Weighting](#%EF%B8%8F-class-weighting)

- [🧪 Modeling](#-modeling)
  - [🔨 Classical models](#-classical-models-naive-bayes-logistic-regression)
  - [⚙️ BiLSTM over GloVe](#%EF%B8%8F-bilstm-over-glove)
  - [🤖 DistilBERT](#-distilbert)

- [📈 Evaluation](#-evaluation)
  - [🔍 Error Analysis](#-error-analysis)
  - [📉 BiLSTM Limitations](#-bilstm-limitations)

- [🚀 Future Work](#-future-work)

- [📁 Repository Structure](#-repository-structure)


## 🗺️ **Overview**

The goal is a multi-class classifier that assigns a topic category to a short news headline. We approach it as a controlled comparison across four model families, holding the task, data split, and evaluation metric fixed while increasing model sophistication.

### 🧠 **Models**

1. **Naive Bayes**: Multinomial Naive Bayes over TF-IDF features.

2. **Logistic Regression**: linear classifier over TF-IDF features.

3. **BiLSTM**: bidirectional LSTM over trainable GloVe word embeddings.

4. **DistilBERT**: a pretrained transformer fine-tuned end to end.

### 🔀 **Field Combinations**

Every model is evaluated on **7 field combinations** built from the `headline`, `short_description`, and `authors` fields (each field alone, each pair, and all three), so we can see not only which model wins but which inputs carry the signal.

### 📏 **Metric**

The primary metric is **macro-averaged F1**, chosen because the categories are heavily imbalanced. Macro-F1 weights every category equally and is therefore not flattered by strong performance on a few large classes (as raw accuracy is).


## 🗂️ **Dataset**

> [`01_preprocessing.ipynb`](01_preprocessing.ipynb)

- **Source:** [News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset/data) (HuffPost, ~209,000 headlines).

- **Fields used:** `headline`, `short_description`, `authors`. The `link` field is deliberately excluded because its URL slug leaks the label.

- **Split:** a single stratified 80/20 train/test split. Deep models further carve a stratified 10% validation slice from the training set for early stopping.
Because the dataset ships as one flat collection, our test set is a random draw from the same distribution as training. This measures **in-distribution generalization** rather than robustness to distribution shift. See [Future Work](#-future-work).

  ![Class Distribution](./assets/images/class_distribution.png)


## 🧹 **Preprocessing**

### 🌳 **Taxonomy Merge**

> [`02_taxonomy.ipynb`](02_taxonomy.ipynb)

The raw dataset contains near-duplicate labels (e.g. overlapping culture, parenting, and wellness tags). We merge these via a versioned mapping so the label space is consistent and non-redundant. The `v1` merge groups are:

```
                         ╭─      ARTS
      ╭─ ARTS & CULTURE ─┼─ ARTS & CULTURE
      │                  ╰─ CULTURE & ARTS
      │                  ╭─  ENVIRONMENT
      ├─  ENVIRONMENT   ─┤
      │                  ╰─     GREEN
      │                  ╭─  FOOD & DRINK
      ├─  FOOD & DRINK  ─┤
      │                  ╰─     TASTE
      │                  ╭─   PARENTING
─ v1 ─┼─   PARENTING    ─┤
      │                  ╰─    PARENTS
      │                  ╭─     STYLE
      ├─ STYLE & BEAUTY ─┤
      │                  ╰─ STYLE & BEAUTY
      │                  ╭─ HEALTHY LIVING
      ├─    WELLNESS    ─┤
      │                  ╰─    WELLNESS
      │                  ╭─ THE WORLDPOST
      ╰─   WORLD NEWS   ─┼─   WORLD NEWS
                         ╰─   WORLDPOST
```

### 🔗 **Field Assembly**

For multi-field combinations, fields are concatenated with a separator token, letting a single model consume any subset of fields uniformly.

### ⚖️ **Class Weighting**

All models apply class weighting (balanced class weights for the classical and BiLSTM models, and a class-weighted cross-entropy loss for DistilBERT) so that rare categories are not ignored.


## 🧪 **Modeling**

Each model was tuned appropriately for its family. Hyperparameters are saved per model under `results/<model>/best_params.json`.

### 🔨 **Classical models (Naive Bayes, Logistic Regression)**

> [`03_naive_bayes.ipynb`](03_naive_bayes.ipynb), [`04_logistic_regression.ipynb`](04_logistic_regression.ipynb)

Grid-searched with 5-fold cross-validation over TF-IDF settings (n-gram range, minimum document frequency, sublinear scaling, stop-word removal) and model-specific hyperparameters (smoothing for Naive Bayes, and regularization strength `C` for Logistic Regression). Both peak with bigram TF-IDF and a minimum document frequency of 5.

```
── headline + short_description + authors ──
alpha         0.1
fit_prior     False
min_df        5
ngram_range   (1, 2)
stop_words    english
sublinear_tf  True
```

```
── headline + short_description + authors ──
C             10
class_weight  balanced
min_df        5
ngram_range   (1, 2)
sublinear_tf  True
```

### ⚙️ **BiLSTM over GloVe**

> [`05_bilstm.ipynb`](05_bilstm.ipynb)

A bidirectional LSTM that reads each headline as a sequence of GloVe word vectors. Two preparation steps mattered most:

- **Matching the text to the vocabulary.** GloVe only knows lowercase words, so lowercasing the headlines let far more words be recognized instead of treated as unknown. This was the single largest improvement for this model.

- **Learning the unknown words.** Words GloVe has never seen, such as many author names, start with random vectors that the model is allowed to train, rather than all sharing one generic "unknown" vector. The pretrained vectors are also left unfrozen so the model can adjust them.

```
── headline + authors ─────────────────────
  embedding      glove-wiki-gigaword-300
  hidden_size    128
  dropout        0.3
  oov_min_freq   3
  batch_size     64
  learning_rate  1e-3
  max_epochs     30 (early stopping, patience 5)
```

### 🤖 **DistilBERT**

> [`06_distilbert.ipynb`](06_distilbert.ipynb)

`distilbert-base-uncased` fine-tuned end to end with a class-weighted loss, early stopping on validation macro-F1, and best-checkpoint restoration. A small joint sweep of batch size and learning rate on two representative field combinations selected batch size 64 and a learning rate of 5e-5.

```
── headline + short_description + authors ──
  checkpoint     distilbert-base-uncased
  max_length     128
  batch_size     64
  learning_rate  5e-5
  max_epochs     12 (early stopping, patience 3)
```


## 📈 **Evaluation**

> [`07_comparison.ipynb`](07_comparison.ipynb)

All models are compared on the same held-out test set and the same seven field combinations. Full per-model diagnostics (confusion matrices, per-class F1, and learning/training curves for every combination) live in the individual notebooks.

![Macro-F1 by Field Combination](assets/images/score_by_config.png)

Two findings hold across **every** model:

- **`authors` is a powerful signal.** Every model's strongest combinations include the author field, and the mapping from a byline to a section is highly predictive. Adding `authors` to `headline` lifts macro-F1 by roughly 0.15 to 0.2 for every model.

- **`short_description` is the weakest field in isolation** (macro-F1 ~0.34 to 0.42 everywhere). It is verbose but low in signal per token relative to the terse, information-dense headline.

### 🔍 **Error Analysis**

The per-class F1 charts and confusion matrices tell a consistent story across all four models: the same handful of categories fail everywhere.

- **Consistently strong** (distinctive vocabulary): `STYLE & BEAUTY`, `FOOD & DRINK`, `WEDDINGS`, `TRAVEL`, `HOME & LIVING`.

- **Consistently weak** (even for DistilBERT): `U.S. NEWS`, `FIFTY`, `IMPACT`, and the broader `WORLD NEWS` / `WEIRD NEWS` / `GOOD NEWS` cluster.

**The failures are driven by semantic ambiguity, not model capacity.** The weak categories are hard to separate because they overlap in the underlying text, in two distinct ways:

- **Mutually confusable "news" labels.** `U.S. NEWS`, `POLITICS`, and `WORLD NEWS` are defined by venue or framing rather than by topic vocabulary. A headline about a domestic political event is a plausible member of all three at once, so the confusion matrices show them bleeding into one another.

- **Themes without a vocabulary.** Editorially-defined buckets like `FIFTY` (content aimed at readers over 50) and `IMPACT` (social-good stories) are cross-cutting themes, not topics. They share no distinctive words with each other, so there is little for any model to key on.

In both cases the label boundaries themselves are soft, and no model can cleanly separate categories that overlap in the text. This also explains why contextual DistilBERT helps most exactly here: contextual understanding disambiguates borderline cases that bag-of-words and static embeddings cannot, which is why its lead over the other models is largest on the vague categories.

### 📉 **BiLSTM Limitations**

The BiLSTM trailing both classical baselines is the study's most counter-intuitive result. A wide range of factors were tested and **ruled out** as the cause:
 
- richer word vectors: larger GloVe (100d, 200d, 300d), fastText, and word2vec-google-news

- standard regularization: dropout, weight decay, and learning-rate scheduling

- reducing unknown words: a casing-aware version drove the unknown-word rate far lower with no change in score

Across all of these the score held flat, which means embedding coverage and tuning were not the bottleneck. The binding constraint appears to be representational: a **static, non-contextual** word vector carries a fixed meaning regardless of surrounding words, and that is the most likely limit rather than the recurrent architecture on top of it. That DistilBERT's context-aware embeddings break through the same ceiling supports this interpretation, though its larger size and pretraining make it an imperfect isolation of the cause.


## 🚀 **Future Work**

Several directions could raise performance or sharpen the analysis:
 
- **Redesign the overlapping categories.** Since several categories genuinely overlap, asking the model to pick exactly one is fighting the labels themselves. The categories could be reorganized into a two-level hierarchy (group the `* NEWS` labels together, then split them in a second step), or the most cross-cutting labels like `IMPACT` and `FIFTY` could be allowed to apply alongside a topic rather than instead of it.

- **Build a harder test set.** Our test set is a random sample from the same data as training, so it measures how well the models handle *more of the same*. A test set that balances the categories, or that deliberately includes rare and unusual topics, would give a tougher and more realistic score. We would expect the numbers to drop, but the ranking between models to hold or even widen, since the strongest model does best on exactly these hard cases.

- **Give the BiLSTM more room.** Some options were never tried: stacking more LSTM layers, using attention to summarize the headline instead of just the last state, tuning the optimizer specifically for this model, or swapping the LSTM for a small transformer trained from scratch.


## 📁 **Repository Structure**

```
news-category/
├── 01_preprocessing.ipynb           # load raw data, clean, stratified train/test split
├── 02_taxonomy.ipynb                # merge near-duplicate categories (versioned)
├── 03_naive_bayes.ipynb             # Multinomial Naive Bayes over TF-IDF
├── 04_logistic_regression.ipynb     # Logistic Regression over TF-IDF
├── 05_bilstm.ipynb                  # BiLSTM over trainable GloVe embeddings
├── 06_distilbert.ipynb              # fine-tuned DistilBERT
├── 07_comparison.ipynb              # cross-model comparison
├── src/news_category/               # shared library imported by every notebook
│   ├── data.py                      # split loading, field assembly, config definitions
│   ├── evaluate.py                  # macro-F1 scoring, cross-validation folds
│   ├── plotting.py                  # confusion matrix, per-class F1, history/curve plots
│   └── display.py                   # labeled console output helper
├── data/                            # dataset and derived splits (gitignored)
│   ├── raw/                         # original Kaggle download
│   └── processed/<version>/         # per-taxonomy train/test splits and label maps
├── results/<model>/                 # scores & best params per model (gitignored)
├── checkpoints/distilbert/<config>/ # DistilBERT training checkpoints (gitignored)
├── assets/images/                   # figures embedded in this README
└── README.md
```
