# Hope Speech Detection — Multilingual NLP Pipeline

**English · Hindi · Bulgarian · Spanish**

End-to-end NLP pipeline for **hope-speech detection** across four languages. Each notebook cleans the data, engineers features (BoW / TF-IDF / sentence embeddings), trains a large suite of classical machine-learning classifiers **and** deep-learning models (CNN, LSTM, Bi-LSTM, GRU, DNN…), and reports full `classification_report` metrics.

> The pipeline is **identical across languages** — the only differences are the data files, label names, and tokenisation needs of each language.

---

## Table of Contents

1. [What Is "Hope Speech"?](#what-is-"hope-speech"?)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Notebook Pipeline](#notebook-pipeline)
5. [Models Evaluated](#models-evaluated)
6. [Results & Observations](#results--observations)
7. [Requirements](#requirements)
8. [How to Run](#how-to-run)
9. [Acknowledgements](#acknowledgements)
10. [License](#license)

---

## What Is "Hope Speech"?

"Hope speech" is the positive counterpart to "hate speech". It refers to content that expresses **encouragement, support, resilience, optimism, or solidarity** — language that uplifts individuals or communities, especially marginalised groups. The binary classification task labels each post as either **hope speech** (positive / uplifting) or **non-hope speech** (negative / hostile / neutral). See the per-language label mapping in the [Dataset](#dataset) section.

---

## Dataset

Source: [Hope Speech Detection (Kaggle)](https://www.kaggle.com/datasets/xtreammachinelearner/hope-speech-detection) · mirrored locally in the `dataset/` folder.

All files live in `dataset/`. Delimiter is `,` (CSV) or `\t` (TSV).

| Language   | Train                | Dev                  | Test (labels)                          | Positive label | Negative label |
|------------|----------------------|----------------------|----------------------------------------|----------------|----------------|
| English    | `hope_eng_train.csv` | `hope_eng_dev.csv`   | `hope_eng_test_without_labels.csv`    | `Hope_speech`   | `Non_hope_speech` |
| Hindi      | `hindi_hope_train.csv` | `hindi_hope_dev.csv` | — (none provided)                      | `Hope`          | `Not-Hope`     |
| Bulgarian  | `Hope_Bulgarian_train.tsv` | `Hope_Bulgarian_dev.tsv` | `Hope_Bulgarian_test_without_labels.tsv` | `True`  | `False`        |
| Spanish    | `Hope_spanish_train.csv` | `Hope_spanish_dev.csv` | `hope_spanish_test_with_labels.csv`   | `HS`            | `NHS`          |

| Split         | Train | Dev  | Test (labeled) |
|-----------------------|-------|------|----------------|
| **English**  | 18,192 | 4,548 | — (4,805 unlabeled) |
| **Hindi**    | 2,562  | 320   | — |
| **Bulgarian**| 4,671  | 589   | — (599 unlabeled) |
| **Spanish**  | 1,312  | 300   | 547 (labeled) |

> **Label note:** `HS`/`NHS` in the Spanish files map to *hope / non-hope* within this collection (consistent with the other languages). If your source uses `HS` = "Hate Speech", invert the mapping before evaluation.


---

## Project Structure

```
hope-speech-detection-multilingual/
│
├── notebooks/
│   ├── Hope_detection_english_(1).ipynb      ← English: full pipeline
│   ├── Hope_detection_hindi_(1).ipynb        ← Hindi: full pipeline
│   ├── Hope_detection_spanish_(2).ipynb       ← Spanish: full pipeline (+ DNN, GRU)
│   └── Copy_of_Hope_detection_bulgarian.ipynb  ← Bulgarian: full pipeline (+ Explainable AI)
│
├── dataset/                                   ← raw train/dev/test data (4 languages)
│   ├── hope_eng_*.csv
│   ├── hindi_hope_*.csv
│   ├── Hope_Bulgarian_*.tsv
│   └── Hope_spanish_*.csv
└── README.md        
```

---

## Notebook Pipeline

Every notebook follows the same modular pipeline:

1. **Install & import libraries**
   `sentence-transformers`, `tensorflow`/`keras`, `sklearn`, `nltk`, `matplotlib`, `pandas`, `numpy`.

2. **Load data** — reads the corresponding `train.csv` / `dev.csv` (and `test_with_labels.csv` where available), using `index_col=0` to drop the unnamed index column.

3. **Oversampling** — `RandomOverSampler` (handles class imbalance) + **CBOW** embeddings (`sentence-transformers`).

4. **Undersampling** — `RandomUnderSampler` + **CBOW** embeddings.

5. **Preprocessing & cleaning** — lowercasing, punctuation/stopword removal, tokenisation, stemming/lemmatisation (`nltk`).

6. **Feature engineering** —
   - **Bag-of-Words (BoW)** vectorisation, and
   - **TF-IDF** vectorisation.

7. **Label conversion** — map string labels → numeric form for model training.

8. **Model training & evaluation** — train each model on the train split, evaluate on the dev split, and print a full `classification_report` (precision / recall / f1 / support) + `accuracy_score`.

9. **Deep-learning** — sequential `keras` models (`Sequential`) built with `Embedding`/`Tokenizer` layers.

> The **Bulgarian** notebook additionally includes an **Explainable AI (XAI)** section. The **Spanish** notebook extends the deep-learning section with a **DNN**, **DNN + Embeddings**, and a **Gated Recurrence Unit (GRU)**.

---

## Models Evaluated

### Classical machine-learning classifiers (all four notebooks)
`Random Forest`, `Logistic Regression`, `Linear SVC`, `Multinomial Naive Bayes`, `XGBoost`, `AdaBoost`, `Decision Tree`, `Gradient Boosting`, `SVM`, `Ridge Classifier`, `Perceptron`, `SGDClassifier`, and `PassiveAggressiveClassifier`.

### Deep-learning models
| Model | English | Hindi | Bulgarian | Spanish |
|-------|:-------:|:-----:|:---------:|:-------:|
| CNN              | ✅ | ✅ | ✅ | ✅ |
| LSTM             | ✅ | ✅ | ✅ | ✅ |
| Bi-LSTM          | ✅ | ✅ | ✅ | ✅ |
| Simple RNN       | ✅ | ✅ | —  | ✅ |
| DNN              | —  | —  | —  | ✅ |
| DNN + Embeddings | —  | —  | —  | ✅ |
| GRU              | —  | —  | —  | ✅ |
| Explainable AI (XAI) | — | — | ✅ | — |

---

## Results & Observations

- **Overfitting is prominent.** Training accuracy reaches high values (≈ 0.95–0.97), while validation accuracy on the dev split is markedly lower (often 0.3–0.6). This is expected given the heavy **class imbalance** (non-hope samples dominate the vast majority of splits, e.g. English train = 16,630 non-hope vs. 1,562 hope).
- **Imbalance handling** (oversampling / undersampling + CBOW) is applied specifically to mitigate this skew.
- **Per-language difficulty varies**: languages with more balanced / lower-resource data (e.g. Spanish dev/test) tend to show different val-accuracy profiles than the large, imbalanced English set.
- The included `classification_report` outputs let you compare **precision / recall / f1 per class** — pay attention to the minority (hope) class metrics.

---


## How to Run

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download the data** — place the contents of `dataset/` next to the notebooks (or update the file paths in the first code cell of each notebook).

3. **Run a notebook** — e.g. in Google Colab (recommended, for GPU):
   - Upload the notebook + the relevant `dataset/` files to Colab, or mount Google Drive.
   - Run all cells top-to-bottom.

   Or locally:
   ```bash
   jupyter notebook Hope_detection_english_(1).ipynb
   ```

4. **Outputs** — each notebook prints `classification_report` and `accuracy_score` for every model on the dev split.

---

## Acknowledgements

- The original collectors and annotators of the **Hope Speech** datasets for each language.
- The [Kaggle dataset "Hope Speech Detection"](https://www.kaggle.com/datasets/xtreammachinelearner/hope-speech-detection) that hosts the source data.
- `sentence-transformers`, `tensorflow`, `scikit-learn`, `nltk`, and `xgboost` communities.

---

## License

The code in this repository is provided for research and educational purposes. The underlying dataset licensing should be respected per the original Kaggle source. If you reuse or extend this work, please attribute the original Hope Speech dataset source and link back to this repository.
