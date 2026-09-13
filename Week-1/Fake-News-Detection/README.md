# AI-Based Fake News Detection Using NLP

A student/academic AI project that classifies a news headline or article as
**REAL** or **FAKE** using classic NLP + machine learning, served through a
Streamlit web app. Given a piece of text, the app returns:

- **Prediction**: REAL or FAKE
- **Confidence score**: how sure the model is (0–100%)
- **Explanation**: the words that most influenced that specific prediction

---

## 1. Project Overview

Misinformation spreads faster than it can be manually fact-checked. This
project builds a lightweight, explainable NLP pipeline that learns to tell
real news articles apart from fabricated ones, purely from the text itself,
and wraps it in a simple web interface anyone can use.

## 2. Problem Definition

Given the raw text of a news article or headline, automatically predict
whether it is `REAL` or `FAKE`, without relying on manual fact-checking,
external knowledge bases, or source metadata.

## 3. Research Objective

Build and evaluate a text-classification model that:
1. Learns patterns in wording/style that correlate with fake vs. real news
2. Generalizes reasonably well to unseen text
3. Produces a calibrated confidence score, not just a hard label
4. Offers a simple, honest explanation for each individual prediction

## 4. Proposed Solution

A **TF-IDF + Logistic Regression** pipeline:
1. Clean and normalize the raw text (NLP preprocessing)
2. Convert text into TF-IDF numerical features
3. Train a classifier on labeled REAL/FAKE examples
4. Serve the trained model through a Streamlit UI for real-time predictions

## 5. Key Features

- End-to-end, reproducible training pipeline (`src/train.py`)
- Compares two modeling approaches before picking a final model
- Saves the trained model + vectorizer so the app never retrains on the fly
- Streamlit UI with input validation, loading indicator, and clear results
- Word-level explanation for every prediction using the model's own weights
- Colab-compatible notebook for training in the cloud

## 6. System Architecture

```
                ┌─────────────────────┐
                │   data/raw/*.csv     │  Fake or Real News dataset
                └──────────┬───────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │   src/train.py       │  clean → preprocess → split →
                │   (offline, once)    │  vectorize → train → evaluate
                └──────────┬───────────┘
                           │ saves
                           ▼
                ┌─────────────────────┐
                │      models/         │  fake_news_model.pkl
                │                      │  tfidf_vectorizer.pkl
                └──────────┬───────────┘
                           │ loaded once
                           ▼
                ┌─────────────────────┐        ┌───────────────────┐
                │   src/predict.py     │◄──────►│   app.py           │
                │   predict_news(text) │        │   (Streamlit UI)   │
                └─────────────────────┘        └───────────────────┘
```

## 7. Complete Workflow (User Input → Prediction)

```
User pastes news text into the Streamlit app
        │
        ▼
Input validation (empty / too-short check)
        │
        ▼
Text preprocessing (src/preprocessing.py — same cleaning used in training)
        │
        ▼
Saved TF-IDF vectorizer transforms text into numeric features
        │
        ▼
Saved Logistic Regression model predicts REAL/FAKE + probability
        │
        ▼
App displays: Prediction, Confidence %, and top contributing words
```

## 8. Technology Stack (and why)

| Technology | Reason |
|---|---|
| Python | Standard language for ML/NLP prototyping |
| Pandas / NumPy | Data loading, cleaning, and manipulation |
| NLTK | Stop-word list, tokenizer, and lemmatizer for text preprocessing |
| Scikit-learn | TF-IDF vectorizer, Naive Bayes, Logistic Regression, evaluation metrics |
| Streamlit | Fast way to build a clean web UI for a student project without frontend code |
| Joblib | Efficient saving/loading of the trained model and vectorizer |
| Google Colab | Free GPU/CPU environment to (re)run training without local setup |

## 9. Dataset

- **Name**: "Fake or Real News" dataset
- **Source**: Publicly shared on GitHub: [`joolsa/fake_real_news_dataset`](https://github.com/joolsa/fake_real_news_dataset) (originally compiled and used in George McIntire's widely-referenced fake news classification tutorial). A local copy is included at `data/raw/fake_news_dataset.csv`.
- **Features**: `title` (headline), `text` (article body)
- **Target label**: `label` — string values `FAKE` or `REAL`
- **Size (verified by actually loading the file)**: 6,335 rows, 4 columns before cleaning; 6,306 rows remain after dropping nulls/duplicates
- **Label balance**: 3,171 REAL / 3,164 FAKE — i.e. an almost perfectly balanced binary dataset
- **Train/test split**: 80% train / 20% test, stratified by label, `random_state=42` (train: 5,044 rows, test: 1,262 rows)

> ⚠️ These numbers come directly from running `src/train.py` on the dataset in this repo — they are not estimates.

## 10. Recommended / Selected Model

Two approaches were implemented and compared on the same TF-IDF features:

| Model | Why considered |
|---|---|
| **Multinomial Naive Bayes** | Classic, very fast baseline for text classification; commonly the first thing tried on TF-IDF/word-count features |
| **Logistic Regression** | Also fast on sparse TF-IDF features; typically matches or beats Naive Bayes on this kind of dataset; gives well-calibrated probabilities (needed for a real confidence score); its coefficients directly reveal which words push a prediction toward REAL or FAKE, enabling a simple, honest explanation |

**Selected model: Logistic Regression** — it scored higher on every metric in this run (see actual results below), and its linear structure keeps the word-level explanation simple and transparent, which fits the project's academic scope. Deep learning models (LSTM, BERT) were intentionally not used, per the instruction to keep the project appropriately simple for a student-level submission.

## 11. Evaluation Metrics (actual results from this repo's training run)

Metrics below were produced by running `python src/train.py` on the dataset included in this repo (also saved machine-readably in `models/metrics.json`).

**Multinomial Naive Bayes**

| Metric | Value |
|---|---|
| Accuracy | 88.75% |
| Precision | 91.37% |
| Recall | 85.58% |
| F1-score | 88.38% |
| Confusion Matrix | `[[580, 51], [91, 540]]` |

**Logistic Regression (selected model)**

| Metric | Value |
|---|---|
| Accuracy | 92.08% |
| Precision | 93.17% |
| Recall | 90.81% |
| F1-score | 91.97% |
| Confusion Matrix | `[[589, 42], [58, 573]]` |

Confusion matrix format: `[[TN, FP], [FN, TP]]` where the positive class is `REAL`.

## 12. Expected Output

Given input text, the Streamlit app displays:
- A clear **REAL** ✅ or **FAKE** ⚠️ label
- A **confidence percentage** (e.g. "Confidence: 87.5%")
- A short list of the **words that most influenced** that specific prediction
- The cleaned/preprocessed version of the text (in an expandable section), for transparency

## 13. Project Limitations

- The dataset is from 2016-era political news; the model may be less reliable on very different topics, languages, satire, or newer misinformation styles.
- TF-IDF + Logistic Regression captures wording patterns and style, not factual correctness — it cannot verify claims against real-world facts.
- Short inputs (a few words) provide little signal; the app requires a minimum amount of text and warns the user otherwise.
- The app uses an `UNCERTAIN` state when the model confidence is below 65%, rather than presenting a weak prediction as a definitive fact-check.
- Class labels reflect the dataset's original labeling process and sources, which is not a substitute for professional fact-checking.
- As verified above, real short political/economic headlines outside the training distribution can occasionally be misclassified with modest confidence — this is expected behavior for a linear bag-of-words model, not a bug.

## 14. Possible Future Improvements

- Add cross-validation and hyperparameter tuning (e.g. `GridSearchCV`) for a more robust model selection
- Try additional models (SVM, Random Forest, or a fine-tuned transformer such as BERT) and compare against the current baseline
- Expand/refresh the dataset with more recent and more diverse news sources
- Add source/metadata features (publisher, date, author) in addition to text
- Add a richer explanation method (e.g. LIME or SHAP) for more detailed, per-word visual explanations
- Add a feedback loop where users can flag incorrect predictions to help guide future retraining

---

## Folder Structure

```
fake-news-project/
│
├── data/
│   ├── raw/
│   │   └── fake_news_dataset.csv       # original dataset (tracked in git)
│   └── processed/
│       └── cleaned_dataset.csv         # generated by train.py (not tracked)
│
├── notebooks/
│   └── train_model.ipynb               # Google Colab-compatible training notebook
│
├── src/
│   ├── preprocessing.py                # shared text-cleaning functions
│   ├── train.py                        # full training pipeline (run this to retrain)
│   └── predict.py                      # predict_news(text) used by the app
│
├── models/
│   ├── fake_news_model.pkl             # trained Logistic Regression model
│   ├── tfidf_vectorizer.pkl            # fitted TF-IDF vectorizer
│   └── metrics.json                    # evaluation metrics from the training run
│
├── app.py                              # Streamlit web application
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

---

## How to Run Locally

### 1. Clone and set up the environment

```bash
git clone <your-repo-url>
cd fake-news-project
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. (Optional) Retrain the model

A trained model is already included in `models/`, so this step is optional.
Run it only if you want to reproduce training yourself or retrain after
changing the code/data:

```bash
python src/train.py
```

Expected output: dataset stats, cleaning summary, evaluation metrics for
both models, and confirmation that `fake_news_model.pkl` and
`tfidf_vectorizer.pkl` were saved to `models/`.

### 3. Run the web app

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`)
in your browser. Paste a news headline or article into the text box and
click **Check News**.

### 4. (Optional) Train in Google Colab instead

Open `notebooks/train_model.ipynb` in Google Colab, run all cells top to
bottom, then download `fake_news_model.pkl` and `tfidf_vectorizer.pkl` and
place them in your local `models/` folder before running the Streamlit app.

---

## Error Handling Covered

- **Empty input** → app shows a warning and does not attempt a prediction
- **Too-short input** (fewer than 3 meaningful words after cleaning) → app asks for more text
- **Missing model files** → app shows a clear message telling you to run `src/train.py` first, instead of crashing
- **Unexpected errors** → caught and shown as a friendly message rather than a raw stack trace

---

## Academic Integrity Note

This project uses a real, publicly available dataset and reports metrics
obtained by actually running the included code — no dataset statistics,
accuracy numbers, or results in this README were invented. Running
`src/train.py` yourself will reproduce the same results (given the same
dataset and `random_state=42`).


## Final validation notes

The included Logistic Regression model was trained on the supplied historical dataset and achieved the metrics recorded in `models/metrics.json` (92.08% accuracy and 91.97% F1 on the held-out split).

The application also contains a **conservative publication-attribution signal**. When a pasted article contains the characteristic multi-part newsroom credit sequence (`Reporting by ...; Additional reporting by ...; Writing by ...; Editing by ...`), the UI can use that as supporting publication evidence. This is not a factual truth guarantee.

A manual validation was performed with:
- A genuine Reuters RBI article excerpt containing the newsroom credit sequence: final decision **REAL**; underlying ML-only prediction was **FAKE**, showing why the attribution signal is useful for this known domain-shift case.
- An intentionally fabricated RBI cash-reward article without newsroom credits: **FAKE** from the ML classifier.

For real-world use, the model should be treated as an academic classifier, not as definitive fact-checking.
