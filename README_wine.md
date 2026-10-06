# 🍷 Wine Review Sentiment Analysis

Can a computer tell if a wine is **highly rated** just by reading the review?

This project reads wine reviews written by critics and tries to predict whether the wine scored high or low. It compares three approaches, checks where the models make mistakes, and saves the best one so it can be used in a small web service.

---

## What's in this project?

| File | What it is |
|------|------------|
| `wine.ipynb` | The main notebook. Everything happens here. |
| `winemag-data-130k-v2.csv` | The data from Kaggle (130k wine reviews). Put it next to the notebook. |
| `artifacts/` | Created when you run the notebook. Holds the saved model, model card, experiment log and `app.py`. |

> No dataset? No problem. If the CSV is missing, the notebook creates a small fake dataset so it still runs from start to end. The fake data is only for testing, so the results won't mean anything.

---

## The big idea

Each review has a **points** score (80 to 100). We turn that into a simple yes/no label:

- **High quality (1)**: points at or above the median
- **Low quality (0)**: points below the median

Then we ask: *can the review text alone predict that label?*

Note that this is a **proxy**. The model learns "does this review sound like a highly rated wine", not "is this wine objectively good".

---

## What the notebook does (step by step)

1. **Load and clean the data**: keep the useful columns, drop missing values, and sample 4,000 reviews.
2. **Create the label**: high or low quality, based on the median points.
3. **Explore**: charts of class balance, points, price and countries.
4. **Clean the text**: lowercase, remove punctuation and common words.
5. **Train three models** on the same train/test split (80/20):
   - TF-IDF + Logistic Regression
   - TF-IDF + Naive Bayes
   - DistilBERT (a pretrained transformer)
6. **Compare them**: accuracy, precision, recall, F1 and confusion matrices.
7. **Error analysis**: look at reviews the best model got wrong.
8. **Aspect-based sentiment**: which parts of a wine (aroma, finish, tannins, acidity...) do reviewers like or dislike?
9. **Explainability**: which words push the model toward "high" or "low".
10. **Fairness check**: does the model work equally well across countries?
11. **Fine-tuning option**: code to fine-tune DistilBERT (off by default, needs a GPU).
12. **Save the best model** plus a `model_card.json` describing it.
13. **Log the experiment** to `artifacts/experiments.csv`.
14. **Write a small Flask app** (`artifacts/app.py`) that serves predictions.
15. **Run unit tests** on the core helper functions.

---

## How to run it

**1. Install what you need**

```bash
pip install pandas numpy matplotlib scikit-learn joblib flask
```

For the transformer model (optional):

```bash
pip install transformers torch accelerate datasets
```

If these aren't installed, the notebook falls back to a simple word-counting scorer so it still runs.

**2. Add the dataset** (optional) by downloading `winemag-data-130k-v2.csv` from Kaggle and placing it next to the notebook.

**3. Run the notebook**

```bash
jupyter notebook wine.ipynb
```

Run the cells from top to bottom.

---

## Use the saved model

After running the notebook, start the web service:

```bash
python artifacts/app.py
```

Then send it a review:

```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"description": "Rich aromas of blackberry with silky tannins and a long finish."}'
```

The port above is Flask's default. If the script uses a different one, change it in the command.

---

## Results

Tested on 800 reviews:

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| **TF-IDF + Logistic Regression** ✅ | **0.77** | 0.76 | 0.91 | **0.83** |
| TF-IDF + Naive Bayes | 0.73 | 0.71 | 0.93 | 0.81 |
| DistilBERT (no fine-tuning) | 0.65 | 0.63 | 0.99 | 0.77 |

**Winner: Logistic Regression.** It gets about 77% of reviews right, and was saved as the best model.

The simple model beat the transformer here. The likely reason is that this transformer was trained to detect general positive/negative feelings, not wine quality. Fine-tuning it on wine data may help.

---

## Things to know

- **The label is a proxy.** Sentiment in a review is not the same thing as wine quality.
- **Accuracy isn't perfect.** About 23% of test reviews were misclassified. Wine reviews are descriptive, and a "positive-sounding" review can still get a lower score.
- **Fairness results are uneven.** Performance varies by country (for example, Austria scored higher than Spain), and several countries had too few reviews to check at all. Treat per-country numbers for small groups with caution.
- **The models only see 4,000 of 130,000 reviews.** Increase `sample_size` in `CONFIG` to use more data.
- **The transformer isn't fine-tuned.** Fine-tuning is included but switched off (`RUN_TRAINING = False`).
- **Not production-ready yet.** The Flask app is a minimal demo.

---




## Tools used

Python · pandas · scikit-learn · Hugging Face Transformers (DistilBERT) · Flask · joblib · matplotlib
