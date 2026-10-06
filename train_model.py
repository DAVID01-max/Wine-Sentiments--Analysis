"""
Train the model and save it, WITHOUT needing the notebook.

Run this once, from the same folder as app.py:
    python train_model.py

It creates:  artifacts/best_model.joblib   (the file app.py needs)

Data used:
    - winemag-data-130k-v2.csv  if it is in this folder (the real Kaggle data)
    - otherwise a small fake dataset, just so you can test the deployment
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "winemag-data-130k-v2.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "artifacts")
SAMPLE_SIZE = 4000     # same as the notebook
SEED = 42


def make_fake_data(n):
    """Tiny made-up dataset so the script works with no CSV."""
    rng = np.random.default_rng(SEED)
    good = ["a delightful wine with bright fruit and a smooth finish",
            "elegant, well balanced, and full of ripe cherry flavor",
            "an outstanding vintage with silky tannins and great depth",
            "crisp acidity and a long, satisfying aftertaste",
            "rich aromas of blackberry and a velvety texture"]
    bad = ["a thin, watery wine with little character",
           "harsh tannins and an unpleasant bitter finish",
           "flat and lifeless, lacking any real fruit expression",
           "overly acidic with a sour, off-putting aroma",
           "disappointing and unbalanced, tastes almost like vinegar"]
    is_good = rng.random(n) > 0.45
    text = [rng.choice(good if g else bad) for g in is_good]
    points = np.clip(rng.normal(np.where(is_good, 90, 82), 3), 80, 100).astype(int)
    return pd.DataFrame({"description": text, "points": points})


# 1. Load data --------------------------------------------------------------
if os.path.exists(CSV_FILE):
    df = pd.read_csv(CSV_FILE)[["description", "points"]].dropna()
    df = df.sample(min(SAMPLE_SIZE, len(df)), random_state=SEED)
    print(f"Using REAL data: {len(df)} reviews")
else:
    df = make_fake_data(SAMPLE_SIZE)
    print("CSV not found -> using FAKE data (fine for testing the API only)")

# 2. Make the label: 1 = points at/above the median, 0 = below ---------------
df["label"] = (df["points"] >= df["points"].median()).astype(int)

# 3. Split, train, evaluate ---------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["description"], df["label"], test_size=0.2, random_state=SEED, stratify=df["label"]
)
model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=3000, stop_words="english")),
    ("clf", LogisticRegression(max_iter=300, random_state=SEED)),
])
model.fit(X_train, y_train)
preds = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, preds):.3f} | F1: {f1_score(y_test, preds):.3f}")

# 4. Save -------------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
path = os.path.join(OUTPUT_DIR, "best_model.joblib")
joblib.dump(model, path)
print(f"Saved model to: {path}")
