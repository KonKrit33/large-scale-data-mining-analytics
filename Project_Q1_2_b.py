# ============================================================
# M161 Q1.2
# ============================================================

import pandas as pd
import numpy as np
import re
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from collections import Counter
from scipy import sparse


# -------------------------
# 0) Load train set
# -------------------------
TRAIN_PATH = r"C:\Users\Konstantinos Krit\Desktop\MSc\Α' Εξάμηνο\High Scale Analytics\Project 2025-2026\bigdata2025classification\train.xlsx"

df = pd.read_excel(TRAIN_PATH)
y = df["Label"].astype(str).values


# -------------------------
# 1) Preprocessing
# -------------------------
BASE_STOPWORDS = set(ENGLISH_STOP_WORDS)
STOPWORDS = BASE_STOPWORDS | {
    "said","says","also","new","year","week","day","mr","mrs",
    "reuters","cnn","bbc","news","people","company","million",
    "percent","world","according","make","billion","today"
}

def clean_text(s):
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"http\S+|www\.\S+|\S+@\S+|\d+"," ", s)
    s = re.sub(r"[^a-z\s]"," ", s)
    return re.sub(r"\s+"," ", s).strip()

def tokenize_filter(s):
    return " ".join([w for w in s.split() if w not in STOPWORDS and len(w) >= 3])

X_raw = (df["Title"].fillna("") + " " + df["Content"].fillna("")).values
X_text = np.array([tokenize_filter(clean_text(t)) for t in X_raw], dtype=object)


# -------------------------
# 2) CV helper
# -------------------------
def cv_accuracy(X, y, estimator, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs = []
    t0 = time.time()

    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        est = estimator
        est.fit(X[tr], y[tr])
        pred = est.predict(X[va])
        acc = accuracy_score(y[va], pred)
        accs.append(acc)
        print(f"[Fold {fold}] Accuracy = {acc:.5f}")

    dt = time.time() - t0
    print(f"\nMean Accuracy = {np.mean(accs):.5f} | Std = {np.std(accs):.5f} | Time = {dt/60:.1f} min\n")
    return np.array(accs)


# ============================================================
# BASELINE A: BoW + Linear SVM
# ============================================================
svm_bow = Pipeline([
    ("bow", CountVectorizer(
        lowercase=False,
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        ngram_range=(1,2),
        min_df=2,
        max_df=0.95,
        max_features=200_000
    )),
    ("clf", LinearSVC(C=1.0, max_iter=5000, random_state=42))
])

print("\n========================")
print("Baseline: SVM (BoW)")
print("========================")
svm_accs = cv_accuracy(X_text, y, svm_bow)


# ============================================================
# BASELINE B: KNN + Jaccard
# ============================================================
def fit_bow_binary(X_text, max_features=100_000):
    vec = CountVectorizer(
        lowercase=False,
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        binary=True,
        ngram_range=(1,1),
        min_df=2,
        max_df=0.95,
        max_features=max_features
    )
    return vec, vec.fit_transform(X_text).astype(np.bool_)

def transform_bow_binary(vec, X_text):
    return vec.transform(X_text).astype(np.bool_)

def jaccard_knn(X_train_bin, y_train, X_query_bin, k=7):
    maj = Counter(y_train).most_common(1)[0][0]
    train_nnz = np.asarray(X_train_bin.getnnz(axis=1)).reshape(-1)
    preds = []

    inter = (X_query_bin.astype(np.uint8) @ X_train_bin.T.astype(np.uint8)).tocsr()
    q_nnz = np.asarray(X_query_bin.getnnz(axis=1)).reshape(-1)

    for i in range(inter.shape[0]):
        row = inter.getrow(i)
        if row.nnz == 0:
            preds.append(maj)
            continue

        idx = row.indices
        inter_counts = row.data.astype(np.float32)
        union = q_nnz[i] + train_nnz[idx] - inter_counts
        jac = inter_counts / np.maximum(union, 1.0)

        topk = idx[np.argpartition(-jac, min(k, len(jac))-1)[:k]]
        preds.append(Counter(y_train[topk]).most_common(1)[0][0])

    return np.array(preds)

def cv_knn(X_text, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs = []

    for fold, (tr, va) in enumerate(skf.split(X_text, y), 1):
        vec, Xtr = fit_bow_binary(X_text[tr])
        Xva = transform_bow_binary(vec, X_text[va])
        pred = jaccard_knn(Xtr, y[tr], Xva)
        acc = accuracy_score(y[va], pred)
        accs.append(acc)
        print(f"[Fold {fold}] Accuracy = {acc:.5f}")

    print(f"\nMean Accuracy = {np.mean(accs):.5f} | Std = {np.std(accs):.5f}\n")
    return np.array(accs)

print("\n========================")
print("Baseline: KNN (Jaccard)")
print("========================")
knn_accs = cv_knn(X_text, y)


# ============================================================
# MY METHOD: TF-IDF + Optimized Linear SVM
# ============================================================
my_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=False,
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        ngram_range=(1,2),
        min_df=2,
        max_df=0.9,
        max_features=300_000,
        sublinear_tf=True
    )),
    ("clf", LinearSVC(
        C=1.5,
        max_iter=5000,
        random_state=42
    ))
])

print("\n========================")
print("MY METHOD: TF-IDF + Linear SVM")
print("========================")
my_accs = cv_accuracy(X_text, y, my_model)


# ============================================================
# FINAL COMPARISON
# ============================================================
print("\n=========== FINAL RESULTS ===========")
print(f"SVM (BoW):     {svm_accs.mean():.5f}")
print(f"KNN (Jaccard): {knn_accs.mean():.5f}")
print(f"MY METHOD:     {my_accs.mean():.5f}")
print("=====================================")