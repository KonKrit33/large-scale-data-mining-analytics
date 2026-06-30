import pandas as pd
import numpy as np
import re
import time

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


# -------------------------
# 0) Load train.xlsx
# -------------------------
df = pd.read_excel("train.xlsx")
assert set(["Title","Content","Label"]).issubset(df.columns)

y = df["Label"].astype(str).values


# -------------------------
# 1) Preprocessing
# -------------------------
BASE_STOPWORDS = set(ENGLISH_STOP_WORDS)
EXTRA_STOPWORDS = {
    "said","says","say","also","one","two","new","year","years",
    "week","weeks","day","days","time","times",
    "mr","mrs","ms","u","us","uk","eu",
    "reuters","ap","cnn","bbc","news",
    "didn","doesn","isn","aren","wasn","weren","hasn","haven","hadn",
    "won","wouldn","couldn","shouldn","mustn","mightn",
    "im","ive","youre","theyre","weve","youve",
    "like","just","got","get","go","going"
}
STOPWORDS = BASE_STOPWORDS | EXTRA_STOPWORDS | {
    "people","company","million","percent","world","according","make","billion",
    "way","know","good","report","high","think","including","long","best","use",
    "today","month","april","old","really","big","work","end","based","set","want",
    "right","told","did"
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

# Text = Title + Content
X_raw = (df["Title"].fillna("") + " " + df["Content"].fillna("")).values
X_text = np.array([tokenize_filter(clean_text(t)) for t in X_raw], dtype=object)


# -------------------------
# 2) 5-fold CV helper
# -------------------------
def cv_accuracy_sklearn_model(X, y, estimator, n_splits=5, seed=42):
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
    print(f"\nMean Accuracy = {np.mean(accs):.5f}  |  Std = {np.std(accs):.5f}  |  Time = {dt/60:.1f} min")
    return np.array(accs)


# ============================================================
# A) SVM (Linear SVM) + Bag of Words
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
    ("clf", LinearSVC(
        C=1.0,
        class_weight=None,   
        random_state=42
    ))
])

print("\n========================")
print("SVM (BoW) 5-fold CV")
print("========================")
svm_accs = cv_accuracy_sklearn_model(X_text, y, svm_bow, n_splits=5, seed=42)


# ============================================================
# B) KNN with Jaccard + Bag of Words (binary BoW) + 5-fold CV
# ============================================================
from collections import Counter
from scipy import sparse

def fit_bow_binary(X_train_text, max_features=100_000):
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
    Xtr = vec.fit_transform(X_train_text).astype(np.bool_)
    return vec, Xtr

def transform_bow_binary(vec, X_text_):
    return vec.transform(X_text_).astype(np.bool_)

def jaccard_knn_predict_sparse(X_train_bin, y_train, X_query_bin, k=5, chunk_size=512):
    # Majority fallback
    maj = Counter(y_train).most_common(1)[0][0]

    
    train_nnz = np.asarray(X_train_bin.getnnz(axis=1)).reshape(-1)

    preds = []
    n = X_query_bin.shape[0]

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        Xq = X_query_bin[start:end]
        q_nnz = np.asarray(Xq.getnnz(axis=1)).reshape(-1)

        # intersections: sparse (chunk x n_train) with counts of shared terms
        inter = (Xq.astype(np.uint8) @ X_train_bin.T.astype(np.uint8)).tocsr()

        # For each row in chunk, find top-k by jaccard among nonzero intersections
        for i in range(inter.shape[0]):
            row = inter.getrow(i)
            if row.nnz == 0:
                preds.append(maj)
                continue

            idx = row.indices
            inter_counts = row.data.astype(np.float32)

            union = q_nnz[i].astype(np.float32) + train_nnz[idx].astype(np.float32) - inter_counts
            
            union = np.maximum(union, 1.0)
            jac = inter_counts / union

            # top-k among candidates
            if jac.size > k:
                topk_pos = np.argpartition(-jac, k-1)[:k]
            else:
                topk_pos = np.arange(jac.size)

            top_idx = idx[topk_pos]
            top_labels = y_train[top_idx]

            # majority vote
            preds.append(Counter(top_labels).most_common(1)[0][0])

    return np.array(preds, dtype=object)

def cv_knn_jaccard(X_text, y, n_splits=5, seed=42, k=5, max_features=100_000, chunk_size=512):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs = []
    t0 = time.time()

    for fold, (tr, va) in enumerate(skf.split(X_text, y), 1):
        Xtr_text, Xva_text = X_text[tr], X_text[va]
        ytr, yva = y[tr], y[va]

        vec, Xtr_bin = fit_bow_binary(Xtr_text, max_features=max_features)
        Xva_bin = transform_bow_binary(vec, Xva_text)

        pred = jaccard_knn_predict_sparse(Xtr_bin, ytr, Xva_bin, k=k, chunk_size=chunk_size)
        acc = accuracy_score(yva, pred)
        accs.append(acc)
        print(f"[Fold {fold}] Accuracy = {acc:.5f}")

    dt = time.time() - t0
    print(f"\nMean Accuracy = {np.mean(accs):.5f}  |  Std = {np.std(accs):.5f}  |  Time = {dt/60:.1f} min")
    return np.array(accs)

print("\n========================")
print("KNN (Jaccard) + BoW(binary) 5-fold CV")
print("========================")
knn_accs = cv_knn_jaccard(X_text, y, n_splits=5, seed=42, k=5, max_features=100_000, chunk_size=512)

# -------------------------
# C) Report-friendly summary
# -------------------------
print("\n\n=========== MODEL PERFORMANCE SUMMARY ===========")
print(f"SVM (BoW):             mean={svm_accs.mean():.5f}  std={svm_accs.std():.5f}")
print(f"KNN (Jaccard + BoW):   mean={knn_accs.mean():.5f}  std={knn_accs.std():.5f}")
print("==================================================")