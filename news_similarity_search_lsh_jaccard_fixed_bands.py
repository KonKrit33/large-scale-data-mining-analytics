"""
News Similarity Search with MinHash LSH and Jaccard Similarity

This script compares brute-force Jaccard nearest-neighbor search with MinHash LSH
for scalable similarity search over news article text.
"""

import pandas as pd
import numpy as np
import re
import time
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from collections import Counter
from scipy import sparse
from datasketch import MinHash, MinHashLSH

# ============================================================
# 0) LOAD DATA
# ============================================================

train = pd.read_excel("train.xlsx")
test = pd.read_excel("test_without_labels.xlsx")

y_train = train["Label"].astype(str).values

# ============================================================
# 1) Text preprocessing
# ============================================================

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
    return " ".join([w for w in s.split() if w not in STOPWORDS and len(w)>=3])

X_train_text = np.array([
    tokenize_filter(clean_text(t))
    for t in (train["Title"].fillna("") + " " + train["Content"].fillna(""))
])

X_test_text = np.array([
    tokenize_filter(clean_text(t))
    for t in (test["Title"].fillna("") + " " + test["Content"].fillna(""))
])

# ============================================================
# 2) BINARY BoW FOR JACCARD
# ============================================================

vec = CountVectorizer(
    lowercase=False,
    tokenizer=str.split,
    preprocessor=None,
    token_pattern=None,
    binary=True,
    min_df=2,
    max_df=0.95,
    max_features=100_000
)

X_train_bin = vec.fit_transform(X_train_text).astype(np.bool_)
X_test_bin  = vec.transform(X_test_text).astype(np.bool_)
X_train_u8 = X_train_bin.astype(np.uint8)
X_test_u8  = X_test_bin.astype(np.uint8)

train_nnz = np.asarray(X_train_bin.getnnz(axis=1)).reshape(-1)

# ============================================================
# 3) BRUTE FORCE KNN (K=7)
# ============================================================

def brute_force_knn(Xtr, ytr, Xq, k=7):
    maj = Counter(ytr).most_common(1)[0][0]
    preds = []
    topk_indices_all = []

    inter = (Xq @ Xtr.T).tocsr()
    q_nnz = np.asarray(Xq.getnnz(axis=1)).reshape(-1)

    for i in range(inter.shape[0]):
        row = inter.getrow(i)
        if row.nnz == 0:
            preds.append(maj)
            topk_indices_all.append([])
            continue

        idx = row.indices
        inter_counts = row.data.astype(np.float32)
        union = q_nnz[i] + train_nnz[idx] - inter_counts
        jac = inter_counts / np.maximum(union,1)

        topk = idx[np.argpartition(-jac, min(k,len(jac))-1)[:k]]
        preds.append(Counter(ytr[topk]).most_common(1)[0][0])
        topk_indices_all.append(set(topk))

    return np.array(preds), topk_indices_all

print("Running Brute Force...")
t0 = time.time()
bf_preds, bf_topk = brute_force_knn(X_train_u8, y_train, X_test_u8, k=7)
bf_query_time = time.time() - t0

print("Brute Force QueryTime:", bf_query_time)

def build_lsh(X_text, num_perm=32):

    rows = 2 
    bands = num_perm // rows

    lsh = MinHashLSH(num_perm=num_perm, params=(bands, rows))

    t0 = time.time()

    for i, doc in enumerate(X_text):
        mh = MinHash(num_perm=num_perm)
        tokens = set(doc.split())  
        mh.update_batch([t.encode("utf8") for t in tokens])       
        lsh.insert(str(i), mh)
        
    build_time = time.time() - t0
    return lsh, build_time

def lsh_knn(lsh, X_test_text, y_train, bf_topk, num_perm=32, k=7):

    maj = Counter(y_train).most_common(1)[0][0]
    preds = []
    total_hits = 0
    total_possible = 0

    t0 = time.time()

    for i, doc in enumerate(X_test_text):

        mh = MinHash(num_perm=num_perm)
        tokens = set(doc.split())
        mh.update_batch([t.encode("utf8") for t in tokens])
        candidates = lsh.query(mh)

        if len(candidates)==0:
            preds.append(maj)
            total_possible += k
            continue

        cand_idx = np.array([int(c) for c in candidates])

        # exact Jaccard only on candidates
        q_vec = X_test_u8[i]
        inter = (q_vec @ X_train_u8[cand_idx].T).toarray().ravel()       
        q_nnz = q_vec.getnnz()
        union = q_nnz + train_nnz[cand_idx] - inter
        jac = inter / np.maximum(union,1)
        if len(cand_idx) <= k:
            topk_local = cand_idx
        else:
            topk_local = cand_idx[np.argpartition(-jac, k-1)[:k]]
        
        preds.append(Counter(y_train[topk_local]).most_common(1)[0][0])

        # recall
        total_hits += len(set(topk_local) & bf_topk[i])
        total_possible += k

    query_time = time.time() - t0
    recall = total_hits / total_possible

    return np.array(preds), query_time, recall

results = []

for perm in [16,32,64]:

    print(f"\n===== LSH num_perm={perm} =====")

    lsh, build_time = build_lsh(X_train_text, num_perm=perm)

    preds, query_time, recall = lsh_knn(
        lsh, X_test_text, y_train,
        bf_topk,
        num_perm=perm
    )

    total_time = build_time + query_time

    results.append({
        "Type": f"LSH-Jaccard (Perm={perm})",
        "BuildTime": build_time,
        "QueryTime": query_time,
        "TotalTime": total_time,
        "Recall": recall
    })

# Brute row
results.insert(0,{
    "Type":"Brute-Force-Jaccard",
    "BuildTime":0,
    "QueryTime":bf_query_time,
    "TotalTime":bf_query_time,
    "Recall":1.0
})

results_df = pd.DataFrame(results)
print(results_df)