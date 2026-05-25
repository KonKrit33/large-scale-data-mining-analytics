import pandas as pd
import numpy as np
import re
import time
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from collections import Counter
from datasketch import MinHash, MinHashLSH

# ============================================================
# 0) LOAD DATA
# ============================================================

train = pd.read_excel(r"C:\Users\Konstantinos Krit\Desktop\MSc\Α' Εξάμηνο\High Scale Analytics\Project 2025-2026\bigdata2025classification\train.xlsx")
test  = pd.read_excel(r"C:\Users\Konstantinos Krit\Desktop\MSc\Α' Εξάμηνο\High Scale Analytics\Project 2025-2026\bigdata2025classification\test_without_labels.xlsx")

y_train = train["Label"].astype(str).values

# ============================================================
# 1) SAME PREPROCESSING AS Q1
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
    return " ".join([w for w in s.split() if w not in STOPWORDS and len(w) >= 3])

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
            topk_indices_all.append(set())
            continue

        idx = row.indices
        inter_counts = row.data.astype(np.float32)
        union = q_nnz[i] + train_nnz[idx] - inter_counts
        jac = inter_counts / np.maximum(union, 1)

        kk = min(k, len(jac))
        topk = idx[np.argpartition(-jac, kk - 1)[:kk]]
        preds.append(Counter(ytr[topk]).most_common(1)[0][0])
        topk_indices_all.append(set(topk))

    return np.array(preds), topk_indices_all

print("Running Brute Force...")
t0 = time.time()
bf_preds, bf_topk = brute_force_knn(X_train_u8, y_train, X_test_u8, k=7)
bf_query_time = time.time() - t0
print("Brute Force QueryTime:", bf_query_time)

# ============================================================
# 4) LSH with explicit threshold (tau=0.9)
#    datasketch chooses (b,r) internally based on num_perm & threshold
# ============================================================

def _closest_params_for_tau(num_perm, tau):
    """
    Find feasible (b,r) with:
      - b*r == num_perm
      - b >= 2
    whose knee s0 ≈ (1/b)^(1/r) is closest to tau.
    """
    best = None
    best_diff = float("inf")

    for r in range(1, num_perm + 1):
        if num_perm % r != 0:
            continue
        b = num_perm // r
        if b < 2:
            continue
        s0 = (1.0 / b) ** (1.0 / r)
        diff = abs(s0 - tau)
        if diff < best_diff:
            best_diff = diff
            best = (b, r, s0)

    if best is None:
        raise ValueError(f"No feasible (b,r) for num_perm={num_perm} with b>=2.")
    return best  
def build_lsh(X_text, num_perm=32, tau=0.9):
    """
    1) Try exact teacher-style: MinHashLSH(threshold=tau)
    2) If datasketch fails (b<2), fallback to closest feasible (b,r) with b*r=num_perm.
    """
    try:
        lsh = MinHashLSH(num_perm=num_perm, threshold=tau)
        mode = "threshold"
        chosen = (lsh.b, lsh.r, None)
    except ValueError as e:
        if "bands are too small" not in str(e):
            raise
        b, r, s0 = _closest_params_for_tau(num_perm=num_perm, tau=tau)
        lsh = MinHashLSH(num_perm=num_perm, params=(b, r))
        mode = "fallback_params"
        chosen = (b, r, s0)

    t0 = time.time()
    for i, doc in enumerate(X_text):
        mh = MinHash(num_perm=num_perm)
        tokens = set(doc.split())
        mh.update_batch([t.encode("utf8") for t in tokens])
        lsh.insert(str(i), mh)

    build_time = time.time() - t0
    return lsh, build_time, mode, chosen

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

        if len(candidates) == 0:
            preds.append(maj)
            total_possible += k
            continue

        cand_idx = np.array([int(c) for c in candidates])

        # exact Jaccard only on candidates (for ranking top-k)
        q_vec = X_test_u8[i]
        inter = (q_vec @ X_train_u8[cand_idx].T).toarray().ravel()
        q_nnz = q_vec.getnnz()
        union = q_nnz + train_nnz[cand_idx] - inter
        jac = inter / np.maximum(union, 1)

        if cand_idx.size <= k:
            topk_local = cand_idx
        else:
            topk_local = cand_idx[np.argpartition(-jac, k - 1)[:k]]

        preds.append(Counter(y_train[topk_local]).most_common(1)[0][0])

        total_hits += len(set(topk_local) & bf_topk[i])
        total_possible += k

    query_time = time.time() - t0
    recall = total_hits / total_possible
    return np.array(preds), query_time, recall

# ============================================================
# 5) RUNS (tau fixed to 0.9 as required)
# ============================================================

TAU = 0.9
results = []

for perm in [16, 32, 64]:
    print(f"\n===== LSH num_perm={perm}, threshold={TAU} =====")

    lsh, build_time, mode, (b_used, r_used, s0_used) = build_lsh(X_train_text, num_perm=perm, tau=TAU)


    preds, query_time, recall = lsh_knn(
        lsh, X_test_text, y_train,
        bf_topk,
        num_perm=perm,
        k=7
    )

    total_time = build_time + query_time

    results.append({
    "Type": f"LSH-Jaccard (Perm={perm})",
    "BuildTime": build_time,
    "QueryTime": query_time,
    "TotalTime": total_time,
    "Recall": recall,
    "Params": f"perm={perm}, tau={TAU}, mode={mode}, (b,r)=({b_used},{r_used})" + (f", s0≈{s0_used:.3f}" if s0_used is not None else "")
})

# Brute row
results.insert(0, {
    "Type": "Brute-Force-Jaccard",
    "BuildTime": 0.0,
    "QueryTime": bf_query_time,
    "TotalTime": bf_query_time,
    "Recall": 1.0,
    "Params": "-"
})

results_df = pd.DataFrame(results)
print(results_df)

