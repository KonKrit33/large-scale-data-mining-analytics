# ============================================
# News Category Prediction with TF-IDF and Linear SVM
# ============================================

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

# -------------------------
# 0) Paths
# -------------------------
TRAIN_PATH = "train.xlsx"
TEST_PATH = "test_without_labels.xlsx"
OUT_PATH = "testSet_categories.csv"
# -------------------------
# 1) Load
# -------------------------
train_df = pd.read_excel(TRAIN_PATH)
test_df  = pd.read_excel(TEST_PATH)

y_train = train_df["Label"].astype(str).values

# -------------------------
# 2) Text preprocessing
# # -------------------------
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
    s = re.sub(r"http\S+|www\.\S+|\S+@\S+|\d+", " ", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def tokenize_filter(s):
    return " ".join([w for w in s.split() if w not in STOPWORDS and len(w) >= 3])

def build_text(df):
    raw = (df["Title"].fillna("") + " " + df["Content"].fillna("")).values
    return np.array([tokenize_filter(clean_text(t)) for t in raw], dtype=object)

X_train_text = build_text(train_df)
X_test_text  = build_text(test_df)

# -------------------------
# 3) Define optimized TF-IDF + Linear SVM model
# -------------------------
model = Pipeline([
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

# -------------------------
# 4) Train on FULL train
# -------------------------
model.fit(X_train_text, y_train)

# -------------------------
# 5) Predict FULL test
# -------------------------
test_pred = model.predict(X_test_text)

# -------------------------
# 6) Export
# -------------------------
out = pd.DataFrame({
    "Id": test_df["Id"].astype(int),
    "Predicted": test_pred
})

out.to_csv(OUT_PATH, index=False)
print(f"Saved: {OUT_PATH} | rows={len(out)}")
print(out.head())