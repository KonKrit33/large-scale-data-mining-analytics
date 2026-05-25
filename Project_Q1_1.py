import pandas as pd
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from wordcloud import WordCloud
import matplotlib.pyplot as plt

df = pd.read_excel("train.xlsx")

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
    s = re.sub(r"http\S+|www\.\S+|\S+@\S+|\d+"," ",s)
    s = re.sub(r"[^a-z\s]"," ",s)
    return re.sub(r"\s+"," ",s).strip()

def tokenize_filter(s):
    return " ".join([w for w in s.split() if w not in STOPWORDS and len(w) >= 3])

df["text"] = (df["Title"].fillna("") + " " + df["Content"].fillna("")).map(clean_text)
df["text"] = df["text"].map(tokenize_filter)

for lab in ["Business","Entertainment","Health","Technology"]:
    corpus = " ".join(df[df["Label"]==lab]["text"])
    wc = WordCloud(width=1400,height=800,background_color="white",
                   max_words=300,collocations=False,
                   random_state=42).generate(corpus)
    wc.to_file(f"wordcloud_{lab.lower()}.png")