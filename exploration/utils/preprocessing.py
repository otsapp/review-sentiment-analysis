import numpy as np
import re
import spacy
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
import umap
from bs4 import BeautifulSoup
import html

logging.basicConfig(level=logging.INFO)
nlp = spacy.load("en_core_web_sm")

LABEL_MAP = {
    "pos": 1,
    "neg": 0
}


def get_bow_matrix_and_vectorizer(docs: list) -> tuple:
    # custom tokenizer
    def tokenizer(doc: str) -> list:
        return [token.text for token in nlp.tokenizer(doc) if not token.is_stop]

    # create bag-of-words (tfidf)
    logging.info("Tokenizing & vectorizing text")
    tfidf = TfidfVectorizer(tokenizer=tokenizer, max_df=0.9, max_features=10000, ngram_range=(1, 2))
    tfidf_fitted = tfidf.fit(docs)
    return tfidf_fitted.transform(docs), tfidf_fitted


def dim_reduction(arr: np.array, n: int, use_hellinger=True) -> tuple:
    logging.info("Reducing dimensions with UMAP")
    metric = "hellinger" if use_hellinger  else "euclidean"
    mapper = umap.UMAP(metric=metric, n_components=n, random_state=42).fit(arr)
    return mapper.transform(arr), mapper


def process_labels(labels: list) -> list:
    return [*map(LABEL_MAP.get, labels)]


def clean_text(text: str):
    text = html.unescape(text)
    # remove html tags
    text = BeautifulSoup(text, "lxml").text
    return text


