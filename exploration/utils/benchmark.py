import spacy
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, roc_auc_score

from data import get_data_as_csv

logging.basicConfig(level=logging.INFO)
nlp = spacy.load("en_core_web_sm")

LABEL_MAP = {
    "pos": 1,
    "neg": 0
}


def main():

    # import data
    df_reviews = get_data_as_csv(mode='train')
    df_reviews['sentiment'] = df_reviews['sentiment'].map(LABEL_MAP)

    # custom tokenizer
    def tokenizer(doc: str) -> list:
        return [token.text for token in nlp.tokenizer(doc) if not token.is_stop]

    # create bag-of-words (tfidf)
    logging.info("Tokenizing & vectorizing text")
    tfidf = TfidfVectorizer(tokenizer=tokenizer, ngram_range=(1, 2))
    matrix = tfidf.fit_transform(df_reviews['review'].to_list())

    # fit classifier
    logging.info("Training benchmark classifier")
    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(matrix, df_reviews['sentiment'])

    # setup evaluation metrics
    logging.info("Beginning benchmark model evaluation")
    df_test = get_data_as_csv(mode='test')
    df_test['sentiment'] = df_test['sentiment'].map(LABEL_MAP)

    X_test = tfidf.transform(df_test['review'].to_list())
    y_true = df_test['sentiment'].to_list()

    y_pred = clf.predict(X_test)

    precision = precision_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)

    logging.info(f"Benchmark precision: {precision}")
    logging.info(f"ROC AUC: {roc_auc}")


if __name__=='__main__':
    main()
