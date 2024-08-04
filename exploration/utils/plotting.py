import numpy as np
import pandas as pd
import re
import random
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)

nlp = spacy.load("en_core_web_sm")


def plot_review_length_dist(df: pd.DataFrame) -> None:
    '''
    Plots the distribution of review lengths in terms of tokens.    
    '''
    # get review token counts
    df['count'] = df['review'].apply(lambda review: len(re.findall(r'\w+', review)))

    # remove the extremes to zoom in a bit
    df_reduced = df[df['count'] < 500]

    # plot hist
    _, axes = plt.subplots(1, 2, sharey=True, figsize=(10, 5))
    sns.histplot(x=df['count'], kde=True, ax=axes[0]).set(title="Full Distribution of Review Lengths.", xlabel="Number of words in review")
    sns.histplot(x=df_reduced['count'], kde=True, ax=axes[1]).set(title="Distribution of Review Lengths Below 500.", xlabel="Number of words in review")


def plot_review_length_dist_by_sentiment(df: pd.DataFrame) -> None:
    '''
    Plots the distribution of review lengths in terms of tokens by sentiment. 
    '''
    # get review token counts
    df['count'] = df['review'].apply(lambda review: len(re.findall(r'\w+', review)))

    # remove the extremes to zoom in a bit
    df_below = df[df['count'] < 500]
    df_above = df[df['count'] > 1000]

    # plot hist
    _, axes = plt.subplots(1, 3, figsize=(15, 5))
    sns.histplot(data=df, x='count', hue='sentiment', kde=True, ax=axes[0])\
        .set(title="Full Distribution of Review Lengths.", xlabel="Number of words in review")
    sns.histplot(data=df_below, x='count', hue='sentiment', kde=True, ax=axes[1])\
        .set(title="Distribution of Review Lengths Below 500.", xlabel="Number of words in review")
    sns.histplot(data=df_above, x='count', hue='sentiment', kde=True, ax=axes[2])\
        .set(title="Distribution of Review Lengths Above 1000.", xlabel="Number of words in review")


def plot_quick_word_frequencies(df: pd.DataFrame, text_col) -> None:
    words = df[text_col].str.split(' ').explode().to_list()

    # some simple exceptions as a quick cleaning procedure
    exceptions = ['.', ',', '?', '!', 'the', 'a', 'and', 'of', 'that', 'this', 'it', 'as', 'to', 'is', 'in', 'was', 'with']
    words = [word.lower() for word in words if word.lower() not in exceptions]
    words = Counter(words).most_common(24)
    words = {word: count for word, count in words}

    plt.figure(figsize=(3, 6))
    sns.barplot(x=list(words.values()), y=list(words.keys())).set(title=f'top {len(list(words.keys()))} words')


def plot_top_adjectives_and_preview_noun_chunks(df: pd.DataFrame, text_col: str) -> None:
    df_sample = _get_balanced_sample(df, label_col='sentiment', sample_size=100)
    docs = [nlp(text) for text in df_sample[text_col].to_list()]
    chunks = [chunk for doc in docs for chunk in _clean_noun_chunks(doc)]

    print("20 noun chunks from sample reviews:")
    print(chunks[:20])

    adjectives = [token.text for doc in docs for token in doc if token.pos_ == 'ADJ']
    adjectives_counts = Counter(adjectives).most_common(20)
    adjectives = {adjectives: count for adjectives, count in adjectives_counts}

    plt.figure(figsize=(3, 6))
    sns.barplot(x=list(adjectives.values()), y=list(adjectives.keys())).set(title=f'top {len(list(adjectives.keys()))} adjectives')


def plot_pos_tag_breakdown(df: pd.DataFrame) -> None:
    '''
    A series of plots exploring the distribution of patrs-of-speech tags in 
    the text of the movie review and news article datasets.    
    '''
    df_sample = _get_balanced_sample(df, label_col='sentiment', sample_size=100)
    df_pos = _get_spacy_pos_df(df_sample, text_col='review', label_col='sentiment')

    # setup comparison dataset for side-by-side
    df_pos_compare = _get_comparison_pos_dataset()

    _, axes = plt.subplots(5, 1, sharex=True, figsize=(15, 15))
    sns.countplot(data=df_pos_compare,
                  x='pos',
                  order = df_pos['pos'].value_counts().index,
                  stat='percent',
                  color='mediumseagreen',
                  ax=axes[0]
                  ).set(title="Figure 1: News dataset, % of tokens per tag .",
                        ylim=(0, 25))

    sns.countplot(data=df_pos,
                  x='pos',   
                  order = df_pos['pos'].value_counts().index,
                  stat='percent',
                  ax=axes[1]
                  ).set(title="Figure 2: Movie dataset, % of tokens per tag.",
                        ylim=(0, 25))
    
    sns.countplot(data=df_pos,
                  x='pos', 
                  hue='label',  
                  order = df_pos['pos'].value_counts().index,
                  stat='percent',
                  ax=axes[2]
                  ).set(title="Figure 3: Movie dataset, % of tokens per tag by sentiment.",
                        ylim=(0, 25))

    sns.barplot(data=_get_unique_tokens_per_tag(df_pos_compare),
                x='pos',
                y='pc_unique',
                color='mediumseagreen',
                ax=axes[3]
                  ).set(title="Figure 4: News dataset, % tokens per tag are unique.",
                        ylim=(0, 100))
    
    sns.barplot(data=_get_unique_tokens_per_tag(df_pos),
                x='pos',
                y='pc_unique',
                ax=axes[4]
                  ).set(title="Movie dataset: % tokens per tag are unique.",
                        ylim=(0, 100))


def _clean_noun_chunks(doc):
    chunks = list()
    for chunk in doc.noun_chunks:
        if all(token.is_stop != True and token.is_punct != True and '-PRON-' not in token.lemma_ for token in chunk) == True:
            if len(chunk) > 1:
                chunks.append(chunk)
    return chunks


def _get_comparison_pos_dataset() -> pd.DataFrame:
    splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
    df_news = pd.read_parquet("hf://datasets/fancyzhx/ag_news/" + splits["train"])

    df_sample = _get_balanced_sample(df_news, label_col='label', sample_size=10)
    return _get_spacy_pos_df(df_sample, text_col='text', label_col='label')


def _get_spacy_pos_df(df: pd.DataFrame, text_col: str, label_col: str) -> pd.DataFrame:
    docs = list()
    for text in df[text_col].tolist():
        docs.append(nlp(text))

    docs = list(zip(docs, df[label_col]))

    pos_tags = list()
    for doc, label in docs:
        for token in doc:
            pos_tags.append((token.lemma_, token.pos_, label))

    return pd.DataFrame(pos_tags, columns=['token', 'pos', 'label'])


def _get_balanced_sample(df: pd.DataFrame, label_col: str, sample_size: int) -> pd.DataFrame:
    df_sample = pd.DataFrame(columns=["text", "label"])
    for label in df[label_col].unique():
        df_temp = df[df[label_col] == label].sample(n=sample_size).reset_index(drop=True)
        df_sample = pd.concat([df_sample, df_temp])
    return df_sample


def _get_unique_tokens_per_tag(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Calculates what % of tokens per part-of-speech tag are unique.
    ie. What % of nouns are distinct?
    '''
    df_gb = df.groupby(['pos']).agg({'token': ['count', 'nunique']})
    df_gb.columns = df_gb.columns.droplevel(0)
    df_gb['pc_unique'] = 100 * (df_gb['nunique'] / df_gb['count'])
    return df_gb
