import numpy as np
import logging
from typing import NamedTuple
from bs4 import BeautifulSoup
import html
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)

LABEL_MAP = {"pos": 1, "neg": 0}
VAL_SIZE = 0.2
RND_STATE = 42
LXML = "lxml"

class ProcessedData(NamedTuple):
    text: str
    label: int


def process_text(text: str):
    text = html.unescape(text)
    # remove html tags
    text = BeautifulSoup(text, LXML).text
    return text


def process_labels(label: str) -> list:
    return LABEL_MAP[label]


def train_val_split(X: np.Array, y: list):
    return train_test_split(X, y, test_size=VAL_SIZE, random_state=RND_STATE)
