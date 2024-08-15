import os


class Config():
    POSITIVE_TRAIN_PATH = os.path.join("aclImdb", "train", "pos")
    NEGATIVE_TRAIN_PATH = os.path.join("aclImdb", "train", "neg")
    LOCAL_DATA_PATH = "data"
    LANGUAGE_MODEL = 'distilbert-base-uncased-finetuned-sst-2-english'
    MODEL_PARAMETERS = dict(n_epochs=100, batch_size=500, lr=0.0001)
    