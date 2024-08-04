# tokenize
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel

MODEL = 'distilbert-base-uncased-finetuned-sst-2-english'
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = TFAutoModel.from_pretrained(MODEL)


def get_document_embeddings(reviews: list) -> np.array:
    '''
    Takes a list of text documents, tokenizes each, 
    extracts token embeddings and performs a simple 
    simple vector average to get the document embeddings.
    '''
    doc_embeddings = list()
    for review in tqdm(reviews):
        # tokenize the review
        input_ids = tf.constant(tokenizer.encode(review))[None, :512]

        # get hidden state from model prediction
        outputs = model(input_ids)
        last_hidden_states = outputs[0]

        # get embeddings for tokens, removing tags from the tokenizer
        token_embeddings = last_hidden_states.numpy()[0][1:-1]

        # take average of the token embeddings
        doc_embeddings.append(np.mean(token_embeddings, axis=0))

    return np.array(doc_embeddings)
