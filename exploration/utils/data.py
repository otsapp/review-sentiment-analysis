import os
import requests
import tarfile
import shutil
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
DATA_DIR = "data"
ACLIMDB = "aclImdb"
POSITIVE_DIR = "pos"
NEGATIVE_DIR = "neg"


def get_data_as_csv(mode='train') -> pd.DataFrame:
    '''
    Reads in data to a pandas dataframe. 
    inputs:
    - mode: either 'train' or 'test' to select which purpose

    outputs:
    - pd.DataFrame: with a text (review) and label (sentiment) column
    
    '''
    logging.info("Reading data into pandas DataFrame")

    reviews = list()
    for path in [os.path.join(DATA_DIR, ACLIMDB, mode, polarity_dir) for polarity_dir in [POSITIVE_DIR, NEGATIVE_DIR]]:
        for f in os.listdir(path):
            file_path = os.path.join(path, f)
            with open(file_path, 'r') as review:
                sentiment = path.split('/')[-1]
                reviews.append((review.read(), sentiment))

    logging.info("Read complete")
    return pd.DataFrame(reviews, columns=['review', 'sentiment'])   


def download_data(url: str) -> None:
    # create data directory
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        logging.info("New data directory created")

    # set target file path for downloaded data
    target_filepath = os.path.join(DATA_DIR, url.split('/')[-1])
    
    # download file
    logging.info("Downloading dataset")

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(target_filepath, 'wb') as f:
            f.write(response.raw.read())

    logging.info("Extracting dataset")
    if url.split('.')[-1] == "gz":
        tar = tarfile.open(target_filepath, "r:gz")
        tar.extractall(DATA_DIR)
        tar.close()
    elif url.split('.')[-1] == "zip":
        shutil.unpack_archive(target_filepath, DATA_DIR)


    # delete zipped file
    os.remove(target_filepath)
    logging.info(f"Dataset now available in {DATA_DIR}/ directory")
