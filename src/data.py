import os
import requests
from typing import NamedTuple, List
import tarfile
import shutil
import logging

from config import Config

logging.basicConfig(level=logging.INFO)
config = Config()


class RawData(NamedTuple):
    text: str
    label: str


def get_data(path: os.PathLike) -> List[RawData]:
    '''
    Reads in data to a list of NamedTuples. 

    inputs:
    - mode: either 'train' or 'test' to select which purpose

    outputs:
    - list(RawData): with text (review) and label (sentiment) 
    '''
    logging.info("Reading data")

    reviews = list()
    for path in [os.path.join(path, polarity_path) for polarity_path in [config.POSITIVE_TRAIN_PATH, config.NEGATIVE_TRAIN_PATH]]:
        for f in os.listdir(path):
            file_path = os.path.join(path, f)
            with open(file_path, 'r') as review:
                sentiment = path.split('/')[-1]
                reviews.append(RawData(text=review.read(), label=sentiment))

    logging.info("Read complete")
    return reviews 


def download_data(url: str) -> None:
    '''
    Downloads and unzips data from the url to a target filepath.

    inputs:
    - url: string representing the url of the dataset
    '''
    # create data directory
    if not os.path.exists(config.LOCAL_DATA_PATH):
        os.makedirs(config.LOCAL_DATA_PATH)
        logging.info("New data directory created")

    # set target file path for downloaded data
    target_filepath = os.path.join(config.LOCAL_DATA_PATH, url.split('/')[-1])
    
    # download file
    logging.info("Downloading dataset")

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(target_filepath, 'wb') as f:
            f.write(response.raw.read())

    logging.info("Extracting dataset")
    if url.split('.')[-1] == "gz":
        tar = tarfile.open(target_filepath, "r:gz")
        tar.extractall(config.LOCAL_DATA_PATH)
        tar.close()
    elif url.split('.')[-1] == "zip":
        shutil.unpack_archive(target_filepath, config.LOCAL_DATA_PATH)


    # delete zipped file
    os.remove(target_filepath)
    logging.info(f"Dataset now available in {config.LOCAL_DATA_PATH}/ directory")
