# Exploration

This section includes initial data exploration and comparison of modeling approaches.

#### Key insights
- Reviews should contain enough information to help distinguish between positive and negative reviews. We know this because of reasonable review length (most between 100-200 tokens) and clear separation when visualising BERT embeddings.
- We see there is less diversity in language vs. a dataset of news articles, so this could make modelling simpler but with less information to make a successful prediction. If model performance struggles, we may have to investigate this further.

#### Caveats

- The data can be downloaded from `https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz` and the notebooks and experiments file expect it to be in a `data/` in this `exploration/` directory.
There is a helper in each notebook commented out to assist in downloading if needed.
- The exploration in these files is quick and dirty, without tests and with hardcoded values. Random states are initiaised in order to maintain a minimum level of reproducibility, however this should work as a one-off piece and some variability in results isn't expected to impact conclusions.

## Data exploration

#### files:
- `01-data-exploration.ipynb`: initial data exploration of the review data including word frequency and review lengths.
- `02-embedding-exploration`: visual exploration to discover whether BERT embeddings should be used as input data in model development.

#### How to run

1. create a virtual environment: `python -m venv myenv`
2. activate virtual environment: `source myenv/bin/activate`
3. install dependencies: `pip install -r exploration/requirements.txt`
4. run notebook cells using virtual environment as the kernel

- If you have trouble setting up a virtual environment you can find a complete tutorial here: `https://www.freecodecamp.org/news/how-to-setup-virtual-environments-in-python/`.
Alternatively you could also setup a poetry or docker based environment if you wish, using the `requirements.txt` file.

## Model training experiment

#### files:
- `experiments.py`: a quick comparison of 2 model architectures (shallow and deeper neural networks) and 2 datasets (tfidf and BERT embeddings). 

#### How to run

1. make sure you're in the project directory: `review-sentiment-analysis/`
2. create a neptune account and project following this article: ``
3. create a `.env` file containing neptune experiment credentials, which you can find in your neptune account: 
```
NEPTUNE_PROJECT="[YOUR ACCOUNT NAME]/[YOUR PROJECT NAME]"
NEPTUNE_KEY="[YOUR KEY]"
```
4. create a virtual environment: `python -m venv myenv`
5. activate virtual environment: `source myenv/bin/activate`
6. install dependencies: `pip install -r exploration/requirements.txt`
7. run experiments file: `python exploration/experiments.py`

- If you have trouble setting up a virtual environment you can find a complete tutorial here: `https://www.freecodecamp.org/news/how-to-setup-virtual-environments-in-python/`.
Alternatively you could also setup a poetry or docker based environment if you wish, using the `requirements.txt` file.
- Neptune is used as an experiment tracker and more detail on it's usage or setup can be found in documentation: `https://docs.neptune.ai/usage/quickstart/`

## Other files
- `utils/`: files containing helper functions for exploration tasks.
- `requirements.txt`: contains the dependencies
