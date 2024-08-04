import os
from dotenv import load_dotenv 
import neptune
from sklearn.model_selection import StratifiedKFold
import torch
import logging
import numpy as np

from utils.data import get_data_as_csv
from utils.model import Shallow, Deep
from utils.train import model_train
from utils.preprocessing import get_bow_matrix_and_vectorizer, process_labels, dim_reduction
from utils.embedding import get_document_embeddings

logging.basicConfig(level=logging.INFO)

load_dotenv() 


def main():
    df_train = get_data_as_csv()
    df_sample = df_train.sample(n=2000)
    _run_experiment_bow(df_sample)
    _run_experiment_bert(df_sample)

    
def _run_experiment_bow(df):
    '''
    Compares model architectures using a bag-of-words input dataset.
    All results are tracked in neptune.
    '''
    X, _ = get_bow_matrix_and_vectorizer(docs=df['review'].to_list())
    X_red, _ = dim_reduction(X, 100)
    y = process_labels(df['sentiment'].to_list())

    # Convert to 2D PyTorch tensors
    X = torch.tensor(X_red, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    # define 5-fold cross validation
    kfold = StratifiedKFold(n_splits=5, shuffle=True)

    params = dict(n_epochs=100, batch_size=500, lr=0.0001)
    meta = dict(name='bag-of-words', 
                description='experiment to assess the performance of model training against a bow dataset')

    for model in [Shallow(), Deep()]:
        _run_cross_validation(kfold, model, X, y, params, meta)

    logging.info(f"Experiment complete for {meta['name']}")


def _run_experiment_bert(df):
    '''
    Compares model architectures with pretrained bert embeddings as input.
    All results are tracked in neptune.
    '''
    X = get_document_embeddings(df['review'].tolist())
    X_red, _ = dim_reduction(X, 100, use_hellinger=False)
    y = process_labels(df['sentiment'].to_list())

    # Convert to 2D PyTorch tensors
    X = torch.tensor(X_red, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    # define 5-fold cross validation
    kfold = StratifiedKFold(n_splits=5, shuffle=True)

    params = dict(n_epochs=100, batch_size=500, lr=0.0001)
    meta = dict(name='bert embeddings', 
                description='experiment to assess the performance of model training against a bert embedding dataset')

    for model in [Shallow(), Deep()]:
        _run_cross_validation(kfold, model, X, y, params, meta)

    logging.info(f"Experiment complete for {meta['name']}")


def _run_cross_validation(kfold: StratifiedKFold, model, X, y, params, meta) -> None:
    '''
    Run 5 fold cross validation on the trianing data & record run data to neptune.    
    '''
    run = neptune.init_run(
        project=os.getenv('NEPTUNE_PROJECT'),
        api_token=os.getenv('NEPTUNE_KEY'),
        name=meta['name'],
        description=meta['description']
    ) 

    # record number of trainable parameters in model as an indicaiton of model size
    params['n_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)

    run['parameters'] = params
    run['model'] = type(model).__name__

    diff_accuracies, val_accuracies = list(), list()
    for train, test in kfold.split(X, y):
        # create model, train, and get accuracy
        val_acc, train_acc = model_train(model, X[train], y[train], X[test], y[test], params=params)

        diff_acc = train_acc - val_acc
        diff_accuracies.append(diff_acc)
        val_accuracies.append(val_acc)


    run['metrics/mean_acc'], run['metrics/std_acc'], run['metrics/mean_diff'] = \
        np.mean(val_accuracies), np.std(val_accuracies), np.mean(diff_accuracies)
    
    logging.info(f"Cross val complete for {meta['name']}")    
    run.stop()


if __name__ == "__main__":
    main()
    