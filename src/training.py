import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

 
def model_train(model, X_train, y_train, X_val, y_val, params):
    '''
    Run model training, result with loading the best performing weights and return the 
    training and validation accuracy that can be used to test fit.

    input:
    - model: pytorch model (a subclass of pytorch.nn.Module) 
    - X_train: input tensor of training data
    - y_train: target tensor of training data
    - X_val: input tensor of validation data
    - y_val: target tensor of validation data
    - params: dict of model parameters

    Output (Tuple):
    - best_acc: best accuracy during training
    - train_acc: accuracy of prediciton on training data
    '''
    # loss function and optimizer
    loss_fn = nn.BCELoss()  # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    
    n_epochs = params['n_epochs']   # number of epochs to run
    batch_size = params['batch_size']  # size of each batch
    batch_start = torch.arange(0, len(X_train), batch_size)
 
    # Hold the best model
    best_acc = - np.inf   # init to negative infinity
    best_weights = None
 
    for epoch in range(n_epochs):
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                X_batch = X_train[start:start+batch_size]
                y_batch = y_train[start:start+batch_size]
                # forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress
                acc = (y_pred.round() == y_batch).float().mean()
                bar.set_postfix(
                    loss=float(loss),
                    acc=float(acc)
                )
        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(X_val)
        acc = (y_pred.round() == y_val).float().mean()
        acc = float(acc)
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())
    # restore model and return best accuracy
    model.load_state_dict(best_weights)

    # calculate accuracy on train set to assess fit
    y_pred_train = model(X_train)
    train_acc = (y_pred_train.round() == y_train).float().mean()

    return best_acc, train_acc
