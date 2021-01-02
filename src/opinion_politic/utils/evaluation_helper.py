"""
evaluation_helper.py is a evaluation file for writing evaluation methods
"""

import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    rounded = torch.round(torch.sigmoid(preds))
    correct = (rounded == y).float()
    return correct.sum().to(DEVICE)/len(correct)


def categorical_accuracy(pred, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_pred = pred.argmax(dim=1, keepdim=True)
    correct = max_pred.squeeze(1).eq(y)
    return correct.sum().to(DEVICE) / torch.FloatTensor([y.shape[0]]).to(DEVICE)
