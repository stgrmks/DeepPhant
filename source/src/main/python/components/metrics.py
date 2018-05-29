_author__ = 'MSteger'

import torch
from sklearn import metrics

def accuracy_score(y, yHat):
    yHat = yHat.max(dim = 1)[1]
    accuracy = torch.sum(y == yHat).double() / yHat.size(0)
    return accuracy.data.tolist() * float(1.)

def sk_accuracy_score(y, yHat):
    y = y.detach().cpu().numpy()
    yHat = yHat.max(dim = 1)[1].detach().cpu().numpy()
    return metrics.accuracy_score(y, yHat)


