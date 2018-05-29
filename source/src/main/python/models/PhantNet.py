_author__ = 'MSteger'

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, autograd
from torchvision import models

class PhantNet(nn.Module):

    def __init__(self, pretrained_models = models.alexnet(pretrained=True) , input_size = (3, 32, 32), num_class = 200):
        super(PhantNet, self).__init__()
        self.features = pretrained_models.features
        for weights in self.features.parameters(): weights.requires_grad = False
        self.flat_fts = self.get_flat_fts(input_size, self.features)
        self.classifier = nn.Sequential(
            nn.Linear(self.flat_fts, 100),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(100, num_class),
        )

    def get_flat_fts(self, in_size, fts):
        f = fts(autograd.Variable(torch.ones(1, *in_size)))
        return int(np.prod(f.size()[1:]))

    def forward(self, x):
        fts = self.features(x)
        flat_fts = fts.view(-1, self.flat_fts)
        return F.softmax(self.classifier(flat_fts), dim = 1)

class PhantTrain(object):

    def __init__(self, model, optimizer, loss, batch_size, device, print_freq = 10, metrics = None, LE = None, verbose = True):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.current_epoch = 1
        self.batch_size = batch_size
        self.device = device
        self.metrics = metrics
        self.LE = LE
        self.print_freq = print_freq
        self.performance_train, self.performance_val, self.performance_test = {}, {}, {}
        self.verbose = verbose

    def compute_metrics(self, y, yHat):
        performance = {}
        for metric_name, metric_fn in self.metrics:
            performance[metric_name] = metric_fn(y, yHat)
        return performance

    def train_epoch(self, epoch, train_data):
        self.model.train()
        batch_performance_train = {}
        for batch_idx, (X, y) in enumerate(train_data):
            X = autograd.Variable(X).to(self.device)
            if self.LE is not None: y = self.LE.transform(y)
            y = autograd.Variable(torch.from_numpy(np.array(y)), requires_grad=False).to(self.device)
            self.optimizer.zero_grad()
            yHat = self.model(X)
            batch_loss = self.loss(yHat, y)
            batch_performance_train[batch_idx] = self.compute_metrics(y = y, yHat = yHat)
            batch_loss.backward()
            self.optimizer.step()
            if (batch_idx % self.print_freq == 0) & (self.verbose): print '\rTrain Epoch: {} [{}/{} ({:.1f}%)]\t Batch Loss: {:.6f} Performance Train: {}'.format(epoch +1, (batch_idx + 1) * self.batch_size, len(train_data.dataset), 100.0 * (batch_idx + 1) * self.batch_size / len(train_data.dataset), batch_loss.item(), batch_performance_train[batch_idx])
        return epoch + 1, batch_performance_train

    def validate(self, epoch, val_data):
        self.model.eval()
        batch_performance_val = {}
        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(val_data):
                X = X.to(self.device)
                if self.LE is not None: y = self.LE.transform(y)
                y = torch.from_numpy(np.array(y)).to(self.device)
                yHat = self.model(X)
                batch_performance_val[epoch] = self.compute_metrics(y = y, yHat = yHat)
        return batch_performance_val

    def fit(self, epochs, train_data, val_data = None):
        epoch = 0
        while epoch < epochs:
            epoch, performance_train = self.train_epoch(epoch = epoch, train_data = train_data)
            performance_train_avg = {key: sum([e[key] for e in performance_train.values()]) / len(performance_train) for key in performance_train.values()[0]}
            self.performance_train[epoch - 1] = performance_train_avg
            performance_str = 'Iteration: {} Performance Train (batch avg): {}'.format(epoch, performance_train_avg)
            if val_data is not None:
                performance_val = self.validate(epoch = epoch - 1, val_data = val_data)
                performance_val_avg = {key: sum([e[key] for e in performance_val.values()]) / len(performance_val) for key in performance_val.values()[0]}
                self.performance_val[epoch - 1] = performance_val_avg
                performance_str = '{} Performance Val (batch avg): {}'.format(performance_str, performance_val_avg)
            if self.verbose: print performance_str

        return self

    def evaluate(self, test_data):
        return self

    def callbacks(self):
        # add tensorboard
        return



if __name__ == '__main__':
    print 'done'