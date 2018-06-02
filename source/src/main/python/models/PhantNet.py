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

    def __init__(self, model, optimizer, loss, batch_size, device, LE = None, verbose = True):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.current_epoch = 1
        self.batch_size = batch_size
        self.device = device
        self.LE = LE
        self.verbose = verbose

    def train_epoch(self, epoch, train_data, callbacks):
        self.model.train()
        self.logger['epoch'] = epoch
        callbacks = self._callbacks(callbacks=callbacks, state='on_epoch_begin', set_model=True, set_logger=True, retrieve_logger = True)
        y_train, yHat_train, batches = [], [], len(train_data)
        for batch_idx, (X, y) in enumerate(train_data):
            self.logger['batch'] = batch_idx + 1
            callbacks = self._callbacks(callbacks=callbacks, state='on_batch_begin', set_model=True, set_logger=True, retrieve_logger = True)
            X = autograd.Variable(X).to(self.device)
            if self.LE is not None: y = self.LE.transform(y)
            y = autograd.Variable(torch.from_numpy(np.array(y)), requires_grad=False).to(self.device)
            self.optimizer.zero_grad()
            yHat = self.model(X)
            yHat_train.append(yHat)
            y_train.append(y)
            batch_loss = self.loss(yHat, y)
            batch_loss.backward()
            self.optimizer.step()
            callbacks = self._callbacks(callbacks=callbacks, state='on_batch_end', set_model=True, y_train = y, yHat_train = yHat, retrieve_logger = True)

        return epoch, callbacks, y_train, yHat_train

    def validate(self, val_data):
        self.model.eval()
        y_val, yHat_val = [], []
        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(val_data):
                X = X.to(self.device)
                if self.LE is not None: y = self.LE.transform(y)
                y = torch.from_numpy(np.array(y)).to(self.device)
                yHat_val.append(self.model(X))
                y_val.append(y)
        return y_val, yHat_val

    def fit(self, epochs, train_data, val_data = None, callbacks = None):
        self.logger = {}
        self.logger['epochs'], self.logger['batches'] = epochs, len(train_data)
        callbacks = self._callbacks(callbacks = callbacks, state = 'set_data', training_data = train_data, validation_data = val_data, retrieve_logger = False)
        callbacks = self._callbacks(callbacks = callbacks, state = 'on_train_begin', set_model = True, set_logger = True, retrieve_logger = True)

        epoch = 0
        while epoch < epochs:
            epoch, callbacks, y_train, yHat_train = self.train_epoch(epoch = epoch+1, train_data = train_data, callbacks = callbacks)
            y_val, yHat_val = self.validate(val_data = val_data)
            callbacks = self._callbacks(callbacks=callbacks, state='on_epoch_end', set_model=True, y_val = y_val, yHat_val = yHat_val, y_train = y_train, yHat_train = yHat_train, set_logger=True, retrieve_logger = True)

        return self

    def evaluate(self, test_data):
        return self

    def _callbacks(self, callbacks, state, set_model = False, set_logger = False, retrieve_logger = False, **params):
        if callbacks is not None:
            for cb in callbacks:
                if set_model: cb.set_model(model = self.model)
                if set_logger: cb.set_logger(logger = self.logger)
                getattr(cb, state)(**params)
                if retrieve_logger: self.logger = cb.get_logger()
        return callbacks

if __name__ == '__main__':
    print 'done'