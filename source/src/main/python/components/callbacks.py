_author__ = 'MSteger'

import torch
from tqdm import tqdm

class Callback(object):
    """
    based on https://github.com/keras-team/keras/blob/master/keras/callbacks.py
    """
    def __init__(self):
        self.training_data = None
        self.validation_data = None
        self.model = None
        self.logger = {}

    def set_model(self, model):
        self.model = model

    def set_data(self, training_data, validation_data):
        self.training_data = training_data
        self.validation_data = validation_data

    def set_logger(self, logger):
        self.logger = logger

    def get_logger(self):
        return self.logger

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self, y_val, yHat_val, y_train, yHat_train):
        pass

    def on_batch_begin(self):
        pass

    def on_batch_end(self, y_train, yHat_train):
        pass

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass

class MetricTracker(Callback):

    def __init__(self, metrics, save = None, verbose = True):
        super(Callback, self).__init__()
        self.metrics = metrics
        self.save = save
        self.verbose = verbose

    def _compute_metrics(self, y_train = None, yHat_train = None, y_val = None, yHat_val = None):
        performance = {}
        for m_name, m_fn in self.metrics:
            performance[m_name] = {}
            if y_train is not None:
                performance[m_name]['train'] = m_fn(torch.cat(y_train), torch.cat(yHat_train)) if type(y_train) == type([]) else m_fn(torch.cat([y_train]), torch.cat([yHat_train]))

            if y_val is not None:
                performance[m_name]['val'] = m_fn(torch.cat(y_val), torch.cat(yHat_val)) if type(y_val) == type([]) else m_fn(torch.cat([y_val]), torch.cat([yHat_val]))

        return performance

    def on_train_begin(self):
        if 'batch_metrics' not in self.logger.keys(): self.logger['batch_metrics'] = {}
        if 'epoch_metrics' not in self.logger.keys(): self.logger['epoch_metrics'] = {}
        return self

    def on_batch_end(self, y_train, yHat_train):
        performance = {self.logger['batch']: self._compute_metrics(y_train = y_train, yHat_train = yHat_train)}
        self.logger['batch_metrics'].update(performance)
        return self

    def on_epoch_end(self, y_val, yHat_val, y_train, yHat_train):
        performance = {self.logger['epoch']: self._compute_metrics(y_train = y_train, yHat_train = yHat_train, y_val = y_val, yHat_val = yHat_val)}
        self.logger['epoch_metrics'].update(performance)

        if self.verbose: print '\nPerformance Epoch {}: {}'.format(self.logger['epoch'], performance)
        return self

class ProgressBar(Callback):

    def on_epoch_begin(self):
        self.progbar = tqdm(total=self.logger['batches'], unit=' batches')
        self.epochs = self.logger['batches']

    def on_batch_end(self, y_train, yHat_train):
        self.progbar.update(1)
        self.progbar.set_description('MODE[TRAIN] EPOCH[{}|{}]'.format(self.logger['epoch'], self.logger['epochs']))
        # TODO: add some metric if they are tracked! maybe with specific param: show_batch_metric etc.
        return self

    def on_epoch_end(self, y_val, yHat_val, y_train, yHat_train):
        self.progbar.close()
        return self

class ModelCheckpoint(Callback):

    def __init__(self, save_folder_path, metric):
        super(Callback, self).__init__()
        self.save_folder_path = save_folder_path
        self.metric = metric

    def on_epoch_end(self, *_):
        # TODO: :)
        return self




