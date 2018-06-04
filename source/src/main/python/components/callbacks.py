_author__ = 'MSteger'

import os
import torch
import datetime
from tqdm import tqdm
from helpers import geo_mean
from tensorboardX import SummaryWriter
import torchvision.utils as vutils

class Callback(object):
    """
    based on https://github.com/keras-team/keras/blob/master/keras/callbacks.py
    """
    def __init__(self):
        self.training_data = None
        self.validation_data = None
        self.model = None
        self.logger = {}

    def set_model(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

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
    # TODO: include save option

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

    def __init__(self, show_batch_metrics = ['accuracy_score', 'sk_accuracy_score', 'log_loss']):
        super(Callback, self).__init__()
        self.show_batch_metrics = show_batch_metrics

    def on_epoch_begin(self):
        self.progbar = tqdm(total=self.logger['batches'], unit=' batches')
        self.epochs = self.logger['batches']

    def on_batch_end(self, y_train, yHat_train):
        self.progbar.update(1)
        desc_string = 'MODE[TRAIN] EPOCH[{}|{}]'.format(self.logger['epoch'], self.logger['epochs'])

        if self.show_batch_metrics is not None:
            for b_metric in self.show_batch_metrics:
                b_metric_val = self.logger['batch_metrics'][self.logger['batch']][b_metric].values()[0]
                b_metric_avg = geo_mean([d[b_metric].values()[0] for d in self.logger['batch_metrics'].values()])
                desc_string = '{} {}[{:.4f}|{:.4f}(avg)]'.format(desc_string, b_metric, b_metric_val, b_metric_avg)
        self.progbar.set_description(desc_string)

        return self

    def on_epoch_end(self, y_val, yHat_val, y_train, yHat_train):
        self.progbar.close()
        return self

class ModelCheckpoint(Callback):

    def __init__(self, save_folder_path, metric = 'log_loss', best_metric_highest = False, verbose = True):
        super(Callback, self).__init__()
        self.save_folder_path = save_folder_path
        self.metric = metric
        self.best_metric_highest = best_metric_highest
        self.verbose = verbose
        if not os.path.exists(save_folder_path): os.makedirs(save_folder_path)

    def _save_checkpoint(self, best_performance, current_performance, dstn):
        checkpoint = {
            'epoch': self.logger['epoch'],
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, dstn)
        if self.verbose: print '\nPerformance  Epoch {} Improved from {} to {}! Saving States & Optimizer to: {}'.format(self.logger['epoch'], best_performance, current_performance, dstn)
        return self

    def on_epoch_end(self, **_):
        model_ckps = [(file, file.split('__')[1].split('.pkl')[0]) for file in os.listdir(self.save_folder_path) if file.endswith('.pkl')]
        current_performance = self.logger['epoch_metrics'][self.logger['epoch']][self.metric]['val']
        current_time = datetime.datetime.now().__str__()
        dstn_string = '{}__{}__{}'.format(current_time, current_performance, self.logger['epoch'])
        dstn = os.path.join(self.save_folder_path, '{}.pkl'.format(dstn_string))
        if len(model_ckps) > 0:
            best_performance = float(sorted(model_ckps, key=lambda x: x[1], reverse=False)[0][1])
            if ((current_performance > best_performance) & (self.best_metric_highest)) | ((current_performance < best_performance) & (not self.best_metric_highest)):
                self._save_checkpoint(best_performance = best_performance, current_performance = current_performance, dstn = dstn)
            else:
                if self.verbose: 'Performance  Epoch {} did not Improve!'.format(self.logger['epoch'])
        else:
            self._save_checkpoint(best_performance = 'NaN', current_performance = current_performance, dstn = dstn)

        return self

class TensorBoard(Callback):

    # TODO: add option to write images; find fix for graph

    def __init__(self, log_dir, update_frequency = 10):
        super(Callback, self).__init__()
        self.log_dir = log_dir
        self.writer = None
        self.update_frequency = update_frequency

    def on_train_begin(self, **_):
        self.writer = SummaryWriter(os.path.join(self.log_dir, datetime.datetime.now().__str__()))
        rndm_input = torch.autograd.Variable(torch.rand(1, *self.model.input_shape), requires_grad = True).to(self.logger['device'])
        fwd_pass = self.model(rndm_input)
        self.writer.add_graph(self.model, fwd_pass)
        return self

    def on_epoch_end(self, **_):
        if (self.logger['epoch'] % self.update_frequency) == 0:

            epoch_metrics = self.logger['epoch_metrics'][self.logger['epoch']]

            for e_metric, e_metric_dct in epoch_metrics.iteritems():
                for e_metric_split, e_metric_val in e_metric_dct.iteritems():
                    self.writer.add_scalar('{}/{}'.format(e_metric_split, e_metric), e_metric_val, self.logger['epoch'])

            for name, param in self.model.named_parameters():
                self.writer.add_histogram(name.replace('.', '/'), param.clone().cpu().data.numpy(), self.logger['epoch'])

        return self

    def on_train_end(self, **_):
        return self.writer.close()




