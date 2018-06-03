_author__ = 'MSteger'

import torch
from torch import nn
from models.PhantNet import PhantNet, PhantTrain
from torchvision import transforms
from components.preprocessing import loaders, PhantDataset
from sklearn.preprocessing import LabelEncoder
from components.callbacks import MetricTracker, ProgressBar, ModelCheckpoint
from components import metrics, helpers
from functools import partial

def model_evaluation(path = '/media/msteger/storage/resources/tiny-imagenet-200'):

    # setup
    batch_size = 32

    # data
    transformer = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    data_transformers = {'train': transformer, 'val': transformer, 'test': transformer}
    data_loaders = loaders(path = path, dataset = PhantDataset, transformers = data_transformers, batch_size = batch_size, shuffle = True, num_workers = 4, pin_memory = True)
    # data_organization= tinyImageNet_Prepare(path = path)
    # classes = data_organization.get_classes(classes_lst = None)
    LE = LabelEncoder().fit(range(2))

    # model
    device = torch.device('cuda')
    model = PhantNet(input_size = (3, 224, 224), num_class = 2)

    # training
    training = PhantTrain(
        model = model.to(device),
        optimizer = torch.optim.Adam(model.classifier[3].parameters(), lr = 0.01),
        loss = nn.CrossEntropyLoss(),
        batch_size = batch_size,
        device = device,
        LE = LE,
        # checkpoint_path = '/media/msteger/storage/resources/DreamPhant/models/foobar/2018-06-03 14:36:19.442171__0.313464730978__17.pkl',
        verbose = True
    )
    training.fit(epochs = 500, train_data = data_loaders['train'], val_data = data_loaders['val'], \
                 callbacks = [
                     MetricTracker(metrics = [('log_loss', metrics.log_loss), ('accuracy_score', metrics.accuracy_score),('sk_accuracy_score', metrics.sk_accuracy_score)]),
                     ProgressBar(show_batch_metrics = ['log_loss']),
                     ModelCheckpoint(save_folder_path = r'/media/msteger/storage/resources/DreamPhant/models/foobar/', metric = 'log_loss', best_metric_highest = False, verbose = True)
                 ])

    # evaluation
    # training.evaluate(test_data = data_loaders['test'])

    return


if __name__ == '__main__':
    model_evaluation(path = '/media/msteger/storage/resources/DreamPhant/datasets')
    print 'done'