_author__ = 'MSteger'

import torch
from torch import nn
from models.PhantNet import PhantNet, PhantTrain
from torchvision import transforms
from components.preprocessing import loaders, PhantDataset
from sklearn.preprocessing import LabelEncoder
from components.callbacks import MetricTracker, ProgressBar
from components import metrics
from functools import partial

def model_evaluation(path = '/media/msteger/storage/resources/tiny-imagenet-200'):

    # setup
    batch_size = 128

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
        verbose = True
    )
    training.fit(epochs = 500, train_data = data_loaders['train'], val_data = data_loaders['val'], \
                 callbacks = [
                     MetricTracker(metrics = [('accuracy_score', metrics.accuracy_score),('sk_accuracy_score', metrics.sk_accuracy_score)]),
                     ProgressBar()
                 ])

    # evaluation
    # training.evaluate(test_data = data_loaders['test'])

    return


if __name__ == '__main__':
    model_evaluation(path = '/media/msteger/storage/resources/DreamPhant/datasets')
    print 'done'