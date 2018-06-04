_author__ = 'MSteger'

import torch
from torch import nn
from models.PhantNet import PhantNet, PhantTrain
from torchvision import transforms
from components.preprocessing import loaders, PhantDataset
from sklearn.preprocessing import LabelEncoder
from components.callbacks import MetricTracker, ProgressBar, ModelCheckpoint, TensorBoard
from components import metrics
from functools import partial

def model_evaluation(experiment_name, path = '/media/msteger/storage/resources/tiny-imagenet-200'):

    # setup
    batch_size = 128

    # data
    transformer_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomChoice([
            transforms.RandomHorizontalFlip(p = 1),
            transforms.RandomGrayscale(p = 1),
            transforms.RandomRotation(degrees = [-180, 180]),
            transforms.RandomVerticalFlip(p=1),
            transforms.RandomResizedCrop(224),
            transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.35)

        ]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transformer_val = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data_transformers = {'train': transformer_train, 'val': transformer_val, 'test': transformer_val}
    data_loaders = loaders(path = path, dataset = PhantDataset, transformers = data_transformers, batch_size = batch_size, shuffle = True, num_workers = 4, pin_memory = True)
    # data_organization= tinyImageNet_Prepare(path = path)
    # classes = data_organization.get_classes(classes_lst = None)
    LE = LabelEncoder().fit(range(2))

    # model
    device = torch.device('cuda')
    model = PhantNet(input_shape = (3, 224, 224), num_class = 2)

    # training
    training = PhantTrain(
        model = model.to(device),
        optimizer = torch.optim.Adam(model.classifier[3].parameters(), lr = 0.01),
        loss = nn.CrossEntropyLoss(),
        batch_size = batch_size,
        device = device,
        LE = LE,
        # checkpoint_path = '/media/msteger/storage/resources/DreamPhant/models/{}/2018-06-03 14:36:19.442171__0.313464730978__17.pkl'.format(experiment_name),
        verbose = True
    )
    training.fit(epochs = 500, train_data = data_loaders['train'], val_data = data_loaders['val'], \
                 callbacks = [
                     MetricTracker(metrics = [
                         ('log_loss', metrics.log_loss),
                         ('accuracy_score', metrics.accuracy_score),
                         ('f1_score', partial(metrics.fbeta_score, beta = 2)),
                         # ('sk_accuracy_score', metrics.sk_accuracy_score),
                         # ('sk_f1_weighted', partial(metrics.sk_f1_score, average = 'weighted')),
                         # ('sk_f1_macro', partial(metrics.sk_f1_score, average='macro')),
                     ]),
                     ProgressBar(show_batch_metrics = ['log_loss']),
                     ModelCheckpoint(save_folder_path = r'/media/msteger/storage/resources/DreamPhant/models/{}/'.format(experiment_name), metric = 'log_loss', best_metric_highest = False, verbose = True),
                     TensorBoard(log_dir = r'/media/msteger/storage/resources/DreamPhant/logs/{}/TensorBoard/'.format(experiment_name), update_frequency = 1)
                 ])

    return


if __name__ == '__main__':
    model_evaluation(path = '/media/msteger/storage/resources/DreamPhant/datasets', experiment_name = r'foobar')
    print 'done'