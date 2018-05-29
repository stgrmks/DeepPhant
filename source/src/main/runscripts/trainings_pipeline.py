_author__ = 'MSteger'

import torch
from torch import nn
from models.PhantNet import PhantNet, PhantTrain
from torchvision import transforms
from components.preprocessing import loaders, tinyImageNet, tinyImageNet_Prepare
from sklearn.preprocessing import LabelEncoder
from components.metrics import accuracy_score, sk_accuracy_score


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
    data_loaders = loaders(path = path, dataset = tinyImageNet, transformers = data_transformers, batch_size = batch_size, shuffle = True, num_workers = 4, pin_memory = True)
    data_organization= tinyImageNet_Prepare(path = path)
    classes = data_organization.get_classes(classes_lst = None)
    LE = LabelEncoder().fit(classes.keys())

    # model
    device = torch.device('cuda')
    model = PhantNet(input_size = (3, 224, 224), num_class = len(classes))
    training = PhantTrain(
        model = model.to(device),
        optimizer = torch.optim.Adam(model.classifier[3].parameters(), lr = 0.01),
        loss = nn.CrossEntropyLoss(),
        batch_size = batch_size,
        device = device,
        LE = LE,
        print_freq = 10,
        metrics = [('accuracy_score', accuracy_score), ('sk_accuracy_score', sk_accuracy_score)],#, ('precision_weighted', partial(metrics.precision_score, average = 'weighted')), ('f1_weighted', partial(metrics.f1_score, average = 'weighted'))]
        verbose = True
    )
    training.fit(epochs = 10, train_data = data_loaders['train'], val_data = data_loaders['val'])
    training.evaluate(test_data = data_loaders['test'])

    return


if __name__ == '__main__':
    model_evaluation(path = '/media/msteger/storage/resources/tiny-imagenet-200')
    print 'done'