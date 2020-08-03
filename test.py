import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd

from utils.train import training
from pamap2.utils import PAMAP2
from models.dim1.vgg import vgg11 
from models.dim1.resnet import resnet34
from models.dim1.densenet import densenet121

def get_dataset(train_persons, test_persons, fast_channel=True):
    ret = pamap2.framing(frame_size, train_persons, activities, attributes, positions, axes)
    x_train, y_train, sub_labels, cid2act, pid2name = ret
    if fast_channel:
        x_train = np.transpose(x_train, [0, 2, 1])
    print('Train: ', cid2act)
    flg = False
    for lid in range(len(activities)):
        if lid not in y_train:
            flg = True 
            print(' >>> [Warning] Subjects(label id {}) not found in train dataset'.format(lid))
    if flg:
        raise RuntimeError('Subject are not enough.')

    ret = pamap2.framing(frame_size, test_persons, activities, attributes, positions, axes)
    x_test, y_test, sub_labels, cid2act, pid2name = ret
    if fast_channel:
        x_test= np.transpose(x_test, [0, 2, 1])
    print('Test: ', cid2act)
    flg = False
    for lid in range(len(activities)):
        if lid not in y_train:
            flg = True 
            print(' >>> [Warning] Subjects(label id {}) not found in train dataset'.format(lid))
    if flg:
        raise RuntimeError('Subject are not enough.')

    train_ds = torch.utils.data.TensorDataset(torch.tensor(x_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.long))
    test_ds = torch.utils.data.TensorDataset(torch.tensor(x_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.long))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader= torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    print('in_shape: {}'.format(x_test.shape[1:]))

    return train_loader, test_loader

frame_size = 256
activities = [1, 2, 3, 4, 5]
attributes = ['acc1']
positions = ['hand', 'chest', 'ankle']
axes = ['x', 'y', 'z']
all_persons = np.array([
    'subject101', 'subject102', 'subject103',
    'subject104', 'subject105', 'subject106',
    'subject107', 'subject108', #'subject109',
])
d_model = len(positions) * len(axes)
n_classes = len(activities)

n_epochs = 2
batch_size = 256
lr = 1e-2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: {}'.format(device))

# Datasets
pamap2 = PAMAP2('E:/datasets/PAMAP2/PAMAP2_Dataset/Protocol/', cache_dir='pamap2/data_cache/org/')

model_list = {
    'vgg11': vgg11,
    'resnet34': resnet34,
    'densenet121': densenet121,
}

param_list = {
    'vgg11': {'lr': 1e-5, 'scheduler': None},
    'resnet34': {'lr': 1e-5, 'scheduler': None},
    'densenet121': {'lr': 1e-5, 'scheduler': None},
}

for test_person in all_persons:
    train_persons = all_persons[all_persons != test_person]
    test_persons = np.array([test_person])

    print('='*100)
    print('Train persons: {}'.format(train_persons))
    print('Test persons: {}'.format(test_persons))

    for model_name in model_list:
        print('[{}]'.format(model_name))
        model = model_list[model_name](in_channels=d_model, num_classes=n_classes).to(device)
        param = param_list[model_name]

        if 'transformer' in model_name: flg = False 
        else: flg = True 
        train_loader, test_loader = get_dataset(train_persons, test_persons, fast_channel=flg)

        # Training
        criterion = nn.CrossEntropyLoss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=param['lr'])
        # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9, last_epoch=-1)
        scheduler = None

        hist = training(model, train_loader, test_loader, n_epochs, criterion, optimizer, scheduler, device=device)

        pd.DataFrame(hist).to_csv('history_{}_test-{}.csv'.format(model_name, test_person))



