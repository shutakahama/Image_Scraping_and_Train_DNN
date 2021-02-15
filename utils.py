import os
import sys
import time
import datetime
import numpy as np
import torch
import torchvision
import torch.nn as nn
from sklearn.model_selection import train_test_split


def progress_report(count, start_time, batchsize, whole_sample):
    duration = time.time() - start_time
    throughput = count * batchsize / duration
    sys.stderr.write('\r{} updates ({} / {} samples) time: {} ({:.2f} samples/sec)'.format(
        count, count * batchsize, whole_sample, str(datetime.timedelta(seconds=int(duration))), throughput))


def load_images(base_path, categories, test_rate, train_length, test_length, seed):
    folders = os.listdir(base_path)

    train_img = np.array([])
    train_label = np.array([])
    test_img = np.array([])
    test_label = np.array([])

    for folder in folders:
        if folder in categories:
            c = categories[folder]
            folder_name = os.path.join(base_path, folder)
            imgs = os.listdir(folder_name)
            imgs = [im for im in imgs if im.split(".")[-1] in ["jpg", "png"]]
            train_set, test_set = train_test_split(imgs, test_size=test_rate, random_state=seed)
            # print(train_set)
            for path in train_set:
                train_img = np.append(train_img, os.path.join(folder_name, path))
                train_label = np.append(train_label, c)
            for path in test_set:
                test_img = np.append(test_img, os.path.join(folder_name, path))
                test_label = np.append(test_label, c)

    if train_length > 0:
        perm = np.random.permutation(len(train_img))
        train_img = train_img[perm][:train_length]
        train_label = train_label[perm][:train_length]
    if test_length > 0:
        perm = np.random.permutation(len(test_img))
        test_img = test_img[perm][:test_length]
        test_label = test_label[perm][:train_length]

    return [train_img, train_label], [test_img, test_label]


def load_model(model_type, num_class):
    if model_type == 'inception':
        model = torch.hub.load('pytorch/vision:v0.6.0', 'googlenet', pretrained=True)
        model.fc = nn.Linear(1024, num_class)
    elif model_type == 'squeezenet':
        model = torchvision.models.squeezenet1_1(pretrained=True)
        model.classifier[1] = nn.Conv2d(512, num_class, kernel_size=(1, 1), stride=(1, 1))
    elif model_type == 'vgg':
        model = torchvision.models.vgg19(pretrained=True)
        model.classifier[6] = nn.Linear(4096, num_class)
    elif model_type == 'resnet':
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = nn.Linear(512, num_class)
    elif model_type == 'mobilenet':
        model = torchvision.models.mobilenet_v2(pretrained=True)
        model.classifier[1] = torch.nn.Linear(1280, num_class)
    else:
        raise NameError

    return model

