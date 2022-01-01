# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import requests
import re
import json
import os
import requests
import ipywidgets as widgets

requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

import os
from PIL import Image
from PIL import UnidentifiedImageError

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def add_img_file_extentions(folder_path):
    list_of_files = os.listdir(folder_path)
    for file_name in list_of_files:
        if "." not in file_name:
            file_path = folder_path + "/" + file_name
            try:
                with Image.open(file_path) as img:
                    extention = img.format.lower()
                    new_file_path = file_path + "." + extention
                    if os.path.exists(file_path):
                        os.rename(file_path, new_file_path)
            except UnidentifiedImageError:
                if os.path.exists(file_path):
                    os.remove(file_path)


def search_images_ddg(key, max_n=200):
    """Search for 'key' with DuckDuckGo and return a unique urls of 'max_n' images
    (Adopted from https://github.com/deepanprabhu/duckduckgo-images-api)
    """
    url = 'https://duckduckgo.com/'
    params = {'q': key}
    res = requests.post(url, data=params)
    searchObj = re.search(r'vqd=([\d-]+)\&', res.text)
    if not searchObj: print('Token Parsing Failed !'); return
    requestUrl = url + 'i.js'
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:71.0) Gecko/20100101 Firefox/71.0'}
    params = (
        ('l', 'us-en'), ('o', 'json'), ('q', key), ('vqd', searchObj.group(1)), ('f', ',,,'), ('p', '1'),
        ('v7exp', 'a'))
    urls = []
    while True:
        try:
            res = requests.get(requestUrl, headers=headers, params=params)
            data = json.loads(res.text)
            for obj in data['results']:
                urls.append(obj['image'])
                max_n = max_n - 1
                if max_n < 1:
                    return list(set(urls))  # dedupe
            if 'next' not in data:
                return list(set(urls))
            requestUrl = url + data['next']
        except:
            pass


def download_url(url: str, dest: str, overwrite: bool = False) -> None:
    # Download `url` to `dest` unless is exists and not `overwrite`
    if os.path.exists(dest) and not overwrite: return
    try:
        u = requests.get(url, stream=True, verify=False)
        u = u.raw

        with open(dest, 'wb') as f:

            nbytes, buffer = 0, [1]
            while len(buffer):
                buffer = u.read(8192)
                nbytes += len(buffer)

                f.write(buffer)
    except Exception:
        print("Unable to download:",url)


def download_classes_data(class_1, class_2):
    data_dir = "data"
    total_images = 100
    validation_set = 0.1

    # !rm - rf
    # {data_dir}
    # !mkdir - p
    # {data_dir}
    # !mkdir - p
    # {data_dir} / train / {class_1}
    # !mkdir - p
    # {data_dir} / train / {class_2}
    # !mkdir - p
    # {data_dir} / val / {class_1}
    # !mkdir - p
    # {data_dir} / val / {class_2}

    train_class_1_path = data_dir + "/train/" + class_1
    train_class_2_path = data_dir + "/train/" + class_2

    val_class_1_path = data_dir + "/val/" + class_1
    val_class_2_path = data_dir + "/val/" + class_2

    class_1_imgs = search_images_ddg(class_1, max_n=total_images)
    class_2_imgs = search_images_ddg(class_2, max_n=total_images)

    partition_index = int(validation_set * total_images)

    for i, img_url in enumerate(class_1_imgs[0:partition_index]):
        image_name = str(i)
        dest = val_class_1_path + "/" + image_name
        print(img_url, dest)
        download_url(img_url, dest)
    add_img_file_extentions(val_class_1_path)

    for i, img_url in enumerate(class_2_imgs[0:partition_index]):
        image_name = str(i)
        dest = val_class_2_path + "/" + image_name
        download_url(img_url, dest)
    add_img_file_extentions(val_class_2_path)

    for i, img_url in enumerate(class_1_imgs[partition_index:-1]):
        image_name = str(i)
        dest = train_class_1_path + "/" + image_name
        download_url(img_url, dest)
    add_img_file_extentions(train_class_1_path)

    for i, img_url in enumerate(class_2_imgs[partition_index:-1]):
        image_name = str(i)
        dest = train_class_2_path + "/" + image_name
        download_url(img_url, dest)
    add_img_file_extentions(train_class_2_path)


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def get_trained_model(data_dir='data'):

    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model


    def visualize_model(model, num_images=6):
        was_training = model.training
        model.eval()
        images_so_far = 0
        fig = plt.figure()

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloaders['val']):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images // 2, 2, images_so_far)
                    ax.axis('off')
                    ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                    imshow(inputs.cpu().data[j])

                    if images_so_far == num_images:
                        model.train(mode=was_training)
                        return
            model.train(mode=was_training)


    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in
                   ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))
    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[class_names[x] for x in classes])

    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 2)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)

    model_conv = torchvision.models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)
    model_conv = model_conv.to(device)
    criterion = nn.CrossEntropyLoss()
    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, 25)
    visualize_model(model_conv)

    plt.ioff()
    plt.show()
    return model_conv, class_names


def make_prediction(image_path, model_conv, class_names):
    loader = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    def image_loader(image_name):
        """load image, returns cuda tensor"""
        image = Image.open(image_name)
        image = loader(image).float()
        image = Variable(image, requires_grad=True)
        image = image.unsqueeze(0)  # this is for VGG, may not be needed for ResNet
        return image.to(device)  # assumes that you're using GPU

    input_image = image_loader(image_path)
    outputs = model_conv(input_image)
    _, preds = torch.max(outputs, 1)
    print("Predicted Class:", class_names[preds[0]])
    imshow(input_image.cpu().data[0])


plt.ion()  # interactive mode