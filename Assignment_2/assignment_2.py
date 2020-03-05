from __future__ import print_function
import sys
import os
sys.path.append("..")

from load_model_37 import load_model

import numpy as np
import cv2
import torch
import torchvision
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
from shutil import copyfile
from os import makedirs

from advbox.adversary import Adversary
from advbox.models.pytorch import PytorchModel
from advbox.attacks.localsearch import SinglePixelAttack

def main():

    file_list = open('file_list.txt', 'r')
    Lines = file_list.readlines()

    for line in Lines:
        save_folder = 'data/' + line.split(line.strip().split('/',2)[2],1)[0]
        if not os.path.exists(save_folder):
            makedirs(save_folder)

        source = 'project/Materials/data/' + line.strip()
        dest = 'data/' + line.strip()
        copyfile(source, dest)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    data_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder('data/clean_images', transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            normalize, ])),
        batch_size=1)

    model = load_model('project/Materials/model/model.pt', 'cpu')

    for i, data in enumerate(data_loader):
        inputs, labels = data
        inputs, labels = inputs.numpy(), labels.numpy()
        output = model(inputs)
        print(inputs.shape)


    orig = cv2.imread('project/Materials/data/clean_images/artifacts/3791.png')
    img = orig.copy().astype(np.float32)

    tf = transforms.Compose([
            transforms.ToTensor()
        ])

    origTf = tf(orig)

    # plt.imshow(img)
    # plt.show()










    print(help(load_model))
    model = load_model('project/Materials/model/model.pt', 'cpu')

    loss_func = torch.nn.CrossEntropyLoss()

    m = PytorchModel(model, loss_func, (0, 1), channel_axis=1)

    attack = SinglePixelAttack(m)


if __name__ == '__main__':
    main()

