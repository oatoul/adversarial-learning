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
from advbox.attacks.localsearch import SinglePixelAttack, LocalSearchAttack


def evaluate_model_for_accuracy(model, device, data_loader):
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    print('\n Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))


def main():

    # file_list = open('file_list.txt', 'r')
    # Lines = file_list.readlines()
    #
    # for line in Lines:
    #     save_folder = 'data/' + line.split(line.strip().split('/',2)[2],1)[0]
    #     if not os.path.exists(save_folder):
    #         makedirs(save_folder)
    #
    #     source = 'project/Materials/data/' + line.strip()
    #     dest = 'data/' + line.strip()
    #     copyfile(source, dest)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    data_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder('data/clean_images', transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            normalize, ])),
        batch_size=64)

    model = load_model('project/Materials/model/model.pt', 'cpu')

    correct = 0
    for i, data in enumerate(data_loader):
        inputs, labels = data
        output = model(inputs)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()

    print('\n Accuracy Before : {}/{} ({:.0f}%)\n'.format(
        correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))




    loss_func = torch.nn.CrossEntropyLoss()
    m = PytorchModel(model, None, (-2.0665, 2.64), channel_axis=0, nb_classes=4)

    device = torch.device("cpu")
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)

        for k in range(data.shape[0]):
            adv_img = np.copy(data[k, :, :])
            original_label = target[k]
            h = 128
            w = 128
            min_ = -2.0665
            max_ = 2.64
            max_pixels = 1000
            pixels = np.random.permutation(h * w)
            pixels = pixels[:max_pixels]
            print(k)
            for i, pixel in enumerate(pixels):
                x = pixel % w
                y = pixel // w

                location = [x, y]

                print("Attack x={0} y={1}".format(x, y))
                channel_axis = 0
                location.insert(channel_axis, slice(None))
                location = tuple(location)

                # 图像经过预处理 取值为整数 通常范围为0-1
                cnt = 0
                for value in np.linspace(min_, max_, num=256):
                    # logger.info("value in [min_={0}, max_={1},step num=256]".format(min_, max_))
                    perturbed = np.copy(adv_img)
                    # 针对图像的每个信道的点[x,y]同时进行修改
                    perturbed[location] = value

                    f = model(torch.from_numpy(perturbed).unsqueeze(0))
                    adv_label = np.argmax(f)

                    cnt += 1
                    # if adversary.try_accept_the_example(adv_img, adv_label):
                    if adv_label != original_label:
                        print(str(cnt) + " found")
                        break


if __name__ == '__main__':
    main()

