import sys
import os.path as osp
import os
# from google.colab import drive
# drive.mount('/content/drive')
# ROOT = osp.join('/content', 'drive', 'My Drive', 'CS5260')
# sys.path.append(osp.join(ROOT, MATRIC_NUM))

import torch

if torch.cuda.is_available():
    print("GPU is available.")
    device = torch.device('cuda')
else:
    print("Change runtime type to GPU for better performance.")
    device = torch.device('cpu')

import numpy as np
from tqdm.autonotebook import tqdm
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from load_model_37 import load_model


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def predict(model, device, data_loader):
    cnt = 0
    with torch.no_grad():
        for data, target, path in data_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)

            for k in range(len(pred)):
                res = os.path.basename(path[k]).split('.')[0] + '#' + str(pred[k].item())
                print(res)

            cnt += len(data)
            if cnt > 500:
                break


def evaluate_model_for_accuracy(model, device, data_loader):
    correct = 0
    cnt = 0
    with torch.no_grad():
        for data, target, path in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            cnt += len(data)
            if cnt > 500:
                break

    # print('\n Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     correct, len(data_loader.dataset),
    #     100. * correct / len(data_loader.dataset)))

    print('\n Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, cnt,
        100. * correct / cnt))


model = load_model('Materials/model/model.pt', 'cpu')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

clean_data_dir = 'Materials/data/clean_images'
adv_data_dir = 'Materials/data/adv_images'

clean_data_loader = torch.utils.data.DataLoader(
    ImageFolderWithPaths(clean_data_dir, transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        normalize, ])),
    batch_size=256)

adv_data_loader = torch.utils.data.DataLoader(
    ImageFolderWithPaths(adv_data_dir, transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        normalize, ])),
    batch_size=256)

# print("Clean data accuracy:")
# evaluate_model_for_accuracy(model, device, clean_data_loader)
#
# print("Adv data accuracy:")
# evaluate_model_for_accuracy(model, device, adv_data_loader)

predict(model, device, clean_data_loader)
