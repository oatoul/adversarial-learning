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
import matplotlib.pyplot as plt


class RandomPad(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, pad_range, fill = 0):
        assert isinstance(pad_range, (int, tuple))
        if isinstance(fill, int):
            self.fill = fill
        if isinstance(pad_range, int):
            self.pad_range = (pad_range, pad_range)
        else:
            assert len(pad_range) == 2
            self.pad_range = pad_range

    def __call__(self, sample):
        # image, landmarks = sample['image'], sample['landmarks']

        # h, w = image.shape[:2]
        fill = self.fill
        min, max = self.pad_range

        pad = np.random.randint(min, max)

        top = np.random.randint(0, pad + 1)
        left = np.random.randint(0, pad + 1)
        bottom = pad - top
        right = pad - left

        tf = transforms.Compose([
            transforms.Pad(padding=(left, top, right, bottom), fill=fill, padding_mode='constant')
            ])

        # landmarks = landmarks + [pad, pad]

        # return {'image': image, 'landmarks': landmarks}
        return tf(sample)


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


def predict(model, device, data_loader, filename, mode):
    cnt = 0
    outF = open(filename, mode)
    with torch.no_grad():
        for data, target, path in data_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)

            for k in range(len(pred)):
                res = str(os.path.basename(path[k]).split('.')[0]) + '#' + str(pred[k].item()) + '\n'
                outF.write(res)

            cnt += len(data)

    print(str(cnt) + " records written to " + filename)
    outF.close()


def create_output(device, data_loader, filename):
    cnt = 0
    outF = open(filename, "w")
    with torch.no_grad():
        for data, target, path in data_loader:
            data, target = data.to(device), target.to(device)
            for k in range(len(target)):
                res = str(os.path.basename(path[k]).split('.')[0]) + '#' + str(target[k].item()) + '\n'
                outF.write(res)
                cnt += 1

    print(str(cnt) + " records writen to " + str(filename))
    outF.close()


def evaluate_with_voting(model, device, data_loader, vote):
    correct = 0
    cnt = 0
    pred_vote = 0
    label = 0
    for k in range(vote):
        for data, target, path in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred_cur = output.argmax(dim=1, keepdim=True)
            if isinstance(pred_vote, int):
                pred_vote = F.one_hot(pred_cur, num_classes=4)
            else:
                pred_vote += F.one_hot(pred_cur, num_classes=4)
            if isinstance(label, int):
                label = target

    pred_res = pred_vote.squeeze().argmax(dim=1, keepdim=True)
    cnt += len(label)
    correct += pred_res.eq(label.view_as(pred_res)).sum().item()

    print('\n{}/{} ({:.0f}%)'.format(
        correct, cnt,
        100. * correct / cnt))


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
            # if cnt > 500:
            #     break

    # print('\n Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     correct, len(data_loader.dataset),
    #     100. * correct / len(data_loader.dataset)))

    print('\n{}/{} ({:.0f}%)'.format(
        correct, cnt,
        100. * correct / cnt))


model = load_model('Materials/model/model.pt', 'cpu')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

clean_data_dir = 'Materials/data/subset/clean_images'
adv_data_dir = 'Materials/data/subset/adv_images'


# resampling filters
NEAREST = NONE = 0
BOX = 4
BILINEAR = LINEAR = 2
HAMMING = 5
BICUBIC = CUBIC = 3
LANCZOS = ANTIALIAS = 1


pad_range_min = 2
pad_range_max = 10
fill = -1
vote = 30
h_flip = 0.5
v_flip = 1
interp = HAMMING

print("Pad " + str(pad_range_min) + " ~ " + str(pad_range_max) + " with fill = " + str(fill))
print("vote " + str(vote) + " h_flip = " + str(h_flip) + " with v_flip = " + str(v_flip) + " interpolation " + str(interp))

clean_data_loader = torch.utils.data.DataLoader(
    ImageFolderWithPaths(clean_data_dir, transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        RandomPad(pad_range=(pad_range_min, pad_range_max), fill=fill),
        transforms.Resize(size=128, interpolation=interp),
        transforms.RandomHorizontalFlip(p=h_flip),
        transforms.RandomVerticalFlip(p=v_flip),
        transforms.ToTensor(),
        normalize, ])),
    batch_size=400)

adv_data_loader = torch.utils.data.DataLoader(
    ImageFolderWithPaths(adv_data_dir, transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        RandomPad(pad_range=(pad_range_min, pad_range_max), fill=fill),
        transforms.Resize(size=128, interpolation=interp),
        transforms.RandomHorizontalFlip(p=h_flip),
        transforms.RandomVerticalFlip(p=v_flip),
        transforms.ToTensor(),
        normalize, ])),
    batch_size=400)

print("Clean data accuracy:")
evaluate_with_voting(model, device, clean_data_loader, vote)

print("Adv data accuracy:")
evaluate_with_voting(model, device, adv_data_loader, vote)


# print("Clean data accuracy:")
# evaluate_model_for_accuracy(model, device, clean_data_loader)
# evaluate_model_for_accuracy(model, device, clean_data_loader)
# evaluate_model_for_accuracy(model, device, clean_data_loader)
#
# print("Adv data accuracy:")
# evaluate_model_for_accuracy(model, device, adv_data_loader)
# evaluate_model_for_accuracy(model, device, adv_data_loader)
# evaluate_model_for_accuracy(model, device, adv_data_loader)

# create_output(device, clean_data_loader, "clean.txt")
# create_output(device, adv_data_loader, "adv.txt")

# predict(model, device, clean_data_loader, "result.txt", "w")
# predict(model, device, adv_data_loader, "result.txt", "a")

# predict(model, device, clean_data_loader, "result_RandomPad.txt", "w")
# predict(model, device, adv_data_loader, "result_RandomPad.txt", "a")


