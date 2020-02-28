from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import os
from torch.utils import data
from os import makedirs
import torchvision
from PIL import Image
import sys
import copy

from Assignment_1.deepfool import deepfool
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


def numpy_loader(input):
    item = np.load(input) / 255.0
    return Image.fromarray(item)


def clip_tensor(A, minv, maxv):
    A = torch.max(A, minv * torch.ones(A.shape))
    A = torch.min(A, maxv * torch.ones(A.shape))
    return A


def evaluate_model_for_accuracy(model, device, data_loader):
    model.eval()

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


def evaluate_adv_images(model, device, kwargs, mean, std, data_loader):
    batch_size = 100
    model.eval()

    adv_data_loader = torch.utils.data.DataLoader(
        torchvision.datasets.DatasetFolder('adv_images',  # Change this to your adv_images folder
                                           loader=numpy_loader,
                                           extensions='.npy',
                                           transform=transforms.Compose([transforms.ToTensor(),
                                                                         transforms.Normalize(mean, std)])),
        batch_size=batch_size, **kwargs)

    evaluate_model_for_accuracy(model, device, adv_data_loader)

    given_dataset = []
    adv_images = []
    labels = []
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            if len(given_dataset) == 0:
                given_dataset = data.squeeze().detach().cpu().numpy()
            else:
                given_dataset = np.concatenate([given_dataset, data.squeeze().detach().cpu().numpy()],
                                               axis=0)

        for data, target in adv_data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            label = target.squeeze().detach().cpu().numpy()
            softmax_values = torch.nn.Softmax()(output).cpu().numpy()[np.arange(batch_size), label]
            adv_images = data
            labels = target

    # Checking the range of generated images
    adv_images_copy = copy.deepcopy(adv_images)
    for k in range(adv_images_copy.shape[0]):
        image_ = adv_images_copy[k, :, :]

        for t, m, s in zip(image_, mean, std):
            t.mul_(s).add_(m)

        image = image_.squeeze().detach().cpu().numpy()
        image = 255.0 * image

        if np.min(image) < 0 or np.max(image) > 255:
            print('Generated adversarial image is out of range.')
            sys.exit()

    adv_images = adv_images.squeeze().detach().cpu().numpy()
    labels = labels.squeeze().detach().cpu().numpy()

    # Checking for equation 2 and equation 3
    if all([x > 0.8 for x in softmax_values.tolist()]):
        print('Softmax values for all of your adv images are greater than 0.8')
        S = 0
        for i in range(10):
            label_indices = np.where(labels == i)[0]
            a_i = adv_images[label_indices, :, :]
            for k in range(10):
                image = a_i[k, :, :]
                S = S + np.min(
                    np.sqrt(
                        np.sum(
                            np.square(
                                np.subtract(given_dataset, np.tile(np.expand_dims(image, axis=0), [1000, 1, 1]))
                            ), axis=(1, 2))))

        print('Value of S : {:.4f}'.format(S / 100))

    else:
        print('Softmax values for some of your adv images are less than 0.8')

def generate_adv_images(mean, std, model, device, data_loader, kwargs):
    adv_images = []
    targeted_class_labels = []
    image_names = []
    # your code to generate adv_images goes here
    #
    # adv_images_all = []
    # targeted_class_labels_all = []
    # image_names_all = []
    #
    # cost = []
    # given_dataset = []
    # with torch.no_grad():
    #     for data, target in data_loader:
    #         data, target = data.to(device), target.to(device)
    #         if len(given_dataset) == 0:
    #             given_dataset = data.squeeze().detach().cpu().numpy()
    #         else:
    #             given_dataset = np.concatenate([given_dataset, data.squeeze().detach().cpu().numpy()],
    #                                            axis=0)
    #
    # for data, target in data_loader:
    #     data, target = data.to(device), target.to(device)
    #     max_it = 50
    #     for k in range(data.shape[0]):
    #         image_ = data[k, :, :]
    #
    #         softmax_t, r, loop_i, label_orig, label_pert, pert_image = deepfool(image=image_, net=model.eval(),
    #                                                                             max_iter=max_it)
    #
    #         if softmax_t > 0.8:
    #             adv_images_all.append(pert_image)
    #             targeted_class_labels_all.append(label_pert)
    #             image_names_all.append(str(k) + "_" + str(label_orig))
    #             cost.append(calculate_cost(pert_image, given_dataset))
    #             # for t, m, s in zip(pert_image, mean, std):
    #             #     t.mul_(s).add_(m)
    #             # pert_image = clip_tensor(pert_image, 0, 1).squeeze().detach().cpu().numpy()
    #             # pert_image = pert_image.squeeze().detach().cpu().numpy()
    #             # pert_image = 255.0 * pert_image
    #             #
    #             # fpath = "adv_images/" + str(label_pert) + "/" + str(k)
    #             # np.save(fpath, pert_image)
    #         # if adv_images.__len__() >= 10:
    #         #     break
    #         print(str(k))
    #
    # save_data("adv_images_all", adv_images_all, image_names_all, targeted_class_labels_all, std, mean)



    # batch_size = adv_images_all.__len__()
    batch_size = 857

    adv_images, image_names, targeted_class_labels = select_adv_images(mean,std,device,data_loader,batch_size,kwargs)

    # for i in range(10):
    #     ind_i = np.where(targeted_class_labels == i).ravel()
    #     cost_i = cost[ind_i]
    #     ind_max = np.argpartition(cost_i, -10)[-10:]
    #     ind_i_max = ind_i[ind_max]
    #     adv_images.append(adv_images_all[ind_i_max])
    #     image_names.append(image_names_all[ind_i_max])
    #     targeted_class_labels.append(targeted_class_labels_all[ind_i_max])

    return adv_images, image_names, targeted_class_labels


def select_adv_images(mean, std, device, data_loader, batch_size, kwargs):
    adv_images_sel = []
    targeted_class_labels_sel = []
    image_names_sel = []

    adv_data_loader_all = torch.utils.data.DataLoader(
        torchvision.datasets.DatasetFolder('adv_images_all',  # Change this to your adv_images folder
                                           loader=numpy_loader,
                                           extensions='.npy',
                                           transform=transforms.Compose([transforms.ToTensor(),
                                                                         transforms.Normalize(mean, std)])),
        batch_size=batch_size, **kwargs)


    given_dataset = []
    adv_images = []
    labels = []

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            if len(given_dataset) == 0:
                given_dataset = data.squeeze().detach().cpu().numpy()
            else:
                given_dataset = np.concatenate([given_dataset, data.squeeze().detach().cpu().numpy()],
                                               axis=0)

        for data, target in adv_data_loader_all:
            data, target = data.to(device), target.to(device)
            adv_images = data
            labels = target

    for i in range(10):
        label_indices = np.where(labels == i)[0]
        a_i = adv_images[label_indices, :, :]

        cost_i = []
        for k in range(a_i.__len__()):
            image = a_i[k, :, :]
            cost_k = calculate_cost(image, given_dataset)
            cost_i.append(cost_k)

        ind = np.argpartition(cost_i, -10)[-10:]
        # ind = np.argpartition(cost_i, 10)[:10]
        selected_images_i = a_i[ind, :, :]

        adv_images_sel.extend(selected_images_i)
        targeted_class_labels_sel.extend(np.array([i]*10))
        image_names_sel.extend(np.array(['0','1','2','3','4','5','6','7','8','9']))

    return adv_images_sel, image_names_sel, targeted_class_labels_sel




def calculate_cost(image, given_dataset):
    image_ = image.squeeze().detach().cpu().numpy()
    return np.min(np.sqrt(np.sum(np.square(
        np.subtract(given_dataset, np.tile(np.expand_dims(image_, axis=0), [1000, 1, 1]))), axis=(1, 2))))


def save_data(save_folder, images, image_names, class_labels, std, mean):
    for image, image_name, class_label in zip(images, image_names, class_labels):
        for t, m, s in zip(image, mean, std):
            t.mul_(s).add_(m)

        image_to_save = image.squeeze().detach().cpu().numpy()
        image_to_save = 255.0 * image_to_save

        if np.min(image_to_save) < 0 or np.max(image_to_save) > 255:
            print('Generated adversarial image is out of range.')
            sys.exit()

        if not os.path.exists(os.path.join(save_folder, str(class_label))):
            makedirs(os.path.join(save_folder, str(class_label)))

        np.save(os.path.join(save_folder, str(class_label), image_name), image_to_save)


def main():
    # Settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=1000)
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--model_path', type=str, default='model/mnist_cnn.pt')
    parser.add_argument('--data_folder', type=str, default='data')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    mean = (0.1307,)
    std = (0.3081,)

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    data_loader = torch.utils.data.DataLoader(
        torchvision.datasets.DatasetFolder('data',
                                           loader=numpy_loader,
                                           extensions='.npy',
                                           transform=transforms.Compose([transforms.ToTensor(),
                                                                         transforms.Normalize(mean, std)])),
        batch_size=args.batch_size, **kwargs)

    model = Net().to(device)

    model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))

    evaluate_model_for_accuracy(model, device, data_loader)

    adv_images, image_names, class_labels = generate_adv_images(mean, std, model, device, data_loader, kwargs)
    # Implement this method to generate adv images
    # statisfying constraints mentioned in the assignment discription

    save_folder = 'adv_images'

    for image, image_name, class_label in zip(adv_images, image_names, class_labels):
        for t, m, s in zip(image, mean, std):
            t.mul_(s).add_(m)

        image_to_save = image.squeeze().detach().cpu().numpy()
        image_to_save = 255.0 * image_to_save

        if np.min(image_to_save) < 0 or np.max(image_to_save) > 255:
            print('Generated adversarial image is out of range.')
            sys.exit()

        if not os.path.exists(os.path.join(save_folder, str(class_label))):
            makedirs(os.path.join(save_folder, str(class_label)))

        np.save(os.path.join(save_folder, str(class_label), image_name), image_to_save)

    evaluate_adv_images(model, device, kwargs, mean, std, data_loader)


if __name__ == '__main__':
    main()
