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

from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients

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
    item = np.load(input)/255.0
    return Image.fromarray(item)

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
        torchvision.datasets.DatasetFolder('adv_images', #Change this to your adv_images folder
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
            if len(given_dataset) ==0:
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

    #Checking the range of generated images
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


    #Checking for equation 2 and equation 3
    if all([x > 0.8 for x in softmax_values.tolist()]):
        print('Softmax values for all of your adv images are greater than 0.8')
        S = 0
        for i in range(10):
            label_indices = np.where(labels==i)[0]
            a_i = adv_images[label_indices, :, :]
            for k in range(10):
                image = a_i[k, :, :]
                S = S + np.min(
                            np.sqrt(
                                np.sum(
                                    np.square(
                                        np.subtract(given_dataset, np.tile(np.expand_dims(image, axis=0), [1000,1,1]))
                                    ),axis=(1,2))))

        print('Value of S : {:.4f}'.format(S / 100))

    else:
        print('Softmax values for some of your adv images are less than 0.8')



def generate_adv_images():
    adv_images = []
    targeted_class_labels = []
    image_names = []
    #your code to generate adv_images goes here
    return adv_images,image_names,targeted_class_labels

def deepfool(image, net, num_classes=10, overshoot=0.02, max_iter=50):

    """
    https://github.com/LTS4/DeepFool/blob/master/Python/test_deepfool.py
       :param image: Image of size HxWx3
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        print("Using GPU")
        image = image.cuda()
        net = net.cuda()
    else:
        print("Using CPU")


    f_image = net.forward(Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]

    I = I[0:num_classes]
    label = I[0]

    input_shape = image.cpu().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = Variable(pert_image[None, :], requires_grad=True)
    fs = net.forward(x)
    fs_list = [fs[0,I[k]] for k in range(num_classes)]
    k_i = label

    while k_i == label and loop_i < max_iter:

        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        for k in range(1, num_classes):
            zero_gradients(x)

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i =  (pert+1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        if is_cuda:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).cuda()
        else:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot)

        x = Variable(pert_image, requires_grad=True)
        fs = net.forward(x)
        k_i = np.argmax(fs.data.cpu().numpy().flatten())

        loop_i += 1

    r_tot = (1+overshoot)*r_tot

    return r_tot, loop_i, label, k_i, pert_image

def main():
    # Settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--no-cuda', action='store_true', default=False,help='disables CUDA training')
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
    data_loader =  torch.utils.data.DataLoader(
        torchvision.datasets.DatasetFolder('data',
                                           loader= numpy_loader,
                                           extensions= '.npy',
                                           transform=transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize(mean, std)])),
                                           batch_size=args.batch_size, **kwargs)

    model = Net().to(device)

    model.load_state_dict(torch.load(args.model_path))

    evaluate_model_for_accuracy(model, device, data_loader)

    adv_images,image_names,class_labels = generate_adv_images(model, )
    #Implement this method to generate adv images
    #statisfying constraints mentioned in the assignment discription

    save_folder = 'adv_images'

    for image,image_name,class_label in zip(adv_images,image_names,class_labels):
        for t, m, s in zip(image, mean, std):
            t.mul_(s).add_(m)

        image_to_save = image.squeeze().detach().cpu().numpy()
        image_to_save = 255.0 * image_to_save

        if np.min(image_to_save) < 0 or np.max(image_to_save) > 255:
            print('Generated adversarial image is out of range.')
            sys.exit()

        if not os.path.exists(os.path.join(save_folder,str(class_label))):
            makedirs(os.path.join(save_folder,str(class_label)))

        np.save(os.path.join(save_folder,str(class_label),image_name), image_to_save)

    evaluate_adv_images(model,device,kwargs,mean,std,data_loader)


if __name__ == '__main__':
    main()
