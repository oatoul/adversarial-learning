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

# def softmax(x):
#     e_x = np.exp(x - np.max(x))
#     return e_x / e_x.sum()

def softmax(logits):
    assert logits.ndim == 1

    # for numerical reasons we subtract the max logit
    # (mathematically it doesn't matter!)
    # otherwise exp(logits) might become too large or too small
    logits = logits - np.max(logits)
    e = np.exp(logits)
    return e / np.sum(e)

def main():

    # np.random.seed(seed=128)

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
        batch_size=100)

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


    device = torch.device("cpu")

    for data, target in data_loader:
        data, target = data.to(device), target.to(device)

        min_ = data.min().numpy()
        max_ = data.max().numpy()
        print("Min " + str(min_) + " Max " + str(max_))

        # 正则化到[-0.5,0.5]区间内
        def normalize(im):
            im = im - (min_ + max_) / 2
            im = im / (max_ - min_)

            LB = -1 / 2
            UB = 1 / 2
            return im, LB, UB

        def unnormalize(im):
            im = im * (max_ - min_)
            im = im + (min_ + max_) / 2
            return im

        h = 128
        w = 128
        # min_ = -2.0665
        # max_ = 2.64
        r = 1.5
        p = 1.0
        d = 5
        t = 128
        R = 150
        max_pt = 1024

        for k in range(data.shape[0]):
            adv_img = np.copy(data[k, :, :])
            ori_img_save = np.copy(data[k, :, :])
            original_label = target[k]

            print("Pert " + str(t) + " pts each round")
            assert 0 <= r <= 2

            # 归一化
            adv_img, LB, UB = normalize(adv_img)

            channels = 3

            # 随机选择一部分像素点 总数不超过全部的10% 最大为128个点
            def random_locations():
                n = int(0.1 * h * w)
                n = min(n, max_pt)
                print("Select " + str(n) + " random locations")
                locations = np.random.permutation(h * w)[:n]
                p_x = locations % w
                p_y = locations // w
                pxy = list(zip(p_x, p_y))
                pxy = np.array(pxy)
                return pxy

            # 针对图像的每个信道的点[x,y]同时进行修改 修改的值为p * np.sign(Im[location]) 类似FGSM的一次迭代
            # 不修改Ii的图像 返回修改后的图像
            channel_axis = 0
            def pert(Ii, p, x, y):
                Im = Ii.copy()
                location = [x, y]
                location.insert(channel_axis, slice(None))
                location = tuple(location)
                Im[location] = p * np.sign(Im[location])
                return Im

            #截断 确保assert LB <= r * Ibxy <= UB 但是也有可能阶段失败退出 因此可以适当扩大配置的原始数据范围
            # 这块的实现没有完全参考论文
            def cyclic(r, Ibxy):

                result = r * Ibxy

                if result < LB:
                    result = result + (UB - LB)
                elif result > UB:
                    result = result - (UB - LB)

                # result=result.clip(LB,UB)
                assert LB <= result <= UB
                return result

            Ii = adv_img
            PxPy = random_locations()

            terminate = False
            # 循环攻击轮
            try_time = 0
            while try_time < R and not terminate:
            # for try_time in range(R) and not terminate:
                try_time += 1
                # 重新排序 随机选择不不超过128个点
                PxPy = PxPy[np.random.permutation(len(PxPy))[:max_pt]]
                L = [pert(Ii, p, x, y) for x, y in PxPy]

                # 批量返回预测结果 获取原始图像标签的概率
                def score(Its):
                    Its = np.stack(Its)
                    Its = unnormalize(Its)
                    batch_logits = model.forward(torch.from_numpy(Its))
                    # scores = [softmax(logits.numpy())[original_label] for logits in batch_logits]
                    scores = [logits.numpy()[original_label] for logits in batch_logits]
                    return scores

                # 选择影响力最大的t个点进行扰动 抓主要矛盾 这里需要注意 np.argsort是升序 所以要取倒数的几个
                scores = score(L)

                indices = np.argsort(scores)[-t:]
                print("try {0} times ".format(try_time))
                # print("try {0} times  selected pixel indices:{1}".format(try_time, str(indices)))

                PxPy_star = PxPy[indices]

                # Generation of new perturbed input Ii
                for x, y in PxPy_star:
                    # 每个颜色通道的[x，y]点进行扰动并截断 扰动算法即放大r倍
                    for b in range(channels):
                        location = [x, y]
                        location.insert(channel_axis, b)
                        location = tuple(location)
                        Ii[location] = cyclic(r, Ii[location])

                # Check whether the perturbed input Ii is an adversarial input
                f = model.forward(torch.from_numpy(unnormalize(Ii)).unsqueeze(0)).numpy()[0]
                print(f)
                adv_label = np.argmax(f)
                adv_label_pro = softmax(f)[adv_label]

                # print("adv_label={0}".format(adv_label))
                print("adv_label= " + str(adv_label) + " pro=" + str(adv_label_pro))

                if adv_label != original_label:
                    print("found")
                    terminate = True

                    ori_folder = 'original/' + str(original_label.numpy()) + '/'
                    adv_folder = 'adversarial/' + str(original_label.numpy()) + '/'

                    if not os.path.exists(ori_folder):
                        makedirs(ori_folder)

                    if not os.path.exists(adv_folder):
                        makedirs(adv_folder)

                    fileName = str(k) + '.pt'
                    torch.save(ori_img_save, ori_folder + fileName)
                    torch.save(unnormalize(Ii), adv_folder + fileName)



                # 扩大搜索范围，把原有点周围2d乘以2d范围内的点都拉进来 去掉超过【w，h】的点
                # "{Update a neighborhood of pixel locations for the next round}"

                PxPy = [
                    (x, y)
                    for _a, _b in PxPy_star
                    for x in range(_a - d, _a + d + 1)
                    for y in range(_b - d, _b + d + 1)]
                PxPy = [(x, y) for x, y in PxPy if 0 <= x < w and 0 <= y < h]
                PxPy = list(set(PxPy))
                PxPy = np.array(PxPy)



if __name__ == '__main__':
    main()

