import torch
import numpy as np
import matplotlib.pyplot as plt
import os

import numpy.linalg
from os import makedirs

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

mean=(0.485, 0.456, 0.406)
std=(0.229, 0.224, 0.225)


ori_folder = 'p=1,t=128/original/'
adv_folder1 = 'p=1,t=10/adversarial/'
adv_folder2 = 'p=1,t=128/adversarial/'
adv_folder3 = 't=128,r=0.75/adversarial/'

sum = 0
cnt = 0

def tf(b):
    a = b
    a = unorm(torch.from_numpy(a)).numpy()
    a = np.moveaxis(a, 0, 2)
    a = torch.FloatTensor(a)
    return a

for j in range(4):
    for k in range(100):
        ori = 0
        adv = 0
        dist = 0

        try:
            ori = torch.load(ori_folder + '/' + str(j) + '/' + str(k) + '.pt')
            ori = tf(ori)
            plt.imshow(ori)
            plt.show()
        except:
            pass

        try:
            adv = torch.load(adv_folder2 + '/' + str(j) + '/' + str(k) + '.pt')
            adv = tf(adv)
            dist = numpy.linalg.norm(ori - adv)
        except:
            pass

        try:
            adv1 = torch.load(adv_folder1 + '/' + str(j) + '/' + str(k) + '.pt')
            adv1 = tf(adv1)
            dist1 = numpy.linalg.norm(ori - adv1)
            if dist1 < dist:
                adv = adv1
                dist = dist1
        except:
            pass

        try:
            adv3 = torch.load(adv_folder3 + '/' + str(j) + '/' + str(k) + '.pt')
            adv3 = tf(adv3)
            dist3 = numpy.linalg.norm(ori - adv3)
            if dist3 < dist:
                adv = adv3
                dist = dist3
        except:
            pass

        try:
            if dist > 0:
                sum += dist
                cnt += 1
                print("k = " + str(k) + " dist = " + str(dist))

                ori_res = 'result/original/' + str(j) + '/'
                adv_res = 'result/adversarial/' + str(j) + '/'

                if not os.path.exists(ori_res):
                    makedirs(ori_res)

                if not os.path.exists(adv_res):
                    makedirs(adv_res)

                fileName = str(k) + '.pt'
                torch.save(ori, ori_res + fileName)
                torch.save(adv, adv_res + fileName)

                # ori = unorm(torch.from_numpy(ori))
                # adv = unorm(torch.from_numpy(adv))
                #
                # abc = ori.numpy().transpose()
                # plt.imshow(abc)
                # plt.show()
                #
                # abc = adv.numpy().transpose()
                # plt.imshow(abc)
                # plt.show()
        except:
            pass

if cnt != 100:
    print("Count = " + str(cnt))

print("Ave = " + str(sum/cnt))

# for t, m, s in zip(ori, mean, std):
#     t.mul_(s).add_(m)
#
# for t, m, s in zip(adv, mean, std):
#     t.mul_(s).add_(m)

print("done")
