import torch
import numpy as np
import matplotlib.pyplot as plt

import numpy.linalg

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
for j in range(4):
    for k in range(100):


        try:
            ori = torch.load(ori_folder + '/' + str(j) + '/' + str(k) + '.pt')
            adv = torch.load(adv_folder3 + '/' + str(j) + '/' + str(k) + '.pt')

            ori = unorm(torch.from_numpy(ori))
            adv = unorm(torch.from_numpy(adv))

            dist = numpy.linalg.norm(ori - adv)
            sum += dist
            cnt += 1
            print("k = " + str(k) + " dist = " + str(dist))


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
