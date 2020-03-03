import numpy as np
from torch.autograd import Variable
import torch as torch
import copy
from torch.autograd.gradcheck import zero_gradients


def deepfool(image, net, num_classes=10, overshoot=0.02, max_iter=50):

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

    softmax_t_prev = 0
    r_tot_prev = 0
    loop_i_prev = 0
    label_prev = 0
    k_i_prev = 0
    pert_image_prev = 0
    found = False
    terminate = False

    while terminate == False and loop_i < max_iter:

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

            pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot and add 1e-4 for numerical stability
        r_i = (pert + 1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        pert_image = image + (1 + overshoot) * torch.from_numpy(r_tot)

        # clip
        pert_image = clip_tensor(pert_image, -0.4242, 2.8214)

        x = Variable(pert_image, requires_grad=True)
        fs = net.forward(x)

        k_i = np.argmax(fs.data.cpu().numpy().flatten())
        softmax_t = torch.nn.Softmax()(fs.data.cpu()).numpy().flatten()[k_i]

        loop_i += 1

        cur_notFound = False

        if k_i != label and softmax_t > 0.8:
            found = True
        else:
            cur_notFound = True

        if found and cur_notFound:
            terminate = True

        if not terminate:
            softmax_t_prev = softmax_t
            r_tot_prev = r_tot
            loop_i_prev = loop_i
            label_prev = label
            k_i_prev = k_i
            pert_image_prev = pert_image


    r_tot_prev = (1 + overshoot) * r_tot_prev

    return softmax_t_prev, r_tot_prev, loop_i_prev, label_prev, k_i_prev, pert_image_prev


def clip_tensor(A, minv, maxv):
    A = torch.max(A, minv * torch.ones(A.shape))
    A = torch.min(A, maxv * torch.ones(A.shape))
    return A
