import os
import numpy as np
import scipy.signal as sig
from PIL import Image
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.optim as optim
from torchdiffeq import odeint
from general import *

def init_seeds(seed=0, deterministic=False):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe

if __name__=="__main__":
    seed = 0
    init_seeds(seed=seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    img_true = Image.open('images/lenna_edge.png')
    img = Image.open('images/lenna.gif')#.convert('RGB')

    img_true = np.array(img_true)[:,:,0]
    img_true = img_true[None, ...] if len(img_true.shape)==2 else img_true.transpose(2, 0, 1)
    img_true = img_true/255
    img_true = torch.tensor(img_true, dtype=torch.float).to(device)

    img = np.array(img)
    img = img[None, ...] if len(img.shape)==2 else img.transpose(2, 0, 1)
    img = img/255
    img = torch.tensor(img, dtype=torch.float).to(device)
    print(f'img_true shape: {img_true.shape}, img_true range: {img_true.min()} ~ {img_true.max()}')
    print(f'img shape: {img.shape}, img range: {img.min()} ~ {img.max()}')

    test = False
    n_iter = 100
    t_max = 10
    t_step_max = 10
    t_step = t_max/t_step_max
    save_path=increment_path(f'images/test_training/')
    loss_curve = []

    net = model(img.shape[-2], img.shape[-2], 1, 3, test=test, t=torch.linspace(0, t_max, t_step_max).to(device)).to(device)
    optimizer = optim.SGD(net.parameters(), lr=5e-4)
    criterion = nn.MSELoss()

    for i in range(n_iter):
        if (i+1)%10 == 0:
            out = net.forward(img, vis_save=True, path_save=get_save_path(save_path, epoch=i, t_max=t_max, t_step_max=t_step_max))
        else:
            out = net.forward(img)
        # print(f'img_out range: {out.min()} ~ {out.max()}')
        loss = torch.mean(torch.abs(img_true - out))
        loss_curve.append(loss.item())
        if (i+1)%10 ==0:
            print(f'epoch:{i+1}, loss:{loss.item()}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # out = net.forward(img, vis_save=True, path_save=get_save_path(save_path, epoch=i, t_max=t_max, t_step_max=t_step_max))
    plt.plot(loss_curve)
    plt.savefig(save_path + f'tmax{t_max}_tstep{t_step_max}_loss.jpg')