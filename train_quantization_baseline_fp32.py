import os
import numpy as np
import scipy.signal as sig
from PIL import Image
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torchdiffeq import odeint
from general import *
from torch.utils import data as Data
import argparse
import time
from torchvision.transforms.functional import InterpolationMode
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10
from torchvision.models import EfficientNet
from torchvision.ops.misc import SqueezeExcitation, Conv2dNormActivation
from torch.nn.modules.container import Sequential
import copy

from quantization import quantizearray

def init_seeds(seed=0, deterministic=False):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0)
    parser.add_argument('--model', default='efficientnet', choices=['vgg16', 'mobilenetv3', 'efficientnet'])
    parser.add_argument('--myvgg', action='store_true')

    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--k', default=3, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--n_batch_load', default=1, type=int)
    parser.add_argument('--print_freq', default=20, type=int)
    parser.add_argument('--save_freq', default=50, type=int)
    parser.add_argument('--n_epoch', default=100, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--nosave_ckp', action='store_true')
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--lr_scheduler', action='store_true')
    return parser.parse_args()

def train_fp(model,
            n_epoch = 100,
            save_ckp = True,
            save_freq = 50,
            print_freq = 10,
):
    loss_curve = []
    loss_curve_val = []
    acc_curve = []
    acc_curve_val = []
    start = time.time()
    iters = len(train_loader)
    for epoch in range(n_epoch):
        loss_ep = []
        loss_ep_val = []
        acc_ep = []
        acc_ep_val = []

        for batch_i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            pred = model.forward(images)
            loss = criterion(pred, labels) 
            writer.add_scalar("Loss/train", loss, epoch)
            loss.backward()
            optimizer.step()
            if args.lr_scheduler:
                scheduler.step(epoch + batch_i / iters)

            pred = torch.argmax(pred, dim=1)
            acc = torch.tensor(pred==labels, dtype=torch.float).mean()
            writer.add_scalar("Acc/train", acc, epoch)
            loss_ep.append(loss.item())
            acc_ep.append(acc.item())

        for batch_i, (images, labels) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)
            pred = model(images)
            loss_val = criterion(pred, labels)

            pred = torch.argmax(pred, dim=1)
            acc_val = torch.tensor(pred==labels, dtype=torch.float).mean()
            
            loss_ep_val.append(loss_val.item())
            acc_ep_val.append(acc_val.item())
        
        for name, param in model.named_parameters():
            writer.add_scalar(name, param.reshape(-1)[0], epoch)
            # writer.add_histogram(name, param, epoch)
        
        loss = sum(loss_ep)/len(train_loader)
        loss_val = sum(loss_ep_val)/len(val_loader)
        acc = sum(acc_ep)/len(train_loader)
        acc_val = sum(acc_ep_val)/len(val_loader)
        
        writer.add_scalar("Loss/val", loss_val, epoch)
        writer.add_scalar("Acc/val", acc_val, epoch)

        loss_curve.append(loss)
        loss_curve_val.append(loss_val)
        acc_curve.append(acc)
        acc_curve_val.append(acc_val)

        if (epoch+1)%print_freq==0:
            print(f'epoch:{epoch+1}, time:{(time.time() - start):.2f}, loss:{loss:.4f}, acc:{acc:.4f}, loss_val:{loss_val:.4f}, acc_val:{acc_val:.4f}')

        if save_ckp and (epoch+1)%save_freq==0:
            torch.save(model, save_path+f'epoch{epoch+1}_tacc{acc:.3f}_vacc{acc_val:.3f}.ckp')

if __name__=="__main__":
    args = args_parser()
    seed = args.seed
    init_seeds(seed=seed)
    test = args.test
    n_epoch = args.n_epoch
    lr=args.lr
    save_ckp = not args.nosave_ckp
    save_freq = args.save_freq
    mod = args.model
    batch_size = args.batch_size
    n_batch_load = args.n_batch_load
    print_freq = args.print_freq

    if save_ckp:
        save_path=increment_path(f'./images/test_training/')
        writer = SummaryWriter(save_path)
        
    h, w, ch, n_cls, k = 32, 32, 3, 10, args.k
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform_train = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    CIFAR10.train_list = CIFAR10.train_list[: n_batch_load]
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform_test)
    val_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=2)

    if mod == 'vgg16':
        model = torchvision.models.vgg16(weights = 'DEFAULT')
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 10)
        model = model.to(device)
    elif mod == 'mobilenetv3':
        model = torchvision.models.mobilenet_v3_large(weights = 'DEFAULT')
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 10)
        model = model.to(device)
    elif mod == 'efficientnet':
        model = torchvision.models.efficientnet_b0(weights = 'DEFAULT')
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)
        model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    if args.lr_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=1, T_mult=1)
    criterion = nn.CrossEntropyLoss()

    train_fp(model = model,
                n_epoch=n_epoch,
                save_ckp=save_ckp,
                save_freq=save_freq,
                print_freq=print_freq)