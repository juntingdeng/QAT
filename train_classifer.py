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

def init_seeds(seed=0, deterministic=False):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe

class classifierCELL(nn.Module):
    def __init__(self, h, w, ch, k, n_cls, initialCondition=None, t=None, test=False, device='cuda'):
        super().__init__()
        self.cellnet1 = cellmodel(h, w, ch, k, 
                                 test=test, 
                                 initialCondition=initialCondition,
                                 t=t,
                                 device=device)
        # self.cellnet2 = cellmodel(h, w, ch, k, 
        #                          test=test, 
        #                          initialCondition=initialCondition,
        #                          t=t,
        #                          device=device)
        # self.cellnet3 = cellmodel(h, w, ch, k, 
        #                          test=test, 
        #                          initialCondition=initialCondition,
        #                          t=t,
        #                          device=device)
        self.fc1 = nn.Linear(147, n_cls)
        # self.fc2 = nn.Linear(4096, 4096)
        # self.fc3 = nn.Linear(4096, n_cls)
        self.relu = nn.ReLU()
        self.vgg16 = vgg16()
    
    def __call__(self, x, train=True):
        return self.forward(x, train=train)
    
    def forward(self, x, train=True):
        if train:
            x = self.cellnet1(x)
            x = nn.functional.interpolate(x, size=224)
            # x = self.cellnet2(x)
            # x = self.cellnet3(x)
        else:
            x = self.cellnet1(x)
            x = nn.functional.interpolate(x, size=7)
        # x = x.reshape(x.shape[0], -1)
        return self.vgg16(x) #self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))

class classifierCNN(nn.Module):
    def __init__(self, h, w, ch, k, n_cls):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class vgg16(nn.Module):
    def __init__(self):
        super().__init__()
        d=1
        self.features = nn.Sequential(
            nn.Conv2d(3, 64//d, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64//d, 64//d, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64//d, 128//d, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128//d, 128//d, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128//d, 256//d, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256//d, 256//d, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256//d, 256//d, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256//d, 512//d, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512//d, 512//d, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512//d, 512//d, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512//d, 512//d, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512//d, 512//d, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512//d, 512//d, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(25088, 4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(4096, 4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(4096, 10, bias=True),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self._initialize_weights()

    def forward(self, x, train=True):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0)
    parser.add_argument('--model', default='cell')
    parser.add_argument('--myvgg', action='store_true')
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--k', default=3, type=int)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--n_batch_load', default=1, type=int)
    parser.add_argument('--t_max', default=10, type=int)
    parser.add_argument('--t_step_max', default=2, type=int)
    parser.add_argument('--print_freq', default=20, type=int)
    parser.add_argument('--save_freq', default=50, type=int)
    parser.add_argument('--n_epoch', default=100, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--save_ckp', action='store_true')
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--lr_scheduler', action='store_true')
    return parser.parse_args()

if __name__=="__main__":
    args = args_parser()
    seed = args.seed
    init_seeds(seed=seed)
    test = args.test
    n_epoch = args.n_epoch
    lr=args.lr
    t_max = args.t_max
    t_step_max = args.t_step_max
    t_step = t_max/t_step_max
    save_ckp = args.save_ckp
    save_freq = args.save_freq
    mod = args.model
    myvgg = args.myvgg
    batch_size = args.batch_size
    n_batch_load = args.n_batch_load
    print_freq = args.print_freq

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    h, w, ch, n_cls, k = 32, 32, 3, 10, args.k
    # path = '../dataset/cifar-10-batches-py/'
    # datasets, labels = load_cifar10(path=path, n_batch=n_batch_load)
    # datasets = datasets.reshape(datasets.shape[0], ch, h, w)/255
    # print(f'dataset shape: {datasets.shape}, labels shape: {labels.shape}')
    # datasets_all = Data.TensorDataset(torch.tensor(datasets, dtype=torch.float).to(device), torch.tensor(labels).to(device))
    # generator = torch.Generator().manual_seed(seed)
    # train_dataset, val_dataset = Data.random_split(datasets_all, [0.6, 0.4], generator=generator)
    # train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    # val_loader = Data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, generator=generator)

    save_path=increment_path(f'images/test_training/')
    writer = SummaryWriter(save_path)

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform_train = transforms.Compose([
        transforms.Resize(160, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(160),
        transforms.ToTensor(),
        transforms.Normalize((0.48235, 0.45882, 0.40784), (0.00392156862745098, 0.00392156862745098, 0.00392156862745098)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(160, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(160),
        transforms.ToTensor(),
        transforms.Normalize((0.48235, 0.45882, 0.40784), (0.00392156862745098, 0.00392156862745098, 0.00392156862745098)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform_test)
    val_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    if mod == 'cell':
        model = classifierCELL(64, 64, ch, k, n_cls, t=torch.linspace(0, t_max, t_step_max).to(device), device=device).to(device)
    elif mod == 'conv':
        model = classifierCNN(h, w, ch, k, n_cls).to(device)
    elif mod == 'vgg16':
        model = vgg16().to(device) if myvgg else torchvision.models.vgg16(num_classes=10).to(device) 

    
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    if args.lr_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=1, T_mult=1)
    criterion = nn.CrossEntropyLoss()

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
            pred = model(images)
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
            writer.add_histogram(name, param, epoch)
        
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
            torch.save(model, save_path+f'tmax{t_max}_tstep{t_step_max}_epoch{epoch+1}_tacc{acc:.3f}_vacc{acc_val:.3f}.ckp')

    writer.close()
    plt.plot(loss_curve, label='train')
    plt.plot(loss_curve_val, label='val')
    plt.legend()
    plt.savefig(save_path + f'tmax{t_max}_tstep{t_step_max}_loss.jpg')
    plt.close()
    plt.plot(acc_curve, label='train')
    plt.plot(acc_curve_val, label='val')
    plt.legend()
    plt.savefig(save_path + f'tmax{t_max}_tstep{t_step_max}_acc.jpg')
    print(f'Finished. Results saved in {save_path}')