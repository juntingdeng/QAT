import torch 
import torch.utils
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from quantization import quantizearray
from train_quantization import QuantizedWrapper, RoundWrapper
import argparse
from torch.utils.tensorboard import SummaryWriter
import dill
from PIL import Image

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quantw', action='store_true')
    parser.add_argument('--quantx', action='store_true')
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--data_statistic', default=True, action='store_true')
    parser.add_argument('--weight_statistic', action='store_true')
    parser.add_argument('--save_path', type=str)

    parser.add_argument('--bitwidth_w', default=4, type=int)
    parser.add_argument('--intbit_w', default=2, type=int)
    parser.add_argument('--bitwidth_x', default=4, type=int)
    parser.add_argument('--intbit_x', default=6, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = args_parser()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load('./images/test_training/runs792/epoch150_tacc0.992_vacc0.835.ckp', map_location=device, pickle_module=dill)
    model = model['model'] if isinstance(model, dict) else model
    data_statistic = args.data_statistic
    inference = args.inference
    weight_statistic = args.weight_statistic

    bitwidth_x = args.bitwidth_x
    intbit_x = args.intbit_x
    bitwidth_w = args.bitwidth_w
    intbit_w = args.intbit_w

    if args.quantx:
        step_sizex = 1/pow(2, bitwidth_x - intbit_x)
        xqmax = pow(2, intbit_x)/2 - step_sizex
        xqmin = -pow(2, intbit_x)/2
        for name, param in model.named_parameters():
            if name.split('.')[-1] == 'alpha_x':
                param.data = quantizearray(param.data, step=step_sizex, min=xqmin, max=xqmax)
    
    if args.quantw:
        step_sizew = 1/pow(2, bitwidth_w - intbit_w)
        wqmax = pow(2, intbit_w)/2 - step_sizew
        wqmin = -pow(2, intbit_w)/2
        for name, param in model.named_parameters():
            if name.split('.')[-1] == 'alpha_w':
                param.data = quantizearray(param.data, step=step_sizew, min=wqmin, max=wqmax)

    transform_train = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
        
    transform_test = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


    model.eval()
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

    if data_statistic:
        avg_w = []
        avg_w2 = []
        std = []
        xmin, xmax = 10000, -10000
        print(f'len.trainloader: {len(list(trainloader))}') #len.trainloader: 196
        for batch_i, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            images = images.reshape(images.shape[0], -1)

            avg_w.append(images.mean())
            avg_w2.append(torch.square(images).mean())
            std.append(images.std(dim=1).mean())
            xmin = min(xmin, images.min())
            xmax = max(xmax, images.max())

        for batch_i, (images, labels) in enumerate(testloader):
            # img = images[0].detach().cpu().numpy().transpose(1,2,0)
            # img = 255*((img - img.min())/(img.max() - img.min()))
            # img = Image.fromarray(img.astype(np.uint8))
            # img.save('test.jpg')
            # break
            images, labels = images.to(device), labels.to(device)
            avg_w.append(images.mean())
            avg_w2.append(torch.square(images).mean())
            std.append(images.std(dim=1).mean())
            xmin = min(xmin, images.min())
            xmax = max(xmax, images.max())

        print(f'Range: {xmin} ~ {xmax}, E[w]: {sum(avg_w)/len(avg_w)}, E[w^2]: {sum(avg_w2)/len(avg_w2)}, std: {sum(std)/len(std)}')
    
    if inference:
        acc_ep_val = []
        for batch_i, (images, labels) in enumerate(trainloader):
            pred = model(images)
            pred = torch.argmax(pred, dim=1)
            acc_val = torch.mean(pred==labels, dtype=torch.float)
            acc_ep_val.append(acc_val.item())
        print(f'acc: {sum(acc_ep_val)/len(acc_ep_val)}')
    
    if weight_statistic:
        writer = SummaryWriter(args.save_path)
        for module in model.modules():
            print(type(module).__module__)
            print(type(module).__name__)
            if isinstance(module, QuantizedWrapper):
                fp_weight = module.full_precision_weight
                qt_weight = RoundWrapper.apply(module.full_precision_weight, module.alpha_w, 
                                               module.beta_w, module.g_w, module.qmin_w, module.qmax_w) 

                writer.add_histogram(f'{name}/FP.Hist', param.data)
                writer.add_histogram(f'{name}/QT.Hist', param.grad)

    if args.quantw:
        print(f'quantw, bitwidth_w: {bitwidth_w}, intbit_w: {intbit_w}')
    if args.quantx:
        print(f'quantx, bitwidth_x: {bitwidth_x}, intbit_x: {intbit_x}')