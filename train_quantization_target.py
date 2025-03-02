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
    parser.add_argument('--model', default='efficientnet', choices=['vgg16', 'mobilenetv3', 'efficientnet', 'resnet18'])
    parser.add_argument('--load_path', default='./images/test_training/runs543/epoch50_tacc0.998_vacc0.914.ckp', type=str)
    parser.add_argument('--myvgg', action='store_true')
    parser.add_argument('--fp_only', action='store_true')
    parser.add_argument('--fp_fixed', action='store_true')

    parser.add_argument('--quant_kl', action='store_true')
    parser.add_argument('--quant_w', default=True, action='store_true')
    parser.add_argument('--quant_x', default=True, action='store_true')
    
    parser.add_argument('--eps', default=0., type=float)
    parser.add_argument('--bitwidth_w', default=8, type=int)
    parser.add_argument('--intbit_w', default=2, type=int)
    parser.add_argument('--bitwidth_x', default=8, type=int)
    parser.add_argument('--intbit_x', default=6, type=int)
    parser.add_argument('--update_freq', default=1, type=int)

    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--k', default=3, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--n_batch_load', default=1, type=int)
    parser.add_argument('--t_max', default=10, type=int)
    parser.add_argument('--t_step_max', default=2, type=int)
    parser.add_argument('--print_freq', default=20, type=int)
    parser.add_argument('--save_freq', default=50, type=int)
    parser.add_argument('--n_epoch', default=100, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--nosave_ckp', action='store_true')
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--lr_scheduler', action='store_true')
    return parser.parse_args()
    
class FakeQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(self, weights, step, min_val, max_val):
        quantized_weights = torch.clamp(torch.round(weights/step)*step, min_val, max_val)
        return quantized_weights 

    @staticmethod
    def backward(self, grad_output):
        return grad_output, None, None, None

class QuantizedWrapper(nn.Module):
    def __init__(self, module, quant_w=True, quant_x=True, bitwidth_w=16, intbit_w=2, bitwidth_x=16, intbit_x=2):
        super().__init__()
        self.module = module  
        self.quant_w = quant_w
        self.quant_x = quant_x
        self.bitwidth_w = bitwidth_w
        self.intbit_w = intbit_w
        self.qstep_w = pow(2, intbit_w) / pow(2, bitwidth_w)
        self.qmax_w = pow(2, intbit_w)/2 - self.qstep_w
        self.qmin_w = -pow(2, intbit_w)/2

        self.bitwidth_x = bitwidth_x
        self.intbit_x = intbit_x
        self.qstep_x = pow(2, intbit_x) / pow(2, bitwidth_x)
        self.qmax_x = pow(2, intbit_x)/2 - self.qstep_x
        self.qmin_x = -pow(2, intbit_x)/2
        self.full_precision_weight = nn.Parameter(self.module.weight) 
        self.fp = False 

    def forward(self, x, fp=False):
        if isinstance(self.module, nn.Linear):
            if not (self.fp or fp):
                quantized_weight = FakeQuantizeSTE.apply(self.full_precision_weight, self.qstep_w, self.qmin_w, self.qmax_w) if self.quant_w else self.full_precision_weight
                y = F.linear(x, quantized_weight, self.module.bias)
                y = FakeQuantizeSTE.apply(y, self.qstep_x, self.qmin_x, self.qmax_x) if self.quant_x else y
                return y
            else:
                return F.linear(x, self.full_precision_weight, self.module.bias)
        elif isinstance(self.module, nn.Conv2d):
            if not (self.fp or fp):
                quantized_weight = FakeQuantizeSTE.apply(self.full_precision_weight, self.qstep_w, self.qmin_w, self.qmax_w) if self.quant_w else self.full_precision_weight
                y = F.conv2d(x, quantized_weight, self.module.bias, stride=self.module.stride, 
                                padding=self.module.padding, groups=self.module.groups, dilation=self.module.dilation)
                y = FakeQuantizeSTE.apply(y, self.qstep_x, self.qmin_x, self.qmax_x) if self.quant_x else y
                return y  
            else:
                return F.conv2d(x, self.full_precision_weight, self.module.bias, stride=self.module.stride, 
                            padding=self.module.padding, groups=self.module.groups, dilation=self.module.dilation)
        else:
            raise TypeError("Unsupported layer type for QAT")

def apply_quantization(model, quant_w=True, quant_x=True, bitwidth_w=16, intbit_w=2, bitwidth_x=16, intbit_x=2):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            setattr(model, name, QuantizedWrapper(module, quant_w, quant_x, bitwidth_w, intbit_w, bitwidth_x, intbit_x))  
        else:
            apply_quantization(module, quant_w, quant_x, bitwidth_w, intbit_w, bitwidth_x, intbit_x) 
    return model

def kl_loss(output_quantized, output_full_precision):
    """
    Compute KL divergence loss between quantized and full-precision outputs.
    """
    log_prob_q = F.log_softmax(output_quantized, dim=-1)  # Log softmax for quantized output
    prob_fp = F.softmax(output_full_precision, dim=-1)  # Softmax for full-precision output
    return F.kl_div(log_prob_q, prob_fp, reduction='batchmean')  # KL loss

class EfficientnetQAT(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.features = model.features
        self.avgpool = model.avgpool
        self.classifier = model.classifier

    def forward(self, x, fp = False):
        for module in self.modules():
            if isinstance(module, QuantizedWrapper):
                module.fp = fp

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class ResnetQAT(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.fc = model.fc
    
    def forward(self, x, fp = False):
        for module in self.modules():
            if isinstance(module, QuantizedWrapper):
                module.fp = fp

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

def train(model,
            n_epoch = 100,
            save_ckp = True,
            save_freq = 50,
            print_freq = 10,
            quant_kl = True,
            quant_w = True,
            quant_x = True,
            update_freq = 1,
            fp_fixed = False
):
    loss_curve = []
    loss_curve_val = []
    loss_kl_curve = []
    loss_kl_curve = []
    acc_curve = []
    acc_curve_val = []
    acc_curve_fp = []
    acc_curve_fp_val = []
    start = time.time()
    iters = len(train_loader)
    grad_states = []
    start_quant = False
    for epoch in range(n_epoch):
        loss_ep = []
        loss_ep_val = []
        loss_kl_ep = []
        loss_kl_ep_val = []
        acc_ep = []
        acc_ep_val = []
        acc_ep_fp = []
        acc_ep_fp_val = []

        for batch_i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            grad_state={}
            images_fp, labels = images.to(device), labels.to(device)
            # with torch.no_grad():
            pred_fp = model_target.forward(images_fp, fp = True) if model_target else model.forward(images_fp, fp = True)
            images = FakeQuantizeSTE.apply(images_fp, qstep_x, qmin_x, qmax_x) if quant_x else images
            pred = model(images)
            loss_ce_fp = criterion(pred_fp, labels) 
            loss_ce = criterion(pred, labels) 
            loss_kl = kl_loss(pred, pred_fp)
            loss = loss_ce + loss_kl if fp_fixed else loss_ce_fp + loss_ce + loss_kl if start_quant else loss_ce_fp 

            loss.backward()
            optimizer.step()
            if args.lr_scheduler:
                scheduler.step(epoch + batch_i / iters)

            pred = torch.argmax(pred, dim=1)
            acc = torch.mean(pred==labels, dtype=torch.float)
            loss_ep.append(loss.item())
            loss_kl_ep.append(loss_kl.item())
            acc_ep.append(acc.item())

            pred_fp = torch.argmax(pred_fp, dim=1)
            acc_fp = torch.mean(pred_fp==labels, dtype=torch.float)
            acc_ep_fp.append(acc_fp.item())

        
        for batch_i, (images, labels) in enumerate(val_loader):
            images_fp, labels = images.to(device), labels.to(device)
            # with torch.no_grad():
            pred_fp = model_target.forward(images_fp, fp = True) if model_target else model.forward(images_fp, fp = True)
            images = FakeQuantizeSTE.apply(images_fp, qstep_x, qmin_x, qmax_x) if quant_x else images
            pred = model(images)

            loss_ce_fp_val = criterion(pred_fp, labels) 
            loss_ce_val = criterion(pred, labels) 
            loss_kl_val = kl_loss(pred, pred_fp)
            loss_val = loss_ce_val + loss_kl_val if fp_fixed else loss_ce_fp_val + loss_ce_val + loss_kl_val if start_quant else loss_ce_fp_val 

            pred = torch.argmax(pred, dim=1)
            acc_val = torch.mean(pred==labels, dtype=torch.float)
            
            loss_ep_val.append(loss_val.item())
            loss_kl_ep_val.append(loss_kl.item())
            acc_ep_val.append(acc_val.item())

            pred_fp = torch.argmax(pred_fp, dim=1)
            acc_fp_val = torch.mean(pred_fp==labels, dtype=torch.float)
            acc_ep_fp_val.append(acc_fp_val.item())
        
        loss = sum(loss_ep)/len(train_loader)
        loss_val = sum(loss_ep_val)/len(val_loader)
        loss_kl = sum(loss_kl_ep)/len(train_loader)
        loss_kl_val = sum(loss_kl_ep_val)/len(val_loader)

        acc = sum(acc_ep)/len(train_loader)
        acc_val = sum(acc_ep_val)/len(val_loader)
        acc_fp = sum(acc_ep_fp)/len(train_loader)
        acc_fp_val = sum(acc_ep_fp_val)/len(val_loader)

        loss_curve.append(loss)
        loss_curve_val.append(loss_val)
        loss_kl_curve.append(loss_kl)
        loss_kl_ep_val.append(loss_kl_val)

        acc_curve.append(acc)
        acc_curve_val.append(acc_val)
        acc_curve_fp.append(acc_fp)
        acc_curve_fp_val.append(acc_fp_val)
        
        start_quant = True if acc_fp_val > 0.9 else start_quant

        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Loss/val", loss_val, epoch)
        writer.add_scalar("Loss.KL/train", loss_kl, epoch)
        writer.add_scalar("Loss.KL/val", loss_kl_val, epoch)

        writer.add_scalar("Acc/train", acc, epoch)
        writer.add_scalar("Acc-FP/train", acc_fp, epoch)
        
        writer.add_scalar("Acc/val", acc_val, epoch)
        writer.add_scalar("Acc-FP/val", acc_fp_val, epoch)

        for name, param in model.named_parameters():
            if name.split('.')[-1] != 'full_precision_weight':
                continue
            grad_state[name] = param.grad.detach().cpu().numpy()

            writer_name = '.'.join(name.split('.')[:-1])
            # writer.add_scalar(f'{writer_name}/Weights', param.reshape(-1)[0], epoch)
            # writer.add_scalar(f'{writer_name}/Grad', param.grad.reshape(-1)[0], epoch)
            writer.add_histogram(f'{writer_name}/Weight.Hist', param, epoch)
            writer.add_histogram(f'{writer_name}/Grad.Hist', param.grad, epoch)
            grad_states.append(grad_state)
        
        if (epoch+1)%print_freq==0:
            print(f'epoch:{epoch+1}, time:{(time.time() - start):.2f}, loss:{loss:.4f}, acc:{acc:.4f}, acc_fp:{acc_fp:.4f}, loss_val:{loss_val:.4f}, acc_val:{acc_val:.4f}, acc_fp_val:{acc_fp_val:.4f}')

        if save_ckp and (epoch+1)%save_freq==0:
            torch.save(model, save_path+f'epoch{epoch+1}_tacc{acc:.3f}_vacc{acc_val:.3f}.ckp')

    writer.close()
    plt.plot(loss_curve, label='train')
    plt.plot(loss_curve_val, label='val')
    plt.legend()
    plt.savefig(save_path + f'loss.jpg')
    plt.close()
    plt.plot(acc_curve, label='train')
    plt.plot(acc_curve_val, label='val')
    plt.legend()
    plt.savefig(save_path + f'acc.jpg')
    print(f'Finished. Results saved in {save_path}')
    np.save(save_path+'grad_states.npy', grad_states)

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
            writer.add_scalar(f'{name}/Weights', param.reshape(-1)[0], epoch)
            writer.add_histogram(f'{name}/Histogram', param, epoch)
        
        loss = sum(loss_ep)/len(train_loader)
        loss_val = sum(loss_ep_val)/len(val_loader)
        acc = sum(acc_ep)/len(train_loader)
        acc_val = sum(acc_ep_val)/len(val_loader)
        
        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Loss/val", loss_val, epoch)

        writer.add_scalar("Acc/train", acc, epoch)
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
    quant_kl = args.quant_kl
    quant_w = args.quant_w
    quant_x = args.quant_x
    bitwidth_w = args.bitwidth_w
    intbit_w = args.intbit_w
    bitwidth_x = args.bitwidth_x
    intbit_x = args.intbit_x
    update_freq = args.update_freq
    fp_only = args.fp_only

    qstep_x = pow(2, intbit_x) / pow(2, bitwidth_x)
    qmax_x = pow(2, intbit_x)/2 - qstep_x
    qmin_x = -pow(2, intbit_x)/2

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
        if quant_w or quant_x:
            model = apply_quantization(model, quant_w=quant_w, quant_x=quant_x, bitwidth_w=bitwidth_w, 
                                       intbit_w=intbit_w, bitwidth_x=bitwidth_x, intbit_x=intbit_x)
            
        model = EfficientnetQAT(model=model)
        if fp_only:
            for module in model.modules():
                if isinstance(module, QuantizedWrapper):
                    module.fp = True
    
    elif mod == 'resnet18':
        model = torchvision.models.resnet18(weights = 'DEFAULT')
        model.fc = nn.Linear(model.fc.in_features, 10)
        model = model.to(device)
        if quant_w or quant_x:
            model = apply_quantization(model, quant_w=quant_w, quant_x=quant_x, bitwidth_w=bitwidth_w, 
                                       intbit_w=intbit_w, bitwidth_x=bitwidth_x, intbit_x=intbit_x)
        model = ResnetQAT(model=model)      

    model_target = torch.load(args.load_path, map_location=device) if args.load_path != '' else None
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    if args.lr_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=1, T_mult=1)
    criterion = nn.CrossEntropyLoss()

    if fp_only:
        train_fp(model = model,
                n_epoch=n_epoch,
                save_ckp=save_ckp,
                save_freq=save_freq,
                print_freq=print_freq)

    else:
        train(model=model,
              n_epoch=n_epoch,
              save_ckp=save_ckp,
              save_freq=save_freq,
              print_freq=print_freq,
              quant_kl=quant_kl,
              quant_w=quant_w,
              quant_x=quant_x,
              update_freq=update_freq)