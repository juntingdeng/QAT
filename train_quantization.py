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

    parser.add_argument('--quant_kl', default=True, action='store_true')
    parser.add_argument('--quant_w',  default=True, action='store_true')
    parser.add_argument('--quant_x',  default=True, action='store_true')
    parser.add_argument('--eps', default=0., type=float)
    parser.add_argument('--bitwidth_w', default=8, type=int)
    parser.add_argument('--intbit_w', default=4, type=int)
    parser.add_argument('--bitwidth_x', default=8, type=int)
    parser.add_argument('--intbit_x', default=4, type=int)
    parser.add_argument('--update_freq', default=1, type=int)

    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--k', default=3, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
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

    def forward(self, x):
        quantized_weight = FakeQuantizeSTE.apply(self.full_precision_weight, self.qstep_w, self.qmin_w, self.qmax_w) if self.quant_w else self.full_precision_weight
        if isinstance(self.module, nn.Linear):
            if not self.fp:
                y = F.linear(x, quantized_weight, self.module.bias)
                y = FakeQuantizeSTE.apply(y, self.qstep_x, self.qmin_x, self.qmax_x) if self.quant_x else y
                return y
            else:
                return F.linear(x, self.full_precision_weight, self.module.bias)
        elif isinstance(self.module, nn.Conv2d):
            if not self.fp:
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
    save_ckp = not args.nosave_ckp
    save_freq = args.save_freq
    mod = args.model
    myvgg = args.myvgg
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
            # EfficientNet.forward_quant = forward_quant.__get__(model)
            model = apply_quantization(model, quant_w=quant_w, quant_x=quant_x, bitwidth_w=bitwidth_w, 
                                       intbit_w=intbit_w, bitwidth_x=bitwidth_x, intbit_x=intbit_x)
        if quant_kl:
            model_fp = copy.deepcopy(model)
            for module in model_fp.modules():
                if isinstance(module, QuantizedWrapper):
                    module.fp = True

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    if args.lr_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=1, T_mult=1)
    criterion = nn.CrossEntropyLoss()

    loss_curve = []
    loss_curve_val = []
    acc_curve = []
    acc_curve_val = []
    acc_curve_fp = []
    acc_curve_fp_val = []
    start = time.time()
    iters = len(train_loader)
    grad_states = []
    for epoch in range(n_epoch):
        loss_ep = []
        loss_ep_val = []
        acc_ep = []
        acc_ep_val = []
        acc_ep_fp = []
        acc_ep_fp_val = []

        for batch_i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            grad_state={}
            images_fp, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                pred_fp = model_fp(images_fp)
            images = FakeQuantizeSTE.apply(images_fp, qstep_x, qmin_x, qmax_x) if quant_x else images
            pred = model(images)
            loss_ce_fp = criterion(pred_fp, labels) 
            loss_ce = criterion(pred, labels) 
            loss_kl = kl_loss(pred, pred_fp)
            loss = loss_ce + loss_kl + loss_ce_fp
            writer.add_scalar("Loss/train", loss, epoch)
            loss.backward()
            optimizer.step()
            if args.lr_scheduler:
                scheduler.step(epoch + batch_i / iters)

            for name, param in model.named_parameters():
                if name.split('.')[-1] != 'full_precision_weight':
                    continue
                grad_state[name] = param.grad.detach().cpu().numpy()
            grad_states.append(grad_state)

            pred = torch.argmax(pred, dim=1)
            acc = torch.tensor(pred==labels, dtype=torch.float).mean()
            writer.add_scalar("Acc/train", acc, epoch)
            loss_ep.append(loss.item())
            acc_ep.append(acc.item())

            pred_fp = torch.argmax(pred_fp, dim=1)
            acc_fp = torch.tensor(pred_fp==labels, dtype=torch.float).mean()
            writer.add_scalar("Acc-FP/train", acc_fp, epoch)
            acc_ep_fp.append(acc_fp.item())

            with torch.no_grad():  
                for param_fp, param_qt in zip(model_fp.parameters(), model.parameters()):
                    param_fp.copy_(param_qt)

        
        for batch_i, (images, labels) in enumerate(val_loader):
            images_fp, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                pred_fp = model_fp(images_fp)
            images = FakeQuantizeSTE.apply(images_fp, qstep_x, qmin_x, qmax_x) if quant_x else images
            pred = model(images)
            loss_val = criterion(pred, labels) + criterion(pred_fp, labels) +  kl_loss(pred, pred_fp)

            pred = torch.argmax(pred, dim=1)
            acc_val = torch.tensor(pred==labels, dtype=torch.float).mean()
            
            loss_ep_val.append(loss_val.item())
            acc_ep_val.append(acc_val.item())

            pred_fp = torch.argmax(pred_fp, dim=1)
            acc_fp_val = torch.tensor(pred_fp==labels, dtype=torch.float).mean()
            acc_ep_fp_val.append(acc_fp_val.item())
        
        for name, param in model.named_parameters():
            writer.add_scalar(name, param.reshape(-1)[0], epoch)
            # writer.add_histogram(name, param, epoch)
        
        loss = sum(loss_ep)/len(train_loader)
        loss_val = sum(loss_ep_val)/len(val_loader)
        acc = sum(acc_ep)/len(train_loader)
        acc_val = sum(acc_ep_val)/len(val_loader)
        acc_fp = sum(acc_ep_fp)/len(train_loader)
        acc_fp_val = sum(acc_ep_fp_val)/len(val_loader)
        
        writer.add_scalar("Loss/val", loss_val, epoch)
        writer.add_scalar("Acc/val", acc_val, epoch)

        loss_curve.append(loss)
        loss_curve_val.append(loss_val)
        acc_curve.append(acc)
        acc_curve_val.append(acc_val)
        acc_curve_fp.append(acc_fp)
        acc_curve_fp_val.append(acc_fp_val)

        if (epoch+1) % update_freq == 0:
            with torch.no_grad():  
                for param_fp, param_qt in zip(model_fp.parameters(), model.parameters()):
                    param_fp.copy_(param_qt)

        if (epoch+1)%print_freq==0:
            print(f'epoch:{epoch+1}, time:{(time.time() - start):.2f}, loss:{loss:.4f}, acc:{acc:.4f}, acc_fp:{acc_fp:.4f}, loss_val:{loss_val:.4f}, acc_val:{acc_val:.4f}, acc_fp_val:{acc_fp_val:.4f}')

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
    np.save(save_path+'grad_states.npy', grad_states)