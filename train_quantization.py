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
import dill
import logging
import sys
import yaml

from quantization import *

def init_seeds(seed=0, deterministic=False):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe

def increment_path(p):
    all = os.listdir(p)
    runs=[]
    for f in all:
        if os.path.isdir(p+'/'+f):
            runs.append(int(f[4:]))
    runs = runs if runs else [0]
    run = p + f'runs{max(runs)+1}/'
    assert not os.path.exists(run)
    os.mkdir(run)
    print(f'Results saved in {run}')
    return run
 
def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parse_cmd', action='store_true')
    parser.add_argument('--config', default='./configs/LSQConfig.yaml', type=str)
    
    parser.add_argument('--seed', default=0)
    parser.add_argument('--data', default='cifar10', choices=['cifar10', 'cifar100', 'imagenet'])
    parser.add_argument('--model', default='resnet18', choices=['vgg16', 'mobilenetv3', 'efficientnet', 'resnet18'])
    parser.add_argument('--load_target', default='./images/test_training/runs846/epoch50_tacc1.000_vacc0.934.ckp', type=str,) 
    parser.add_argument('--load_pretrained', default='', type=str,)
    parser.add_argument('--load_target_csd', action='store_true')
    parser.add_argument('--fp_only', action='store_true')
    parser.add_argument('--fp_fixed', action='store_true')
    parser.add_argument('--_1st_1last', action='store_true')
    parser.add_argument('--interp', default='linear', choices=['linear', 'cubic'])
    parser.add_argument('--replace_qtlayer', action='store_true')

    parser.add_argument('--alpha_init', default=2, type=float)
    parser.add_argument('--lsq', action='store_true')
    parser.add_argument('--quant_w', action='store_true')
    parser.add_argument('--quant_x', action='store_true')
    parser.add_argument('--eps', default=0., type=float)

    parser.add_argument('--bitwidth_w', default=4, type=int)
    parser.add_argument('--intbit_w', default=0, type=int)
    parser.add_argument('--bitwidth_x', default=4, type=int)
    parser.add_argument('--intbit_x', default=0, type=int)

    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--noKL', action='store_true')
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--lr_scheduler', action='store_true')
    parser.add_argument('--n_epoch', default=1000, type=int)

    parser.add_argument('--print_freq', default=1, type=int)
    parser.add_argument('--save_freq', default=50, type=int)
    parser.add_argument('--nosave_ckp', action='store_true')

    return parser.parse_args()

def yaml_parser(f, args):
    with open(f, 'r') as stream:
        try:
           args_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return args_yaml

def kl_loss(output_quantized, output_full_precision):
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
            if isinstance(module, QuantizedWrapper) or \
                isinstance(module, QtConv2d) or \
                isinstance(module, QtLinear):
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
        return x
    
def train(model,
          train_loader,
          val_loader,
          model_target,
            criterion,
            optimizer,
            lr_scheduler = None,
            n_epoch = 100,
            save_ckp = True,
            save_freq = 50,
            print_freq = 10,
            quant_w = True,
            quant_x = True,
            lsq = True,
            fp_fixed = False,
            KL = True,
            qstep_x=1,
            qmin_x=0,
            qmax_x=255,
            activation_in=None,
            activation_out=None
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
    best_acc_val = 0
    for epoch in range(n_epoch):
        loss_ep = []
        loss_ep_val = []
        loss_kl_ep = []
        loss_kl_ep_val = []
        acc_ep = []
        acc_ep_val = []
        acc_ep_fp = []
        acc_ep_fp_val = []

        model.train()
        for batch_i, (images, labels) in enumerate(train_loader):
            # print(f'barch: {batch_i}')
            optimizer.zero_grad()
            grad_state={}
            images_fp, labels = images.to(device), labels.to(device)
            # print(f'images range: {images_fp.min()} ~ {images_fp.max()}')
            # with torch.no_grad():
            pred_fp = model_target.forward(images_fp, fp = True) if model_target else model.forward(images_fp, fp = True)

            if lsq:
                images = images_fp
            else:
                images = FixPointQuant.apply(images_fp, qstep_x, qmin_x, qmax_x) if quant_x else images

            # print(images.mean())
            pred = model(images)
            loss_ce_fp = criterion(pred_fp, labels) 
            loss_ce = criterion(pred, labels) 
            loss_kl = kl_loss(pred, pred_fp) if KL else torch.tensor(0).to(device)
            loss = 2*loss_ce + loss_kl if fp_fixed else loss_ce_fp + loss_ce + loss_kl if start_quant else loss_ce_fp 

            loss.backward()
            optimizer.step()
            if lr_scheduler:
                lr_scheduler.step()

            pred = torch.argmax(pred, dim=1)
            acc = torch.mean(pred==labels, dtype=torch.float)
            loss_ep.append(loss.item())
            loss_kl_ep.append(loss_kl.item())
            acc_ep.append(acc.item())

            pred_fp = torch.argmax(pred_fp, dim=1)
            acc_fp = torch.mean(pred_fp==labels, dtype=torch.float)
            acc_ep_fp.append(acc_fp.item())

        model.eval()
        for batch_i, (images, labels) in enumerate(val_loader):
            images_fp, labels = images.to(device), labels.to(device)
            # with torch.no_grad():
            pred_fp = model_target.forward(images_fp, fp = True) if model_target else model.forward(images_fp, fp = True)

            if lsq:
                images = images_fp
            else:
                images = FixPointQuant.apply(images_fp, qstep_x, qmin_x, qmax_x) if quant_x else images

            pred = model(images)

            loss_ce_fp_val = criterion(pred_fp, labels) 
            loss_ce_val = criterion(pred, labels) 
            loss_kl_val = kl_loss(pred, pred_fp) if KL else torch.tensor(0).to(device)
            loss_val = loss_ce_val + loss_kl_val if fp_fixed else loss_ce_fp_val + loss_ce_val + loss_kl_val if start_quant else loss_ce_fp_val 

            pred = torch.argmax(pred, dim=1)
            acc_val = torch.mean(pred==labels, dtype=torch.float)
            
            loss_ep_val.append(loss_val.item())
            loss_kl_ep_val.append(loss_kl_val.item())
            acc_ep_val.append(acc_val.item())

            pred_fp = torch.argmax(pred_fp, dim=1)
            acc_fp_val = torch.mean(pred_fp==labels, dtype=torch.float)
            acc_ep_fp_val.append(acc_fp_val.item())

        for name, input in activation_in.items():
            writer.add_histogram(f'{name}/Val/Input', input, epoch)
        for name, output in activation_out.items():
            writer.add_histogram(f'{name}/Val/Output', output, epoch)

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

        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Loss/val", loss_val, epoch)
        writer.add_scalar("Loss.KL/train", loss_kl, epoch)
        writer.add_scalar("Loss.KL/val", loss_kl_val, epoch)

        writer.add_scalar("Acc/train", acc, epoch)
        writer.add_scalar("Acc-FP/train", acc_fp, epoch)
        
        writer.add_scalar("Acc/val", acc_val, epoch)
        writer.add_scalar("Acc-FP/val", acc_fp_val, epoch)

        alpha_xs = []
        alpha_ws = []
        alpha_grad_xs = []
        alpha_grad_ws = []
        for name, param in model.named_parameters():
            if name.split('.')[-1] == 'full_precision_weight':
                grad_state[name] = param.grad.detach().cpu().numpy()
                writer_name = '.'.join(name.split('.')[:-1])

                writer.add_histogram(f'{writer_name}/Weight.Hist', param.data, epoch)
                writer.add_histogram(f'{writer_name}/Grad.Hist', param.grad, epoch)
                grad_states.append(grad_state)
            
            if name.split('.')[-1] == 'alpha_x':
                alpha_xs.append(param.data)
                alpha_grad_xs.append(param.grad)
            
            if name.split('.')[-1] == 'alpha_w':
                alpha_ws.append(param.data)
                alpha_grad_ws.append(param.grad)
        
        if fp_fixed or start_quant:
            writer.add_histogram(f'alpha_x/Weight.Hist', torch.tensor(alpha_xs).to(device), epoch)
            writer.add_histogram(f'alpha_x/Grad.Hist', torch.tensor(alpha_grad_xs).to(device), epoch)
            writer.add_histogram(f'alpha_w/Weight.Hist', torch.tensor(alpha_ws).to(device), epoch)
            writer.add_histogram(f'alpha_w/Grad.Hist', torch.tensor(alpha_grad_ws).to(device), epoch)
        
        start_quant = True if (not fp_fixed) and (acc_fp_val > 0.9) else start_quant
        if (epoch+1)%print_freq==0:
            logger.info(f'epoch:{epoch+1}, time:{(time.time() - start):.2f}, loss:{loss:.4f}, acc:{acc:.4f}, acc_fp:{acc_fp:.4f}, loss_val:{loss_val:.4f}, acc_val:{acc_val:.4f}, acc_fp_val:{acc_fp_val:.4f}')

        if acc_val >= best_acc_val:
            best_acc_val = acc_val
            best_acc = acc
            best_model = {'model': model,
                          'acc': best_acc_val}
        if save_ckp and (epoch+1)%save_freq==0:
            torch.save(best_model, save_path+f'epoch{epoch+1}_tacc{best_acc:.4f}_vacc{best_acc_val:.4f}.ckp', pickle_module=dill)

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
    logger.info(f'Finished. Results saved in {save_path}')
    # np.save(save_path+'grad_states.npy', grad_states)

def train_fp(model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            lr_scheduler = None,
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
    best_acc_val = 0
    for epoch in range(n_epoch):
        loss_ep = []
        loss_ep_val = []
        acc_ep = []
        acc_ep_val = []

        model.train()
        for batch_i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            pred = model.forward(images, fp = True)
            loss = criterion(pred, labels) 
            loss.backward()
            optimizer.step()
            if lr_scheduler:
                lr_scheduler.step()

            pred = torch.argmax(pred, dim=1)
            acc = torch.mean(pred==labels, dtype=torch.float)
            loss_ep.append(loss.item())
            acc_ep.append(acc.item())

        model.train()
        for batch_i, (images, labels) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)
            pred = model.forward(images, fp = True)
            loss_val = criterion(pred, labels)

            pred = torch.argmax(pred, dim=1)
            acc_val = torch.mean(pred==labels, dtype=torch.float)
            loss_ep_val.append(loss_val.item())
            acc_ep_val.append(acc_val.item())

        loss = sum(loss_ep)/len(train_loader)
        loss_val = sum(loss_ep_val)/len(val_loader)
        acc = sum(acc_ep)/len(train_loader)
        acc_val = sum(acc_ep_val)/len(val_loader)
        
        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Loss/val", loss_val, epoch)
        writer.add_scalar("Acc/train", acc, epoch)
        writer.add_scalar("Acc/val", acc_val, epoch)
        for name, param in model.named_parameters():
            writer.add_histogram(f'{name}/Histogram', param, epoch)

        loss_curve.append(loss)
        loss_curve_val.append(loss_val)
        acc_curve.append(acc)
        acc_curve_val.append(acc_val)

        if (epoch+1)%print_freq==0:
            logger.info(f'epoch:{epoch+1}, time:{(time.time() - start):.2f}, loss:{loss:.4f}, acc:{acc:.4f}, loss_val:{loss_val:.4f}, acc_val:{acc_val:.4f}')

        if acc_val >= best_acc_val:
            best_acc_val = acc_val
            best_acc = acc
            best_model = {'model': model,
                          'acc': best_acc_val}
            
        if save_ckp and (epoch+1)%save_freq==0:
            torch.save(best_model, save_path+f'epoch{epoch+1}_tacc{best_acc:.4f}_vacc{best_acc_val:.4f}.ckp', pickle_module=dill)

def csd_intersect(csd_model, csd_target, load_target):
    keys_model = csd_model.keys()
    keys_target = csd_target.keys()
    cnt = 0
    notloaded = []
    for k, v in csd_model.items():
        if k in keys_target and csd_target[k].shape == v.shape:
            csd_model[k] = csd_target[k]
            cnt += 1
        else:
            notloaded.append(k)
    logger.info(f'Load {cnt}/{len(keys_model)} parameters from {load_target}.')
    return csd_model

def csd_intersect_(csd_model, csd_target, load_target):
    keys_model = csd_model.keys()
    keys_target = csd_target.keys()
    cnt = 0
    notloaded = []
    for k, v in csd_target.items():
        if k in keys_model and csd_model[k].shape == v.shape:
            csd_model[k] = csd_target[k]
            cnt += 1

        elif k.split('.')[-1] == 'full_precision_weight':
            knew = '.'.join(k.split('.')[:-1]+['weight']) 
            if knew in keys_model and csd_model[knew].shape == v.shape:
                csd_model[knew] = csd_target[k]
                cnt += 1

        else:
            notloaded.append(k)
    logger.info(f'Load {cnt}/{len(keys_model)} parameters from {load_target}.')
    return csd_model


activation_in = {}
activation_out = {}
if __name__ == '__main__':
    args = args_parser()
    args_yaml = yaml_parser(args.config, args=args)

    if not args.parse_cmd and args_yaml:
        for k, v in vars(args).items():
            if k in args_yaml.keys():
                vars(args)[k] = args_yaml[k][args_yaml['model']] if k == 'load_target' and args_yaml['load_target']!='' \
                    else args_yaml[k]

    seed = args.seed
    init_seeds(seed=seed)

    mod = args.model
    lr=args.lr
    KL = not args.noKL
    n_epoch = args.n_epoch
    save_ckp = not args.nosave_ckp
    save_freq = args.save_freq
    batch_size = args.batch_size
    print_freq = args.print_freq

    fp_only = args.fp_only
    fp_fixed = args.fp_fixed
    alpha_init = args.alpha_init
    lsq = args.lsq
    quant_w = args.quant_w
    quant_x = args.quant_x
    bitwidth_w = args.bitwidth_w
    intbit_w = args.intbit_w
    bitwidth_x = args.bitwidth_x
    intbit_x = args.intbit_x

    qstep_x = pow(2, intbit_x) / pow(2, bitwidth_x) if not lsq else 1
    qmax_x = pow(2, intbit_x)/2 - qstep_x
    qmin_x = -pow(2, intbit_x)/2

    if save_ckp:
        save_path=increment_path(f'./images/test_training/')
        writer = SummaryWriter(save_path)
    
    ## Add log
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    stdout_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(f'{save_path}logs.log')
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    ## Write args
    write_args = ''
    for k, v in vars(args).items():
        write_args += f'{k}:{v}, '
    logger.info(write_args)
        
    h, w, ch, n_cls = 32, 32, 3, 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.interp == 'cubic':
        interp = InterpolationMode.BICUBIC
    elif args.interp == 'linear':
        interp = InterpolationMode.BILINEAR

    transform_train = transforms.Compose([
        transforms.Resize(256, interpolation=interp), #Efficientnet: BICUBIC, Resnet: BILINEAR
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256, interpolation=interp),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    if args.data == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        n_cls = 10
    elif args.data == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        n_cls = 100
    elif args.data == 'imagenet':
        trainset = torchvision.datasets.ImageNet(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.ImageNet(root='./data', train=False, download=True, transform=transform_test)
        n_cls = 1000
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    if args.load_pretrained != '':
        model = torch.load(args.load_pretrained, map_location=device, pickle_module=dill)['model']
    else:  
        if mod == 'vgg16':
            model = torchvision.models.vgg16(weights = 'DEFAULT')
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, n_cls)
            model = model.to(device)
        elif mod == 'mobilenetv3':
            model = torchvision.models.mobilenet_v3_large(weights = 'DEFAULT')
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, n_cls)
            model = model.to(device)
        elif mod == 'efficientnet':
            model = torchvision.models.efficientnet_b0(weights = 'DEFAULT') 
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_cls)
            model = model.to(device)

            if args.replace_qtlayer:
                model = replace_QtLayer(model=model, lsq=lsq, bitwidth_w=bitwidth_w, bitwidth_x=bitwidth_x)
            else:
                model = apply_quantization(model, quant_w=quant_w, quant_x=quant_x, lsq=lsq, bitwidth_w=bitwidth_w, 
                                        intbit_w=intbit_w, bitwidth_x=bitwidth_x, intbit_x=intbit_x, alpha_init=alpha_init,
                                        activation_in=activation_in, activation_out=activation_out)
            
            model = EfficientnetQAT(model=model)
        
        elif mod[:6] == 'resnet':
            if mod[6:]=='18':
                model = torchvision.models.resnet18(weights = 'DEFAULT') 
            elif mod[6:]=='101':
                model = torchvision.models.resnet101(weights = 'DEFAULT')
                 
            model.fc = nn.Linear(model.fc.in_features, n_cls)
            model = model.to(device)
            if args.replace_qtlayer:
                model = replace_QtLayer(model=model, lsq=lsq, bitwidth_w=bitwidth_w, bitwidth_x=bitwidth_x)
            else:
                model = apply_quantization(model, quant_w=quant_w, quant_x=quant_x, lsq=lsq, bitwidth_w=bitwidth_w, 
                                        intbit_w=intbit_w, bitwidth_x=bitwidth_x, intbit_x=intbit_x, alpha_init=alpha_init,
                                        activation_in=activation_in, activation_out=activation_out)
            model = ResnetQAT(model=model)      

        if args._1st_1last:
            for layer in model.modules():
                if isinstance(layer, QuantizedWrapper):
                    layer.bitwidth_w = 8
                    layer.bitwidth_x = 8
                    break
            last_layer = None
            for layer in model.modules():
                if isinstance(layer, QuantizedWrapper):
                    last_layer = layer
            last_layer.bitwidth_w = 8
            last_layer.bitwidth_x = 8
        
        if fp_only:
            for module in model.modules():
                if isinstance(module, QuantizedWrapper):
                    module.fp = True

    model_target = torch.load(args.load_target, map_location=device, pickle_module=dill)['model'] if args.load_target != '' else None
    if args.load_target_csd and not args.load_pretrained:
        csd_target = model_target.state_dict()
        csd_model = model.state_dict()
        csd_model = csd_intersect_(csd_model=csd_model, csd_target=csd_target, load_target=args.load_target)
        model.load_state_dict(csd_model)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay) if args.optim == 'adam' \
                    else optim.SGD(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=200) if args.lr_scheduler else None
    criterion = nn.CrossEntropyLoss()

    if fp_only:
        train_fp(model = model,
                 train_loader=train_loader,
                 val_loader=val_loader,
                 criterion=criterion,
                 optimizer=optimizer,
                 lr_scheduler=lr_scheduler,
                n_epoch=n_epoch,
                save_ckp=save_ckp,
                save_freq=save_freq,
                print_freq=print_freq)

    else:
        train(model=model,
              train_loader=train_loader,
              val_loader=val_loader,
              model_target=model_target,
              criterion=criterion,
              optimizer=optimizer,
              lr_scheduler=lr_scheduler,
              n_epoch=n_epoch,
              save_ckp=save_ckp,
              save_freq=save_freq,
              print_freq=print_freq,
              quant_w=quant_w,
              quant_x=quant_x,
              fp_fixed=fp_fixed,
              KL=KL,
              qmax_x=qmax_x,
              qmin_x=qmin_x,
              qstep_x=qstep_x,
              activation_in=activation_in,
              activation_out=activation_out)
        