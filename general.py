import os
import numpy as np
import scipy.signal as sig
from PIL import Image
import matplotlib.pyplot as plt
import random
import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.optim as optim
from torchdiffeq import odeint
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange


def init_seeds(seed=0, deterministic=False):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe

class TimeEmbedding(nn.Module):
    def __init__(self, dim): # dim = 2 * dim_feature = 2 * ch, half_dim = dim_feature = ch
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.half_dim = dim // 2

        self.wt = nn.Parameter(torch.randn(self.half_dim), requires_grad = True)
        self.time_mlp = nn.Sequential(
        nn.Flatten(),
        nn.Linear(self.dim, 4*self.dim),
        nn.GELU(),
        nn.Linear(4*self.dim, self.dim)
        )

    def SinusoidalPosEmb(self, x):
        theta = 10000
        device = x.device
        emb = math.log(theta) / (self.half_dim - 1)
        emb = torch.exp(torch.arange(self.half_dim, device=device) * -emb)#[None, ...]
        emb = x[:, None] * emb[None, ...]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        return emb

    def MlpEmb(self, x):
        return self.time_mlp(x)

    def FourieredEmb(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.wt, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = x * fouriered # torch.cat((x, fouriered), dim = -1)
        return fouriered
    
class ODEFunc(torch.nn.Module):
    def __init__(self, Ib, A1,
                  B1, maxpool, imgsz):
        super(ODEFunc, self).__init__()
        self.Ib = Ib
        self.A1 = A1
        self.B1 = B1
        self.maxpool = maxpool

        self.transform = transforms.Compose([
            transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(imgsz),
            transforms.Lambda(lambda x: (x - x.min())/(x.max() - x.min())),
            transforms.Normalize((0.48235, 0.45882, 0.40784), (0.00392156862745098, 0.00392156862745098, 0.00392156862745098)),
        ])
    
    def cnn(self, x):
        return 0.5 * (abs(x + 1) - abs(x - 1))

    def forward(self, t, y, u):
        dy1 = -y + self.Ib + self.B1(u) + self.A1(self.cnn(y))
        # dy1 = self.transform(dy1) if dy1.shape[-1] != y.shape[-1] else dy1      
        return dy1

class classifierCELL(nn.Module):
    def __init__(self,
                 ch=3, 
                 k=3, 
                 time_dim=6, 
                 imgsz = 32,
                 mod_emb='sin', 
                 vgg_feature = False,
                 initialCondition=None, 
                 t=None, 
                 test=False, 
                 time_embedding = False,
                 device='cuda',
                 vis_save=False, 
                 path_save=''):
        
        super().__init__()
        self.imgsz = imgsz
        self.cellnet1 = Cellmodel(
                            ch=ch, 
                            k=k, 
                            time_dim=time_dim, 
                            imgsz = 32,
                            mod_emb=mod_emb, 
                            initialCondition=initialCondition, 
                            t=t, 
                            test=test, 
                            time_embedding = time_embedding,
                            device=device,
                            vis_save=vis_save, 
                            path_save=path_save)

        self.vgg = Vgg(feature=vgg_feature)
    
    def forward(self, x, time, train=True):
        if train:
            x = self.cellnet1(x, time)
            x = nn.functional.interpolate(x, size=self.imgsz)
        else:
            x = self.cellnet1(x, time)
            x = nn.functional.interpolate(x, size=self.imgsz)
        return self.vgg(x)
    
class Cellmodel(nn.Module):
    def __init__(self, 
                 ch=3, 
                 k=3, 
                 time_dim=6, 
                 imgsz = 32,
                 mod_emb='sin', 
                 initialCondition=None, 
                 t=None, 
                 test=False, 
                 time_embedding = False,
                 device='cuda',
                 vis_save=False, 
                 path_save=''):
        super().__init__()
        ds = 4
        self.ch = ch
        self.imgsz = imgsz
        self.emb = TimeEmbedding(dim=time_dim)
        self.time_embedding = time_embedding
        self.mod_emb = mod_emb
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.A1 = nn.Conv2d(3, 3, k, padding='same')
        self.B1 = nn.Conv2d(3, 3, k, padding='same')
        self.vis_save = vis_save
        self.path_save = path_save

        if test:
            tempA = torch.tensor(np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])[None,None, ...], dtype=torch.float).to(device)
            tempB = torch.tensor(np.array([[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]])[None,None, ...], dtype=torch.float).to(device)
            tempA = torch.broadcast_to(tempA, (ch, ch, k, k)).clone()
            tempB = torch.broadcast_to(tempB, (ch, ch, k, k)).clone()
            self.A.weight = Parameter(tempA)
            self.B.weight = Parameter(tempB)
        self.initialCondition = initialCondition if initialCondition else torch.zeros(1, self.imgsz, self.imgsz).to(device)
        self.t = None if self.time_embedding else t if t!=None else torch.linspace(0, 10, 10).to(device)
        self.Ib = -1.0
        self._initialize_weights()

        self.ode_func = ODEFunc(self.Ib, self.A1,
                           self.B1, self.maxpool, imgsz=self.imgsz)

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
    
    def __call__(self, img, time = None):
        if self.time_embedding:
            return self.forward_diffusion(img, time)
        else:
            return self.forward(img)

    def forward(self, img):
        u = img
        z0 = u * self.initialCondition
        ode_result = odesolver(z0, self.t, dt=self.t[1]-self.t[0], u=u, func=self.ode_func)
        out = self.ode_func.cnn(ode_result)
        if self.vis_save:
            self.save(out)
        return out
    
    def forward_diffusion(self, img, time):
        u = img
        
        out = self.ode_func(0, u, u)
        if self.mod_emb == 'sin':
            time_emb = self.emb.SinusoidalPosEmb(time)
        elif self.mod_emb == 'fourier':
            time_emb = self.emb.FourieredEmb(time)
        
        time_emb = self.emb.time_mlp(time_emb)
        scale, shift = torch.chunk(time_emb, 2, dim=1)
        out = img*scale[..., None, None] + shift[..., None, None]
        out = self.ode_func.cnn(out)
        if self.vis_save:
            self.save(out)
        return out

    def save(self, out_save):
        out_save = out_save[0] if len(out_save.shape) > 3 else out_save
        out_save = out_save.detach().cpu().numpy().transpose(1, 2, 0)
        # out_save = np.uint8(np.round(out_save.mean(axis=0) * 255))
        out_save = np.uint8(np.round(out_save * 255))
        out_save = Image.fromarray(out_save).convert('RGB')
        out_save.save(self.path_save+'visual_test.jpg')


class Vgg(nn.Module):
    def __init__(self, feature=False):
        super().__init__()
        d=4
        self.feature = feature
        self.features = nn.Sequential(
            nn.Conv2d(3, 64//d, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64//d, 64//d, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ) if self.feature else None

        self.classifier = nn.Sequential(
            nn.Linear(147, 1024, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 10, bias=True),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self._initialize_weights()

    def forward(self, x, train=True):
        x = self.features(x) if self.feature else x
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

def get_save_path(p, epoch, t_max, t_step_max):
    save_path = p + f'tmax{t_max}_tstep{t_step_max}_epoch{epoch}.jpg'
    return save_path

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar10(path, n_batch):
    batches = os.listdir(path)
    batches = [b for b in batches if b.split('_')[0]=='data']
    print(batches)
    datasets = []
    labels = []
    for batch in batches[ :n_batch]:
        dict = unpickle(path+batch)
        dataset = dict[b'data']
        label = np.array(dict[b'labels'])
        datasets.append(dataset)
        labels.append(label)

    datasets, labels = np.concatenate(datasets), np.concatenate(labels)
    print(f'datasets shape: {datasets.shape}, labels shape: {labels.shape}')
    return datasets, labels


class OdeSolver(): # choices ['Euler', 'RK4', 'RK4_alstep']
    def __init__(self, m='euler'):
        self.m = m
    
    def __call__(self, y0, t, dt, u, func):
        if self.m == 'Euler':
            return self.Eulersolver(y0, t, dt, u, func)
        elif self.m == 'RK4':
            return self.RK4solver(y0, t, dt, u, func)
        elif self.m == 'RK4_alstep':
            return self.RK4_altstep_solver(y0, t, dt, u, func)

    def Eulersolver(self, y0, t, dt, u, func):
        y = [y0]
        for ti in range(1, t.shape[0]):
            y.append(y[-1] + func(t, y[-1], u)*dt)
        return y[-1]
    
    def RK4solver(self, y0, t, dt, u, func):
        y = [y0]
        f = []
        for ti in range(1, t.shape[0]):
            f.append(func(t, y[-1], u))
            k1 = f[-1]
            k2 = func(t, y[-1] + dt*k1/2, u)
            k3 = func(t, y[-1] + dt*k2/2, u)
            k4 = func(t, y[-1] + dt*k3, u)
            y.append(y[-1] + dt*(k1 + 2*k2 + 2*k3 + k4)/6)
        return y[-1]
    
    def RK4_altstep_solver(self, y0, t, dt, u, func):
        _one_third = 1/3
        y = [y0]
        f = []
        for ti in range(1, t.shape[0]):
            f.append(func(t, y[-1], u))
            k1 = f[-1]
            k2 = func(t, y[-1] + dt * k1 * _one_third, u)
            k3 = func(t, y[-1] + dt * (k2 - k1 * _one_third), u)
            k4 = func(t, y[-1] + dt * (k1 - k2 + k3), u)
            y.append(y[-1] + (k1 + 3 * (k2 + k3) + k4) * dt * 0.125)
        # y = torch.stack(y, dim=0)
        return y[-1]