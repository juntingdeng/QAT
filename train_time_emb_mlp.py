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
import subprocess

def print_nvidia_smi():
    try:
        result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, text=True)
        print(result.stdout)
    except Exception as e:
        print(f"Error running nvidia-smi: {e}")

def init_seeds(seed=0, deterministic=False):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe

class Trainer():
    def __init__(self, 
                 time_embedding,
                 time_dim, 
                 mod='sin', 
                 ode_solver='RK4_alstep',
                 time_steps = 10, 
                 train_time_steps = 1000,
                 weight_decay = 0,
                 lr = 1e-3,
                 lr_scheduler = False):
        # dim = 2 * dim_feature = 2 * ch, half_dim = dim_feature = ch
        self.time_embedding = time_embedding
        self.time_dim = time_dim
        self.mod = mod
        self.time_steps = time_steps
        self.train_time_steps = train_time_steps
        self.emb = TimeEmbedding(dim=time_dim)
        self.optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=self.optimizer, T_0=1, T_mult=1) if lr_scheduler else None
        self.criterion = nn.MSELoss()
        self.ode_solver = OdeSolver(ode_solver)

    def train(self, 
                model, 
                train_loader, 
                val_loader,                  
                n_epoch = 100,
                print_freq = 20,
                save_freq = 50,
                save_ckp = False,
                save_path = './'):

        loss_curve = []
        loss_curve_val = []
        acc_curve = []
        acc_curve_val = []
        start = time.time()
        iters = len(train_loader)

        for epoch in range(n_epoch):
            # print_nvidia_smi()
            loss_ep = []
            loss_ep_val = []
            acc_ep = []
            acc_ep_val = []
            model.train()
            for batch_i, (images_, labels_) in enumerate(train_loader):
                if batch_i > 100:
                    break
                # print(f'train batch {batch_i}')
                images, labels = images_.to(device), labels_.to(device)
                loss, acc = 0, 0
                for t in range(args.train_time_steps):
                    self.optimizer.zero_grad()
                    ode_result = self.ode_solver(images, t=torch.linspace(0, 10, 11).to(device), dt=1, u=images, func=model.ode_func)
                    timesteps = torch.randint(0, args.time_steps, (images.shape[0],), device=device).float() if self.time_embedding else None
                    pred = model(images, timesteps)
                    loss_t = self.criterion(pred, ode_result)
                    loss += loss_t
                    loss_t.backward()
                    self.optimizer.step()

                if self.lr_scheduler:
                    self.scheduler.step(epoch + batch_i / iters)

                loss /= args.train_time_steps
                writer.add_scalar("Loss/train", loss, epoch)
                loss_ep.append(loss.item())

            model.eval()
            for batch_i, (images, labels) in enumerate(val_loader):
                if batch_i > 100:
                    break
                # print(f'train batch {batch_i}')
                images, labels = images.to(device), labels.to(device)
                ode_result = self.ode_solver(images, t=torch.linspace(0, 10, 11).to(device), dt=1, u=images, func=model.ode_func)
                loss_val, acc_val = 0, 0
                for t in range(args.train_time_steps):
                    timesteps = torch.randint(0, args.time_steps, (images.shape[0],), device=device).float() if self.time_embedding else None
                    pred = model(images, timesteps)
                    loss_t = self.criterion(pred, ode_result)
                    loss_val += loss_t

                loss_val /= args.train_time_steps
                loss_ep_val.append(loss_val.item())
            
            for name, param in model.named_parameters():
                writer.add_scalar(name, param.reshape(-1)[0], epoch)
                writer.add_histogram(name, param, epoch)
        
            loss = sum(loss_ep)/len(train_loader)
            loss_val = sum(loss_ep_val)/len(val_loader)
            writer.add_scalar("Loss/val", loss_val, epoch)

            loss_curve.append(loss)
            loss_curve_val.append(loss_val)

            if (epoch+1)%print_freq==0:
                print(f'epoch:{epoch+1}, time:{(time.time() - start):.2f}, loss:{loss:.4f}, loss_val:{loss_val:.4f}')

            if save_ckp and (epoch+1)%save_freq==0:
                torch.save(model, save_path+f'epoch{epoch+1}_tloss{loss:.3f}_vloss{loss_val:.3f}.ckp')

        writer.close()
        plt.plot(loss_curve, label='train')
        plt.plot(loss_curve_val, label='val')
        plt.legend()
        plt.savefig(save_path + f'loss.jpg')
        plt.close()
        print(f'Finished. Results saved in {save_path}')
    
def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0)
    parser.add_argument('--model', default='cell')
    parser.add_argument('--mod_emb', default='fourier', choices=['sin', 'fourier'])
    parser.add_argument('--ode_solver', default='RK4_alstep', choices=['Euler', 'RK4', 'RK4_alstep'])
    parser.add_argument('--myvgg', action='store_true')
    parser.add_argument('--vgg_feature', action='store_true')
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--k', default=3, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--n_batch_load', default=1, type=int)
    parser.add_argument('--t_max', default=10, type=int)
    parser.add_argument('--t_step_max', default=2, type=int)
    parser.add_argument('--print_freq', default=1, type=int)
    parser.add_argument('--save_freq', default=50, type=int)
    parser.add_argument('--n_epoch', default=100, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--save_ckp', action='store_true')
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--lr_scheduler', action='store_true')
    parser.add_argument('--time_steps', default=10, type=int)
    parser.add_argument('--train_time_steps', default=100, type=int)
    parser.add_argument('--time_embedding', default=True, action='store_true')
    return parser.parse_args()

if __name__=="__main__":
    args = args_parser()
    init_seeds(seed=args.seed)
    batch_size = args.batch_size

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    h, w, ch, n_cls, k, imgsz = 32, 32, 3, 10, args.k, 224

    save_path=increment_path(f'runs/time_emb_mlp/')
    writer = SummaryWriter(save_path)

    transform_train = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(imgsz),
        transforms.ToTensor(),
        # transforms.Normalize((0.48235, 0.45882, 0.40784), (0.00392156862745098, 0.00392156862745098, 0.00392156862745098)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(imgsz),
        transforms.ToTensor(),
        # transforms.Normalize((0.48235, 0.45882, 0.40784), (0.00392156862745098, 0.00392156862745098, 0.00392156862745098)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform_test)
    val_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    
    _, (test_data, _) = next(enumerate(train_loader))
    print(test_data.min(), test_data.max())

    model = Cellmodel(ch=ch, 
                k=k, 
                time_dim=ch*2, 
                imgsz=imgsz,
                mod_emb=args.mod_emb,  
                time_embedding = args.time_embedding,
                device=device,
                vis_save=True, 
                path_save=save_path).to(device)
    
    trainer = Trainer(time_embedding=args.time_embedding,
                      time_dim=ch*2,
                      mod=args.mod_emb,
                      ode_solver=args.ode_solver,
                      time_steps=args.time_steps,
                      train_time_steps=args.train_time_steps,
                      weight_decay=args.weight_decay,
                      lr=args.lr,
                      lr_scheduler=args.lr_scheduler)
    
    trainer.train(model=model,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      n_epoch=args.n_epoch,
                      print_freq=args.print_freq,
                      save_ckp=args.save_ckp,
                      save_freq=args.save_freq,
                      save_path=save_path)