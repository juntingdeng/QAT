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

class Trainer():
    def __init__(self, 
                 time_embedding,
                 time_dim, 
                 mod='sin', 
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
        self.criterion = nn.CrossEntropyLoss()

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
            loss_ep = []
            loss_ep_val = []
            acc_ep = []
            acc_ep_val = []
            model.train()
            for batch_i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                loss, acc = 0, 0
                for t in range(args.train_time_steps):
                    timesteps = torch.randint(0, args.time_steps, (images.shape[0],), device=device).float() if self.time_embedding else None
                    self.optimizer.zero_grad()
                    pred = model(images, timesteps)
                    loss_t = self.criterion(pred, labels)
                    loss += loss_t
                    loss_t.backward()
                    self.optimizer.step()

                    pred = torch.argmax(pred, dim=1)
                    acc_t = torch.tensor(pred==labels, dtype=torch.float).mean()
                    acc += acc_t

                if self.lr_scheduler:
                    self.scheduler.step(epoch + batch_i / iters)

                loss /= args.train_time_steps
                acc /= args.train_time_steps
                writer.add_scalar("Loss/train", loss, epoch)
                writer.add_scalar("Acc/train", acc, epoch)
                loss_ep.append(loss.item())
                acc_ep.append(acc.item())

            model.eval()
            for batch_i, (images, labels) in enumerate(val_loader):
                images, labels = images.to(device), labels.to(device)
                loss_val, acc_val = 0, 0
                for t in range(args.train_time_steps):
                    timesteps = torch.randint(0, args.time_steps, (images.shape[0],), device=device).float() if self.time_embedding else None
                    pred = model(images, timesteps)
                    loss_t = self.criterion(pred, labels)
                    loss_val += loss_t

                    pred = torch.argmax(pred, dim=1)
                    acc_t = torch.tensor(pred==labels, dtype=torch.float).mean()
                    acc_val += acc_t
                loss_val /= args.train_time_steps
                acc_val /= args.train_time_steps
                
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
    
def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0)
    parser.add_argument('--model', default='cell')
    parser.add_argument('--mod_emb', default='fourier', choices=['sin', 'fourier'])
    parser.add_argument('--myvgg', action='store_true')
    parser.add_argument('--vgg_feature', action='store_true')
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--k', default=3, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
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
    parser.add_argument('--time_embedding', action='store_true')
    return parser.parse_args()

if __name__=="__main__":
    args = args_parser()
    init_seeds(seed=args.seed)
    batch_size = args.batch_size

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    h, w, ch, n_cls, k, imgsz = 32, 32, 3, 10, args.k, 224

    save_path=increment_path(f'runs/diffusion/')
    writer = SummaryWriter(save_path)

    transform_train = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48235, 0.45882, 0.40784), (0.00392156862745098, 0.00392156862745098, 0.00392156862745098)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
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

    if args.model == 'cell':
        t_odesover = torch.linspace(0, args.t_max, args.t_step_max).to(device)
        model = classifierCELL(ch=ch, 
                 k=k, 
                 time_dim=ch*2, 
                 imgsz=imgsz,
                 mod_emb=args.mod_emb, 
                 vgg_feature = args.vgg_feature,
                 time_embedding = args.time_embedding,
                 device=device,
                 vis_save=True,
                 path_save = save_path).to(device)
        
    elif args.model == 'vgg16':
        model = Vgg().to(device) if args.myvgg else torchvision.models.vgg16(num_classes=10).to(device) 
    
    trainer = Trainer(time_embedding=args.time_embedding,
                      time_dim=ch*2,
                      mod=args.mod_emb,
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
                      save_freq=args.save_freq,
                      save_path=save_path)