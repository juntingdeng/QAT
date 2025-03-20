import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from quantization import *
from train_quantization import init_seeds

init_seeds(0)

x = torch.rand((1,32,32,32))
model = nn.Conv2d(32, 32, 3)
qt_model = QuantizedWrapper(model, quant_w=True, quant_x=True, lsq=1, 
                            bitwidth_w=4, intbit_w=0, bitwidth_x=4, intbit_x=0, alpha_init=2)
qt_model_new = QtConv2d(32, 32, 3, wbit=4, xbit=4)

# x = torch.rand((1,256))
# model = nn.Linear(256, 1)
# qt_model = QuantizedWrapper(model, quant_w=True, quant_x=True, lsq=1, 
#                             bitwidth_w=4, intbit_w=0, bitwidth_x=4, intbit_x=0, alpha_init=2)
# qt_model_new = QtLinear(256, 1, wbit=4, xbit=4)

optim1 = optim.SGD(model.parameters(), lr = 1)
optim2 = optim.SGD(qt_model.parameters(), lr = 1)
optim3 = optim.SGD(qt_model_new.parameters(), lr = 1)

n = 1000
t_inf, t_back, t_step = 0, 0, 0
for i in range(n):
    model.zero_grad()
    t1 = time.time()
    y = model(x).sum()
    t2 = time.time()
    y.backward()
    t3 = time.time()
    optim1.step()
    t4 = time.time()
    t_inf += t2-t1
    t_back += t3-t2
    t_step += t4-t3
print(f'FP time inference: {t_inf/n}, backward: {t_back/n}, step: {t_step/n}')

t_inf, t_back, t_step = 0, 0, 0
for i in range(n):
    qt_model.zero_grad()
    t1 = time.time()
    y = qt_model(x).sum()
    t2 = time.time()
    y.backward()
    t3 = time.time()
    optim2.step()
    t4 = time.time()
    t_inf += t2-t1
    t_back += t3-t2
    t_step += t4-t3
print(f'QT time inference: {t_inf/n}, backward: {t_back/n}, step: {t_step/n}')

t_inf, t_back, t_step = 0, 0, 0
for i in range(n):
    qt_model.zero_grad()
    t1 = time.time()
    y = qt_model_new(x).sum()
    t2 = time.time()
    y.backward()
    t3 = time.time()
    optim3.step()
    t4 = time.time()
    t_inf += t2-t1
    t_back += t3-t2
    t_step += t4-t3
print(f'QT-New time inference: {t_inf/n}, backward: {t_back/n}, step: {t_step/n}')