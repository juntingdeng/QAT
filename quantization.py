
import numpy as np
import torch
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def quantizearray(input, step, min, max):
    quantized = input/step
    quantized = quantized + quantized.round().detach() - quantized.detach()
    quantized = quantized * step
    quantized = torch.clip(quantized, min=min, max=max)
    quantized = quantized.requires_grad_()
    quantized.retain_grad()
    return quantized

class STEWrapper(torch.autograd.Function):
    @staticmethod
    def forward(self,x):
        y = x.round()
        return y 

    @staticmethod
    def backward(self, grad_output):
        return grad_output
    
class FixPointQuant(torch.autograd.Function):
    @staticmethod
    def forward(self, weights, step, min_val, max_val):
        quantized_weights = torch.clamp(torch.round(weights/step)*step, min_val, max_val)
        return quantized_weights 

    @staticmethod
    def backward(self, grad_output):
        return grad_output, None, None, None

class LSQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights, alpha, beta, g, qmin, qmax):
        ctx.save_for_backward(weights, alpha, beta)
        ctx.g, ctx.qmin, ctx.qmax = g, qmin, qmax
        _weights_q = ((weights-beta)/alpha).round().clamp(qmin, qmax)
        weights_q = _weights_q * alpha + beta
        return weights_q

    @staticmethod
    def backward(ctx, grad_weight):
        weights, alpha, beta = ctx.saved_tensors
        g, qmin, qmax = ctx.g, ctx.qmin, ctx.qmax
        _weights_q = (weights-beta)/alpha

        ind_small = (_weights_q < qmin).float()
        ind_big = (_weights_q > qmax).float()
        ind_mid = 1. - ind_small - ind_big

        grad_alpha = ((ind_small * qmin + ind_big * qmax + ind_mid * (- _weights_q + _weights_q.round())) * grad_weight * g).sum().view(1)
        grad_beta = ((ind_mid * 0 + ind_big + ind_small)* grad_weight * g).sum().view(1)
        grad_weight = ind_mid * grad_weight
        
        return grad_weight, grad_alpha, grad_beta, None, None, None

class QuantizedWrapper(nn.Module):
    def __init__(self, module, quant_w=True, quant_x=True, lsq=None,
                 bitwidth_w=16, intbit_w=2, bitwidth_x=16, intbit_x=2, _1st_1last=False, alpha_init=None):
        super().__init__()
        self.module = module  
        self.quant_w = quant_w
        self.quant_x = quant_x
        self.lsq = lsq
        self.bitwidth_w = bitwidth_w if not _1st_1last else 8
        self.intbit_w = intbit_w
        self.qstep_w = pow(2, intbit_w) / pow(2, bitwidth_w) if not lsq else 1
        self.qmax_w = pow(2, intbit_w)/2 - self.qstep_w if not lsq else pow(2, bitwidth_w-1) - 1
        self.qmin_w = -pow(2, intbit_w)/2 if not lsq else -pow(2, bitwidth_w-1)

        self.bitwidth_x = bitwidth_x if not _1st_1last else 8
        self.intbit_x = intbit_x
        self.qstep_x = pow(2, intbit_x) / pow(2, bitwidth_x) if not lsq else 1
        self.qmax_x = pow(2, intbit_x)/2 - self.qstep_x if not lsq else pow(2, bitwidth_x-1) - 1
        self.qmin_x = -pow(2, intbit_x)/2 if not lsq else -pow(2, bitwidth_x-1)
        self.full_precision_weight = nn.Parameter(self.module.weight) 
        self.quantized_weight = None

        self.fp = False 
        self.alpha_x_initialized = False
        self.g_w = 1.0 / math.sqrt(self.full_precision_weight.numel() * self.qmax_w)

        if self.quant_w and self.quant_x:
            if alpha_init == 1: #LSQ
                self.alpha_w = nn.Parameter(2*self.full_precision_weight.abs().mean()/math.sqrt(self.qmax_w))
                self.alpha_x = nn.Parameter(2*self.full_precision_weight.abs().mean()/math.sqrt(self.qmax_x))
                self.beta_w = nn.Parameter(torch.tensor(0.))
                self.beta_x = nn.Parameter(torch.tensor(0.))
    
            elif alpha_init == 2: #LSQ+
                # BILINEAR -- Dataset[train, val]: Range(-2.1179039478302,2.640000104904175), E[w]=0.08617200702428818, E[w^2]=1.1822057962417603, std=0.8140683770179749
                # BILINEAR(test2) -- Dataset[train, val]: Range(-2.1179039478302,2.640000104904175), E[w]: 0.08747030794620514, E[w^2]: 1.1379928588867188, std: 0.7935436964035034
                # BICUBIC -- Dataset[train, val]: Range(-2.1179039478302,2.640000104904175), E[w]=0.0860467255115509, E[w^2]=1.1820635795593262, std= 0.8139052987098694
                mu_w, sigma_w = self.full_precision_weight.mean(), self.full_precision_weight.std()
                mu_x, sigma_x = 0.08617200702428818, 0.8140683770179749 ## BILINEAR
                min_x, max_x = -2.1179039478302, 2.640000104904175
                self.alpha_w = nn.Parameter(max(abs(mu_w-3*sigma_w), abs(mu_w+3*sigma_w))/self.qmax_w)
                self.beta_w = nn.Parameter(mu_w)
                self.alpha_x = nn.Parameter(torch.tensor((max_x - min_x)/(self.qmax_x - self.qmin_x)))
                self.beta_x = nn.Parameter(min_x - self.qmin_x*self.alpha_x)
            
            # Dataset[train, val]: E[w]=0.08601280301809311, E[w^2]=1.1822082996368408. (First batch of images.mean(): -0.1799)
            elif alpha_init == 3:
                e1_w, e2_w, c1_w, c2_w = self.full_precision_weight.mean(), (self.full_precision_weight**2).mean(), 3.2, -2.1
                e1_x, e2_x, c1_x, c2_x = 0.08601280301809311, 1.1822082996368408, 3.2, -2.1
                self.alpha_w = nn.Parameter(torch.tensor(c1_w*math.sqrt(e2_w)-c2_w*e1_w)/math.sqrt(self.qmax_w)) 
                self.alpha_x = nn.Parameter(torch.tensor(c1_x*math.sqrt(e2_x)-c2_x*e1_x)/math.sqrt(self.qmax_x)) 
                self.beta_w = nn.Parameter(torch.tensor(0.))
                self.beta_x = nn.Parameter(torch.tensor(0.))

    def forward(self, x, fp=False):
        # self.module.bias = None
        if isinstance(self.module, nn.Linear):
            if not (self.fp or fp):
                if self.lsq==1:
                    self.quantized_weight = LSQuant.apply(self.full_precision_weight, self.alpha_w, self.beta_w, self.g_w, self.qmin_w, self.qmax_w) 
                    x = LSQuant.apply(x, self.alpha_x, self.beta_x, 1.0/math.sqrt(x.numel() * self.qmax_x), self.qmin_x, self.qmax_x) 

                elif self.lsq==2:
                    self.quantized_weight = self.alpha_w*STEWrapper.apply((self.full_precision_weight-self.beta_w)/self.alpha_w).clamp(self.qmin_w, self.qmax_w) + self.beta_w
                    x = self.alpha_x*STEWrapper.apply((x-self.beta_x)/self.alpha_x).clamp(self.qmin_x, self.qmax_x) + self.beta_x

                else:
                    self.quantized_weight = FixPointQuant.apply(self.full_precision_weight, self.qstep_w, self.qmin_w, self.qmax_w) if self.quant_w else self.full_precision_weight
                    x = FixPointQuant.apply(x, self.qstep_x, self.qmin_x, self.qmax_x) if self.quant_x else x

                y = F.linear(x, self.quantized_weight, self.module.bias)
                return y
            else:
                return F.linear(x, self.full_precision_weight, self.module.bias)
            
        elif isinstance(self.module, nn.Conv2d):
            if not (self.fp or fp):
                if self.lsq==1:
                    self.quantized_weight = LSQuant.apply(self.full_precision_weight, self.alpha_w, self.beta_w, self.g_w, self.qmin_w, self.qmax_w) 
                    x = LSQuant.apply(x, self.alpha_x, self.beta_x, 1.0/math.sqrt(x.numel() * self.qmax_x), self.qmin_x, self.qmax_x) 

                elif self.lsq==2:
                    self.quantized_weight = self.alpha_w*STEWrapper.apply((self.full_precision_weight-self.beta_w)/self.alpha_w).clamp(self.qmin_w, self.qmax_w) + self.beta_w
                    x = self.alpha_x*STEWrapper.apply((x-self.beta_x)/self.alpha_x).clamp(self.qmin_x, self.qmax_x) + self.beta_x 
                
                else:
                    self.quantized_weight = FixPointQuant.apply(self.full_precision_weight, self.qstep_w, self.qmin_w, self.qmax_w) if self.quant_w else self.full_precision_weight
                    x = FixPointQuant.apply(x, self.qstep_x, self.qmin_x, self.qmax_x) if self.quant_x else x
                
                y = F.conv2d(x, self.quantized_weight, self.module.bias, stride=self.module.stride, 
                                padding=self.module.padding, groups=self.module.groups, dilation=self.module.dilation) 
                return y
            else:
                return F.conv2d(x, self.full_precision_weight, self.module.bias, stride=self.module.stride, 
                            padding=self.module.padding, groups=self.module.groups, dilation=self.module.dilation)
        else:
            raise TypeError("Unsupported layer type for QAT")
        
    def forward_(self, x):
        self.module.bias = None
        if isinstance(self.module, nn.Linear):
            _weights_q = (self.full_precision_weight/self.alpha_w).round().clamp(self.qmin_w, self.qmax_w)     
            y = F.linear(x, _weights_q, self.module.bias)
            y = y * self.alpha_w
            y = LSQuant.apply(y, self.alpha_x, self.beta_x, 1.0/math.sqrt(y.numel() * self.qmax_x), self.qmin_x, self.qmax_x) 
            return y

        elif isinstance(self.module, nn.Conv2d):
            _weights_q = (self.full_precision_weight/self.alpha_w).round().clamp(self.qmin_w, self.qmax_w) 
            y = F.conv2d(x, _weights_q, self.module.bias, stride=self.module.stride, 
                            padding=self.module.padding, groups=self.module.groups, dilation=self.module.dilation)
            y = y * self.alpha_w
            y = LSQuant.apply(y, self.alpha_x, self.beta_x, 1.0/math.sqrt(y.numel() * self.qmax_x), self.qmin_x, self.qmax_x) 
            return y  

def apply_quantization(model, pre_name = '', quant_w=True, quant_x=True, lsq=None,
                       bitwidth_w=16, intbit_w=2, bitwidth_x=16, intbit_x=2, alpha_init=None, activation_in=None, activation_out=None):
    for name, module in model.named_children():
        cur_name = pre_name+'.'+name if pre_name != '' else name
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            wrapper = QuantizedWrapper(module, quant_w, quant_x, lsq, bitwidth_w, intbit_w, bitwidth_x, intbit_x, alpha_init=alpha_init)
            wrapper.register_forward_hook(get_activation(name=cur_name, activation_in=activation_in, activation_out=activation_out))
            setattr(model, name, wrapper)  
            
        else:
            apply_quantization(module, cur_name, quant_w, quant_x, lsq, bitwidth_w, intbit_w, bitwidth_x, intbit_x, 
                               alpha_init=alpha_init, activation_in=activation_in, activation_out=activation_out) 
    return model

def get_activation(name, activation_in, activation_out):
    def hook(model, input, output):
        input_tuple = tuple(input) 
        activation_in[name] = input_tuple[0] if isinstance(input_tuple, tuple) and len(input_tuple) == 1 \
                else (x.detach() for x in input_tuple)
        activation_out[name] = output.detach()
    return hook

class QtConv2d(nn.Conv2d):
    def __init__(self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode='zeros', 
            device=None,
            dtype=None,
            wbit=4,
            xbit=4,
            lsq=1,
            fp=False):
        super().__init__(in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device=device,
            dtype=dtype,)
        self.wbit = wbit
        self.xbit = xbit
        self.qmax_w = pow(2, self.wbit-1) - 1
        self.qmin_w = -pow(2, self.wbit-1)
        self.qmax_x = pow(2, self.xbit-1) - 1
        self.qmin_x = -pow(2, self.xbit-1)
        self.lsq=lsq
        self.fp=fp

        mu_w, sigma_w = self.weight.mean(), self.weight.std()
        min_x, max_x = -2.1179039478302, 2.640000104904175
        self.alpha_w = nn.Parameter(max(abs(mu_w-3*sigma_w), abs(mu_w+3*sigma_w))/self.qmax_w)
        self.beta_w = nn.Parameter(mu_w)
        self.alpha_x = nn.Parameter(torch.tensor((max_x - min_x)/(self.qmax_x - self.qmin_x)))
        self.beta_x = nn.Parameter(min_x - self.qmin_x*self.alpha_x)
        self.g_w = 1.0 / math.sqrt(self.weight.numel() * self.qmax_w)

    def forward(self, input):
        if self.fp:
            return self._conv_forward(input, self.weight, self.bias)
        
        if self.lsq == 1:
            qweight = LSQuant.apply(self.weight, self.alpha_w, self.beta_w, self.g_w, self.qmin_w, self.qmax_w) 
            qinput = LSQuant.apply(input, self.alpha_x, self.beta_x, 1.0/math.sqrt(input.numel() * self.qmax_x), self.qmin_x, self.qmax_x) 
        elif self.lsq == 2:
            qweight = self.alpha_w*STEWrapper.apply((self.weight-self.beta_w)/self.alpha_w).clamp(self.qmin_w, self.qmax_w) + self.beta_w
            qinput = self.alpha_x*STEWrapper.apply((input-self.beta_x)/self.alpha_x).clamp(self.qmin_x, self.qmax_x) + self.beta_x

        return self._conv_forward(qinput, qweight, self.bias)

class QtLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        device=None,
        dtype=None,
        wbit=4,
        xbit=4,
        lsq=1,
        fp=False):
        super().__init__(in_features,
                out_features,
                bias,
                device,
                dtype,)
        
        self.wbit = wbit
        self.xbit = xbit
        self.qmax_w = pow(2, self.wbit-1) - 1
        self.qmin_w = -pow(2, self.wbit-1)
        self.qmax_x = pow(2, self.xbit-1) - 1
        self.qmin_x = -pow(2, self.xbit-1)
        self.lsq=lsq
        self.fp=fp

        mu_w, sigma_w = self.weight.mean(), self.weight.std()
        min_x, max_x = -2.1179039478302, 2.640000104904175
        self.alpha_w = nn.Parameter(max(abs(mu_w-3*sigma_w), abs(mu_w+3*sigma_w))/self.qmax_w)
        self.beta_w = nn.Parameter(mu_w)
        self.alpha_x = nn.Parameter(torch.tensor((max_x - min_x)/(self.qmax_x - self.qmin_x)))
        self.beta_x = nn.Parameter(min_x - self.qmin_x*self.alpha_x)
        self.g_w = 1.0 / math.sqrt(self.weight.numel() * self.qmax_w)

    def forward(self, input):
        if self.fp:
            return F.linear(input, self.weight, self.bias)
        
        if self.lsq == 1:
            qweight = LSQuant.apply(self.weight, self.alpha_w, self.beta_w, self.g_w, self.qmin_w, self.qmax_w) 
            qinput = LSQuant.apply(input, self.alpha_x, self.beta_x, 1.0/math.sqrt(input.numel() * self.qmax_x), self.qmin_x, self.qmax_x) 
        elif self.lsq == 2:
            qweight = self.alpha_w*STEWrapper.apply((self.weight-self.beta_w)/self.alpha_w).clamp(self.qmin_w, self.qmax_w) + self.beta_w
            qinput = self.alpha_x*STEWrapper.apply((input-self.beta_x)/self.alpha_x).clamp(self.qmin_x, self.qmax_x) + self.beta_x
        
        return F.linear(qinput, qweight, self.bias)

def replace_QtLayer(model, lsq=None, bitwidth_w=16, bitwidth_x=16):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            bias = module.bias if module.bias == None else True
            qtLayer = QtLinear(module.in_features, module.out_features, bias,
                               device=module.weight.device, dtype=module.weight.dtype,
                            wbit=bitwidth_w, xbit=bitwidth_x, lsq=lsq)
            setattr(model, name, qtLayer)  
        elif isinstance(module, nn.Conv2d):
            bias = module.bias if module.bias == None else True
            qtLayer = QtConv2d(module.in_channels, module.out_channels, module.kernel_size, module.stride,
                               module.padding, module.dilation, module.groups, bias, module.padding_mode, 
                               device=module.weight.device, dtype=module.weight.dtype,
                            wbit=bitwidth_w, xbit=bitwidth_x, lsq=lsq)
            setattr(model, name, qtLayer)  
        else:
            replace_QtLayer(module, lsq, bitwidth_w, bitwidth_x)
    return model

