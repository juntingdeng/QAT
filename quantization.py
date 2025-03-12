
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

