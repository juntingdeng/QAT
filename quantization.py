
import numpy as np
import torch

def quantizearray(input, step, min, max):
    quantized = input/step
    quantized = quantized + quantized.round().detach() - quantized.detach()
    quantized = quantized * step
    quantized = torch.clip(quantized, min=min, max=max)
    quantized = quantized.requires_grad_()
    quantized.retain_grad()
    return quantized