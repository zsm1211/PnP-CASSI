import numpy as np
import torch.nn as nn
import torch
import torch.utils.data as data
from torch.optim import Adam
from torch.optim import lr_scheduler
from collections import OrderedDict
import os
import math
from torch.utils.data import DataLoader
import random
from torch.optim import Adam

def pixel_unshuffle(input, upscale_factor):
    batch_size, channels, in_height, in_width = input.size()

    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor

    input_view = input.contiguous().view(
        batch_size, channels, out_height, upscale_factor,
        out_width, upscale_factor)

    channels *= upscale_factor ** 2
    unshuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return unshuffle_out.view(batch_size, channels, out_height, out_width)

class PixelUnShuffle(nn.Module):


    def __init__(self, upscale_factor):
        super(PixelUnShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        return pixel_unshuffle(input, self.upscale_factor)

def sequential(*args):

    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

def conv_rule(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True,if_relu=True):
    L=[]
    L.append(nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding,bias=bias))
    if if_relu:
        L.append(nn.ReLU(inplace=True))
    return sequential(*L)

class HSI_SDeCNN(nn.Module):
    def __init__(self, in_nc=7, out_nc=1, nc=128, nb=15):

        super(HSI_SDeCNN,self).__init__()
        sf=2

        self.m_down=PixelUnShuffle(upscale_factor=sf)

        m_head=conv_rule(in_nc*sf*sf+1,nc)
        m_body=[conv_rule(nc,nc) for _ in range(nb-2)]
        m_tail=conv_rule(nc,out_nc*sf*sf,if_relu=False)

        self.model = sequential(m_head, *m_body, m_tail)
        self.m_up = nn.PixelShuffle(upscale_factor=sf)
    def forward(self, x,sigma):
        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h / 2) * 2 - h)
        paddingRight = int(np.ceil(w / 2) * 2 - w)
        x = torch.nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        x = self.m_down(x)
        m = sigma.repeat(1, 1, x.size()[-2], x.size()[-1])
        x = torch.cat((x, m), 1)
        x = self.model(x)
        x = self.m_up(x)

        x = x[..., :h, :w]
        return x