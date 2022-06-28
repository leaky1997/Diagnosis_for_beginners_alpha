# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 14:40:46 2020

@author: 李奇
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# =============================================================================
# Self-attention
# =============================================================================


class Self_Attn(nn.Module):
    """
    Self attention Layer
    """
    def __init__(self, in_dim, activation):
        """
        Args:
            in_dim:输入特征的维度
            activation:选用的激活函数
        """
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.adv_gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        
        self.adv_value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()  # size = [1, 256, 8, 8]
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)  # B X CX(N),size=[1,64,32]
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)  # B X C x (*W*H),size=[1, 32, 64]
        energy = torch.bmm(proj_query, proj_key)  # transpose check, size=[1, 64, 64]
        attention = self.softmax(energy)  # B X (N) X (N), size=[1, 64, 64]

        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)  # B X C X N, size=[1, 256, 64]

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B X C X N, size=[1, 256, 64]
        out = out.view(m_batchsize, C, width, height)  # B X C X N, size=[1, 256, 8, 8]
        
        out = self.gamma*out + x
        return out, attention

    def adv_Attn(self, x, attention):
        m_batchsize, C, width, height = x.size()
        
        proj_value = self.adv_value_conv(x).view(m_batchsize, -1, width*height)  # B X C X N
        
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
               
        out = self.adv_gamma*(1-out) + x
        return x
        

    