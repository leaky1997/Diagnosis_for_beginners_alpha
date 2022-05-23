"""
CNN网络模型
Created on Mon Jun 10 19:49:25 2019
@author: 李奇
"""
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
import utils
import torch.nn.functional as F


def conv1x1(in_channels, out_channels, bias=True, groups=1):
    """
    1x1 卷积
    Args：
    in_channels：输入通道
    out_channels：输出通道

    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias, groups=groups)


class CNNmodel(nn.Module):
    """
    诊断模型类
    """
    def __init__(self, input_size=32, class_num=10):
        """
        初始化类参数
        Args：
        input_size：输入尺寸
        class_num：健康状态分类数

        """
        super(CNNmodel, self).__init__()
        
        self.input_size = input_size
        self.class_num = class_num

        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, 2, 1),            
            nn.LeakyReLU(0.2),
            # 下采样 size=16
            
            nn.Conv2d(64, 128, 3, padding=1),            
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, 2, 1),            
            nn.LeakyReLU(0.2),
            # 下采样 size=8
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.LeakyReLU(0.2),
        )

        self.cla = conv1x1(256, self.class_num)

        utils.initialize_weights(self)
        
    def forward(self, x, output_flag=True):
        """
        Args：
        x：输入模型的诊断数据
        return: 输出预测
        """
        x = self.conv(x)
        x = F.avg_pool2d(x, x.data.size()[-2:])
        c = self.cla(x)
        if output_flag:
            c = c.view(c.size(0), c.size(1))
            return c
        else:
            return x


if __name__=='__main__':
    """
    模型测试
    """
    net = CNNmodel(input_size=32, class_num=10)
    input = torch.randn(1, 1, 32, 32)
    output = net.forward(input)
    
    utils.print_network(net)


            