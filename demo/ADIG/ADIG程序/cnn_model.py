"""
Created on Mon Jun 10 19:49:25 2019
@author: 李奇
"""
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
import utils
import torch.nn.functional as F
from Grad_reverse_layer import grad_reverse as GRL
from src.snlayers.snconv2d import SNConv2d
from src.snlayers.snlinear import SNLinear
from Self_attention import Self_Attn


def conv1x1(in_channels, out_channels, bias=True, groups=1):
    """
    1*1逐点卷积，调整通道数量
    Args:
        in_channels: 输入通道
        out_channels: 输出通道
        bias: 偏置
        groups: 卷积核分组数

    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias, groups=groups)


def snconv1x1(in_channels, out_channels, bias=True, groups=1):
    """
    谱归一化（Spectral Normalization）
    对判别器的每层参数矩阵施加Lipschitz约束，提高GAN训练的稳定性
    Args:
        in_channels: 输入通道
        out_channels: 输出通道
        bias: 偏置
        groups: 卷积核分组数

    """
    return SNConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias, groups=groups)


class CNNmodel(nn.Module):
    def __init__(self, input_size=32, class_num=10, domains=3, norm=None, lambd=0.2):
        """
        诊断模型
        Args:
            input_size: 输入数据尺寸
            class_num: 健康状况数量
            domains: 域数量
            norm:
            lambd:
        """
        super(CNNmodel, self).__init__()
        
        self.input_size = input_size
        self.class_num = class_num
        self.lambd = lambd
        # 原始特征提取器，无BN，无SN，无IN层
        self.feature_extractor_org = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, 2, 1),            
            nn.LeakyReLU(0.2),
            # [h, w] = 16
            
            nn.Conv2d(64, 128, 3, padding=1),            
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, 2, 1),            
            nn.LeakyReLU(0.2),
            # [h, w] = 8
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.LeakyReLU(0.2),
        )
        # 特征提取器+IN（实例正则化）
        self.feature_extractor_with_ln = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.InstanceNorm2d(64),      
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.InstanceNorm2d(64),             
            nn.LeakyReLU(0.2),
            # [h, w] = 16
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.InstanceNorm2d(128),             
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, 2, 1),
            nn.InstanceNorm2d(128),             
            nn.LeakyReLU(0.2),
            # [h, w] = 8
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.InstanceNorm2d(256),             
            nn.LeakyReLU(0.2),
        )
        # 特征提取器+BN
        self.feature_extractor_with_bn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            # [h, w] = 16

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            # [h, w] = 8

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        # 特征提取器+SN（谱归一化）
        self.feature_extractor_with_sn = nn.Sequential(
            SNConv2d(1, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            SNConv2d(64, 64, 3, 2, 1),            
            nn.LeakyReLU(0.2),
            # [h, w] = 16
            
            SNConv2d(64, 128, 3, padding=1),            
            nn.LeakyReLU(0.2),
            SNConv2d(128, 128, 3, 2, 1),            
            nn.LeakyReLU(0.2),
            # [h, w] = 8
            
            SNConv2d(128, 256, 3, padding=1),
            nn.LeakyReLU(0.2),
            )
        # 特征提取器+SN+BN
        # self.feature_extractor_with_sn_and_BN(
        #     SNConv2d(1, 64, 3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.LeakyReLU(0.2),
        #
        #     SNConv2d(64, 128, 3, 2, 1),
        #     nn.BatchNorm2d(128),
        #     nn.LeakyReLU(0.2),
        #
        #     SNConv2d(128, 256, 3, 2, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.LeakyReLU(0.2),
        #
        #     SNConv2d(256, 256, 3, 2, 1),
        #     nn.BatchNorm2d(256),
        #     nn.LeakyReLU(0.2),
        # )

        # SBDS的骨架
        # self.clf = nn.Sequential(
        #     nn.Conv2d(256, 256, 3, 2, 1),
        #     nn.LeakyReLU(0.2),
        #     nn.AvgPool2d(4),
        #     # 平均池化
        #     conv1x1(256, self.class_num)
        #        )
        
        self.attentionC = Self_Attn(256, 'relu')
        self.attentionD = Self_Attn(256, 'relu')

        # SUST的骨架，分类器
        self.clf = nn.Sequential(
            nn.Conv2d(256, 256, 3, 2, 1),  # 下采样
            nn.LeakyReLU(0.2),
            nn.Dropout(),
            # [h, w] = 4*4
            nn.AvgPool2d(4),
            # [h, w] = 1*1
            conv1x1(256, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(),                                                               
            conv1x1(64, self.class_num)
                )
                
#        self.Dclf = nn.Sequential(
#                
#                nn.Conv2d(256, 128, 3, 2,1),     
#                nn.LeakyReLU(0.2),
#                nn.Conv2d(128, 64, 3, 2,1),     
#                nn.LeakyReLU(0.2),
#                nn.AvgPool2d(2),                #平均池化
#                conv1x1(64,domains)               
#                )
        self.Dclf = nn.Sequential(
            SNConv2d(256, 256, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Dropout(),                                           
            nn.AvgPool2d(4),
            snconv1x1(256, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(),
            snconv1x1(64, domains)
                )
#        self.Dclf = nn.Sequential(
#            nn.Conv2d(256, 256, 3, 2,1),
#            nn.LeakyReLU(0.2),
#            nn.Dropout(),                                           
#            nn.AvgPool2d(4),
#            conv1x1(256,64),
#            nn.LeakyReLU(0.2),
#            nn.Dropout(),
#            conv1x1(64,domains)        
#                )
        
        utils.initialize_weights(self)
        if norm == 'ln':
            self.feature_extractor = self.feature_extractor_with_ln
        elif norm == 'sn':
            self.feature_extractor = self.feature_extractor_with_sn
#        elif norm=='bn':
#            self.feature_extractor = self.feature_extractor_with_bn
        else:
            self.feature_extractor = self.feature_extractor_org

    def mid_rep(self, x, num_fe=0, num_c=0, num_d=0):
        """

        Args:
            x:
            num_fe:
            num_c:
            num_d:

        Returns:

        """
        if all([num_fe == 0, num_c == 0, num_d == 0]):
            # common knowledge
            return None
        
        if num_fe:
            for i in range(num_fe):
                net = self.feature_extractor[i]
                x = net(x)
            return x
     
        x = self.feature_extractor(x)   
        # 1-cclassifier knowledge
        if num_c:
            for i in range(num_c):
                net = self.clf[i]
                # trace nn.Sequential
                x = net(x)
            return x
        
        if num_d:            
            for i in range(num_d):
                net = self.Dclf[i]
                # trace nn.Sequential
                x = net(x)
            return x

    def forward(self, x):
        """
        前向传播
        Args:
            input: 输入数据

        Returns:故障分类预测标签与领域分类标签

        """
        x = self.feature_extractor(x)  # size = [256, 256, 8, 8]
        xt1 = x
        x_grl = GRL(x, lambd=self.lambd)  # size = [256, 256, 8, 8]

        x, _ = self.attentionC(x)  # size = [256, 256, 8, 8]
        xt2 = x
        # print(xt1 == xt2)
        # x_grl,_ = self.attentionD(x_grl)

        c = self.clf(x)
        d = self.Dclf(x_grl)
        
        c = c.view(c.size(0), c.size(1))
        d = d.view(d.size(0), d.size(1))

        return c, d

    
if __name__=='__main__':
    net = CNNmodel(input_size=32, class_num=10, domains=3, norm='ln')
    input = torch.randn(1, 1, 32, 32)
    output = net.forward(input)
    print('output_c.shape = ', output[0].shape)
    print('output_d.shape = ', output[1].shape)
    
    # utils.print_network(net)
