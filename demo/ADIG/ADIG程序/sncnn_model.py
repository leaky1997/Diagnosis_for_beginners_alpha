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
from src.snlayers.snconv2d import SNConv2d
from src.snlayers.snlinear import SNLinear


def conv1x1(in_channels, out_channels, bias=True, groups=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias, groups=groups)

class CNNmodel(nn.Module):
    def __init__(self, input_size=32, class_num=10):
        super(CNNmodel, self).__init__()
        
        self.input_size = input_size
        self.class_num = class_num

        self.feature_extractor = nn.Sequential(
            
            SNConv2d(1, 64, 3,padding=1),
#            nn.Conv2d(1, 64, 3,padding=1),
            nn.BatchNorm2d(64),            
            nn.LeakyReLU(0.2),
            
            
            SNConv2d(64, 128, 3, 2, 1),            
#            nn.Conv2d(64, 128, 3, 2, 1),            
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            
            SNConv2d(128, 256, 3, 2,padding=1), 
#            nn.Conv2d(128, 256, 3, 2,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            SNConv2d(256, 256, 3, 2,1),            
#            nn.Conv2d(256, 256, 3, 2,1),
            nn.BatchNorm2d(256),            
            nn.LeakyReLU(0.2),            
        )

        self.clf = nn.Sequential(
                
                nn.AvgPool2d(4),                                #平均池化
                
                conv1x1(256,self.class_num),  

                )

        utils.initialize_weights(self)
        
    def forward(self,input):
        features = self.feature_extractor(input)
        y = self.clf(features)
        

        y=y.view(y.size(0),y.size(1))  #[1,10,1,1]变二维
        return y,features

#%%
class Dmodel(nn.Module):
    def __init__(self,domain_class=4):
        super(Dmodel, self).__init__()

        self.Dclf = nn.Sequential(
                
                SNConv2d(256, 128, 3, 1,1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),

                
                SNConv2d(128, 64, 3, 2,1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),
                
#                nn.Linear(4*4*64,64),
                nn.AvgPool2d(2),                #平均池化
                conv1x1(64,domain_class)

                )
        self.DclfwithoutSN = nn.Sequential(
                
                nn.Conv2d(256, 128, 3, 1,1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),

                
                nn.Conv2d(128, 64, 3, 2,1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),
                
#                nn.Linear(4*4*64,64),
                nn.AvgPool2d(2),                #平均池化
                conv1x1(64,domain_class)

                )

        self.clf2 = nn.Sequential(
                SNLinear(4*4*256,256),
                nn.LeakyReLU(0.2),
                SNLinear(256,domain_class),
#                nn.LeakyReLU(0.2),
#                SNLinear(64,domain_class)

                
                )

        utils.initialize_weights(self)
    def forward(self,input):        
        x=self.DclfwithoutSN(input)
#        x = input.view(input.size(0),-1)
#        x = self.clf2(x)
        return x.view(x.size(0),x.size(1))
#%%
#class S_CNN(nn.Module):
#    def __init__(self,input_size=32, class_num=10,feature_dim=256):
#        super(S_CNN, self).__init__()
#        
#        self.input_size = input_size
#        self.class_num = class_num
#        self.conv = nn.Sequential(
#            nn.Conv2d(1, 64, 3,padding=1),            
#            nn.LeakyReLU(0.2),
#            nn.Conv2d(64, 64, 3, 2, 1),            
#            nn.LeakyReLU(0.2),
#            
#            nn.Conv2d(64, 128, 3, padding=1),            
#            nn.LeakyReLU(0.2),
#            nn.Conv2d(128, 128, 3, 2, 1),            
#            nn.LeakyReLU(0.2),
#            
#            nn.Conv2d(128, feature_dim, 3, padding=1),
#            nn.LeakyReLU(0.2),
#        )
#
#
#        utils.initialize_weights(self)
#        
#    def forward(self,input,output_flag=True):
#        x = self.conv(input)
#        return x
#    
#    
##class T_CNN(nn.Module):
#            
#class Clf(nn.Module):
#    def __init__(self,input_size=32, class_num=10,feature_dim=256):
#        super(Clf, self).__init__()
#        
#        self.input_size = input_size
#        self.class_num = class_num
#        self.conv = nn.Sequential(
#            nn.Conv2d(feature_dim, feature_dim, 3, 2,1),     
#            nn.LeakyReLU(0.2),
#        )
#
#        self.clf = conv1x1(feature_dim,self.class_num)
#
#        utils.initialize_weights(self)
#        
#    def forward(self,input,output_flag=True):
#        x = self.conv(input)
#        x=F.avg_pool2d(x, x.data.size()[-2:])
#        c = self.clf(x)
#        if output_flag:
#            c=c.view(c.size(0),c.size(1))
#            return c
#        else:
#            return x
    
if __name__=='__main__':
#%%
    net = CNNmodel(input_size=32,class_num=10)

    input = torch.randn(1,1,32,32)
    output = net.forward(input)
    
    utils.print_network(net)
    
    Dnet = Dmodel()
    utils.print_network(Dnet)
#%%
#    feature=net.feature_extractor(input)
#    d=Dmodel()
#    utils.print_network(d)
#    Dcla=d(feature)
#%%把特征提取和分类器分开来，每个都是50k的参数量
#    s_cnn=S_CNN(input_size=32, class_num=10,feature_dim=256)
#    clf=Clf(input_size=32, class_num=10,feature_dim=256)
#    input = torch.randn(1,1,32,32)
#    feature_output = s_cnn.forward(input)
#    output=clf.forward(feature_output)
#    utils.print_network(s_cnn)
#    utils.print_network(clf)

            