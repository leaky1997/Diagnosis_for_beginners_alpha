#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 09:13:34 2019

@author: liqi
"""
# from ShuffleNetV2 import ShuffleNetV2 as sfmodel
# from SBDS_cnn_model import CNNmodel

from readdata import dataset
from cnn_model import CNNmodel
from os.path import splitext
import pandas as pd
import scipy
import os
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import utils
from AutomaticWeightedLoss import AutomaticWeightedLoss
from data_loader import Loader_single, Loader_unif_sampling, Loader_unif_sampling_DA


class Diagnosis():
    """
    diagnosis model
    """
    def __init__(self, n_class=10, lr=0.001, batch_size=64,
                 gpu=True, domains=3, save_dir='save_dir/',
                 model_name='default', norm=None):
        """
        Args:
            n_class:健康状态数量
            lr:学习率
            batch_size:训练批量大小
            gpu:是否选用GPU训练
            domains:域数量
            save_dir:训练历史数据保存路径
            model_name:选用模型名称
            norm:正则化
        """
        print('diagnosis begin')

        self.net = CNNmodel(input_size=32, class_num=n_class, domains=domains, norm=norm, lambd=-1)
        self.gpu = gpu
        self.save_dir = save_dir
        self.model_name = model_name
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if self.gpu:
            self.net.cuda()
        
        self.lr = lr
        self.balanceloss = AutomaticWeightedLoss(2)  # 自适应权重损失
        self.optimizer = optim.Adam([{'params': self.net.parameters()},
                                     {'params': self.balanceloss.parameters(), 'lr': self.lr*10}],
                                    lr=self.lr, weight_decay=0.0001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=self.lr)
        self.loss_function = nn.CrossEntropyLoss()  # 交叉熵损失
        self.MSE = nn.MSELoss()  # 均方差损失
        self.batch_size = batch_size

        # 保存历史数据
        self.train_hist = {}
        self.train_hist['loss'] = []
        self.train_hist['acc'] = []
        self.train_hist['testloss'] = []
        self.train_hist['testacc'] = []
        
#    def sourceds(self,num):
#        
#        labels=np.zeros((num,1))
#        labels[:,0]=1
#        ds = torch.tensor(labels).cuda()
#        return ds
        
    def caculate_acc(self, prediction, y_):
        """
        计算准确率
        """
        correct = torch.zeros(1).squeeze().cuda()
        total = torch.zeros(1).squeeze().cuda()
        correct += (prediction == y_).sum().float()
        total += len(y_)
        acc = (correct/total).cpu().detach().data.numpy()
        return acc

    def fit(self, s_dataset_train, s_dataset_test, t_dataset, epoches=1, test_flag='t'):
        """
        Args:
            s_dataset_train:源域训练集
            s_dataset_test：源域测试集
            t_dataset：目标域数据集
            epoches：训练迭代次数
            test_flag：
        """
        print('training start!!')
            
#       torch_dataset = TensorDataset(x_train,y_train)
#         加载数据集
        loader =DataLoader(
            dataset=s_dataset_train,     
            batch_size=self.batch_size,      
            shuffle=True,               
            num_workers=2,
            )
        
        whole_time = time.time()
        for epoch in range(epoches):
            loss_epoch = []
            acc_epoch = []
            epoch_start_time = time.time()
            for iter, (x1, x2, y1, y2, d1, d2) in enumerate(loader):
                if self.gpu:
                    x1, y1 = x1.cuda(), y1.cuda()
                    x2, y2 = x2.cuda(), y2.cuda()
#                    x3, y3 = x3.cuda(), y3.cuda()
                xs = torch.cat((x1, x2), dim=0)
                ys = torch.cat((y1, y2), dim=0)
                ds = torch.cat((d1, d2), dim=0)
                ds = ds.cuda()  # view(-1,1)
                
                self.optimizer.zero_grad()

                y_pre,d_pre = self.net(xs)
            
                Dloss = self.loss_function(d_pre,ds)
                
                loss = self.loss_function(y_pre, torch.max(ys, 1)[1])#+0.2*Dloss 
                loss = self.balanceloss(loss+Dloss)
                prediction = torch.argmax(y_pre, 1)
                ys=torch.argmax(ys,1)



                loss_epoch.append(loss.item())
                loss.backward()
                self.optimizer.step()
                
                acc=self.caculate_acc(prediction,ys)
                
                acc_epoch.append(acc)
                
            epoch_end_time = time.time()
            
            loss=np.mean(loss_epoch)
            acc=np.mean(acc_epoch)
            epoch_time=epoch_end_time-epoch_start_time
            self.train_hist['loss'].append(loss)         
            self.train_hist['acc'].append(acc)
            if test_flag=='s':
                self.s_evaluation(s_dataset_test,train_step=True)
            elif test_flag=='t':
                self.t_evaluation(t_dataset,train_step=True)
            
            if epoch%1==0:
                print("Epoch: [%2d] Epoch_time:  [%8f] \n loss: %.8f, acc:%.8f" %
                              ((epoch + 1),epoch_time,loss,acc))

            



        total_time=epoch_end_time-whole_time
        print('Total time: %2d'%total_time)
        print('best acc: %4f'%(np.max(self.train_hist['acc'])))
        print('best testacc: %4f'%(np.max(self.train_hist['testacc'])))
        

                
                
        self.save_his()
#        self.save_prediction(x_test)
        print('=========训练完成=========')
        
        return self.train_hist
    def s_evaluation(self,s_dataset_test,train_step=False):
        '''
        x_test,y_test,train_step=False
        '''
        print('evaluation')
        
        self.net.eval()
        loaders =DataLoader(
            dataset=s_dataset_test,     
            batch_size=self.batch_size,      
            shuffle=True,               
            num_workers=32,              
            )
        

        loss_epoch=[]
        acc_epoch=[]
        with torch.no_grad():
            for iter, (x1, x2, x3,y1,y2,y3,d1,d2,d3) in enumerate(loaders):
                if self.gpu:
                    x1, y1 = x1.cuda(), y1.cuda()
                    x2, y2 = x2.cuda(), y2.cuda()
#                    x3, y3 = x3.cuda(), y3.cuda()
                xs = torch.cat((x1,x2),dim=0)
                ys = torch.cat((y1,y2),dim=0)
                
                self.optimizer.zero_grad()

                y_pre,_ = self.net(xs)
            
                loss = self.loss_function(y_pre, torch.max(ys, 1)[1])          
                prediction = torch.argmax(y_pre, 1)
                ys=torch.argmax(ys,1)



                loss_epoch.append(loss.item())

                
                acc=self.caculate_acc(prediction,ys)
                
                acc_epoch.append(acc)
                    
            loss=np.mean(loss_epoch)
            acc=np.mean(acc_epoch)
        
        

        
        print("***\ntest result \n loss: %.8f, acc:%.4f\n***" %
              (loss,
               acc))
        
        if train_step:
            self.train_hist['testloss'].append(loss)
            self.train_hist['testacc'].append(acc)
            
            if acc>=np.max(self.train_hist['testacc']) :      
                self.save()
                
                print(' a better model is saved')

              


        self.net.train()
        return loss,acc
    
    def t_evaluation(self,t_dataset,train_step=False):
        '''
        x_test,y_test,train_step=False
        '''
        print('evaluation')
        
        self.net.eval()

        
        loadert =DataLoader(
            dataset=t_dataset,     
            batch_size=self.batch_size,      
            shuffle=False,               
            num_workers=32,              
            )
        loss_epoch=[]
        acc_epoch=[]
        with torch.no_grad():
            for iter, (xt,yt) in enumerate(loadert):
                if self.gpu:
                    xt, yt = xt.cuda(), yt.cuda()

                
                self.optimizer.zero_grad()

                y_pre,_ = self.net(xt)
            
                loss = self.loss_function(y_pre, torch.max(yt, 1)[1])          
                prediction = torch.argmax(y_pre, 1)
                yt=torch.argmax(yt,1)



                loss_epoch.append(loss.item())

                
                acc=self.caculate_acc(prediction,yt)
                
                acc_epoch.append(acc)
                    
            loss=np.mean(loss_epoch)
            acc=np.mean(acc_epoch)
        
        

        
        print("***\ntest result \n loss: %.8f, acc:%.4f\n***" %
              (loss,
               acc))
        
        if train_step:
            self.train_hist['testloss'].append(loss)
            self.train_hist['testacc'].append(acc)
            
            if acc>=np.max(self.train_hist['testacc']) :      
                self.save()
                
                print(' a better model is saved')

              


        self.net.train()
        return loss,acc
    
    def save(self):
        '''
        save model and its parameters
        '''
        torch.save(self.net.state_dict(), self.save_dir+self.model_name+'.pkl')
        
    def save_prediction(self,data,data_name='test'):
        '''
        save prediction
        '''
        data=data.cuda()

        final_pre,_=self.net(data)
        
        final_pre=torch.argmax(final_pre,1).cpu().detach().data.numpy()
        dic={}
        dic['prediction']=[]
        dic['prediction']=final_pre
        prediction=pd.DataFrame(dic)
        prediction.to_csv(self.save_dir+data_name+'prediction.csv')
        
    def save_his(self):
        '''
        save history
        '''
        data_df = pd.DataFrame(self.train_hist)
        data_df.to_csv(self.save_dir+'MS'+self.model_name+'history.csv')
    def load(self):
        
        self.net.load_state_dict(torch.load(self.save_dir +'net_parameters.pkl'))

def data_reader(datadir,gpu=True):
    '''
    read data from mat or other file after readdata.py   
    '''
    datatype=splitext(datadir)[1]
    if datatype=='.mat':

        print('datatype is mat')

        data=scipy.io.loadmat(datadir)
        
        x_train=data['x_train']
        x_test=data['x_test']     
        y_train=data['y_train']     
        y_test=data['y_test']
    if datatype=='':
        pass
    
    x_train=torch.from_numpy(x_train)
    y_train=torch.from_numpy(y_train)
    x_test=torch.from_numpy(x_test)
    y_test=torch.from_numpy(y_test)
    return x_train,y_train,x_test,y_test
    

#%%    
if __name__ == '__main__':
    
    data_root = 'E:/01实验室文件/师兄数据集/datasets/'
    
    CWRU_list = ['CWRU_0hp_10.mat',
                 'CWRU_1hp_10.mat',
                 'CWRU_2hp_10.mat',
                 'CWRU_3hp_10.mat',
                   ]
    LB_list = ['LB_10hz_4.mat',
                 'LB_20hz_4.mat',
                 'LB_40hz_4.mat',
                 'LB_50hz_4.mat',
                   ]
    SUST_list = ['SUST_0.mat',
                 'SUST_20.mat',
                 'SUST_40.mat',
                 'SUST_60.mat',]   
    
    CWRU_list =CWRU_list
    s_dataset_train = Loader_unif_sampling(data_root+CWRU_list[0],
                                     data_root+CWRU_list[1],
                                     data_root+CWRU_list[2],
                                     mode ='train')
    s_dataset_test = Loader_unif_sampling(data_root+CWRU_list[0],
                                     data_root+CWRU_list[1],
                                     data_root+CWRU_list[2],
                                     mode ='test')

    t_dataset = Loader_single(data_root+CWRU_list[3],mode ='test')

    
    diag=Diagnosis(n_class=10,lr=0.001,batch_size=64,domains=3)
    

    diag.fit(s_dataset_train,s_dataset_test,t_dataset,epoches=100)
    
    diag.t_evaluation(t_dataset)

#%%
#    tdatadir=data_root+CWRU_list[1]
#    xt_train,yt_train,xt_test,yt_test=data_reader(tdatadir)
#    diag.evaluation(xt_test,yt_test)
    
#    diag.load() 
    
#    diag.save_prediction(diag.x_test)

#    utils.print_network(diag.net)
    

#%%test
#     diag.net(xs_train[500:510].cuda())