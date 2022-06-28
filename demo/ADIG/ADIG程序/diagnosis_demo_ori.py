#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 09:13:34 2019

@author: liqi
"""
#from ShuffleNetV2 import ShuffleNetV2 as sfmodel
#%%
from readdata import dataset
from cnn_model import CNNmodel
from os.path import splitext
import pandas as pd
import scipy
import os
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
import utils
from Log import loger
#%%

class Diagnosis():
    '''
    diagnosis model
    '''
    def __init__(self,n_class=10,lr=0.001,batch_size=64,
                 gpu=True,save_dir='save_dir/',model_name='default'):
        print('diagnosis begin')
        
        
        self.net=CNNmodel(input_size=32,class_num=n_class,norm='ln')
        self.gpu=gpu
        self.save_dir=save_dir
        
        self.model_name=model_name
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if self.gpu:
            self.net.cuda()
        
        self.lr=lr
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,step_size=5,gamma = self.lr)
        self.loss_function = nn.CrossEntropyLoss()

        self.batch_size=batch_size

        self.train_hist = {}
        self.train_hist['loss']=[]
        self.train_hist['acc']=[]
        self.train_hist['testloss']=[]
        self.train_hist['testacc']=[]
        
    def caculate_acc(self,prediction,y_):
        '''
        计算acc
        '''
        
        if self.gpu:
            correct = torch.zeros(1).squeeze().cuda()
            total = torch.zeros(1).squeeze().cuda()
        else:
            correct = torch.zeros(1).squeeze()
            total = torch.zeros(1).squeeze()
        correct += (prediction == y_).sum().float()
        total+=len(y_) 
        acc=(correct/total).cpu().detach().data.numpy()
        return acc
        
    def fit(self,x_train,y_train,x_test=None,y_test=None,epoches=1):
        '''
        x_train,y_train,x_test=False,y_test=False,epoches=1
        '''
        print('training start!!')
        
        if x_test.numpy().any()==None:
            test_flag=None
        else:
            test_flag=True
            
        torch_dataset = TensorDataset(x_train,y_train)
        loader =DataLoader(
            dataset=torch_dataset,     
            batch_size=self.batch_size,      
            shuffle=True,               
            num_workers=2,              
            )
        
        whole_time=time.time()
        
        
        for epoch in range(epoches):
            loss_epoch=[]
            acc_epoch=[]

            
            epoch_start_time = time.time()
            for iter, (x_, y_) in enumerate(loader):
                if self.gpu:
                    x_, y_ = x_.cuda(), y_.cuda()
                
                self.optimizer.zero_grad()
                y_pre,_ = self.net(x_)
                loss = self.loss_function(y_pre, torch.max(y_, 1)[1])          
                prediction = torch.argmax(y_pre, 1)
                y_=torch.argmax(y_,1)



                loss_epoch.append(loss.item())
                loss.backward()
                self.optimizer.step()
                
                acc=self.caculate_acc(prediction,y_)
                
                acc_epoch.append(acc)
                
            epoch_end_time = time.time()
            
            loss=np.mean(loss_epoch)
            acc=np.mean(acc_epoch)
            epoch_time=epoch_end_time-epoch_start_time
            self.train_hist['loss'].append(loss)         
            self.train_hist['acc'].append(acc)
            if test_flag:
                self.evaluation(x_test=x_test,y_test=y_test,train_step=True)

            if epoch%5==0:
                print("Epoch: [%2d] Epoch_time:  [%8f] \n loss: %.8f, acc:%.8f" %
                              ((epoch + 1),epoch_time,loss,acc))

            



        total_time=epoch_end_time-whole_time
        print('Total time: %2d'%total_time)
        print('best acc: %4f'%(np.max(self.train_hist['acc'])))
        print('best testacc: %4f'%(np.max(self.train_hist['testacc'])))
        

                
                
        self.save_his()
        self.save_prediction(x_test)
        print('=========训练完成=========')
        
        return self.train_hist
    def evaluation(self,x_test,y_test,train_step=False):
        '''
        x_test,y_test,train_step=False
        '''
        print('evaluation')
        
        self.net.eval()
        with torch.no_grad(): 
            if self.gpu:
                x_test=x_test.cuda()
                y_test=y_test.cuda()
                
            
            
            y_test_ori=torch.argmax(y_test,1)
            y_test_pre,_=self.net(x_test)
            test_loss=self.loss_function(y_test_pre,y_test_ori)
            y_tset_pre=torch.argmax(y_test_pre,1)
            

            acc=self.caculate_acc(y_tset_pre,y_test_ori)
        

        
        print("***\ntest result \n loss: %.8f, acc:%.4f\n***" %
              (test_loss.item(),
               acc))
        
        if train_step:
            self.train_hist['testloss'].append(test_loss.item())
            self.train_hist['testacc'].append(acc)
            
            if acc>=np.max(self.train_hist['testacc']) :      
                self.save()
                
                print(' a better model is saved')

              


        self.net.train()
        return test_loss.item(),acc
    
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
        data_df.to_csv(self.save_dir+self.model_name+'history.csv')
    def load(self):
        
        self.net.load_state_dict(torch.load(self.save_dir +self.model_name+'.pkl'))

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
    
#    sdatadir='datasets/CWRU_0hp_10.mat'#be careful to capital and small letter
#    tdatadir='datasets/CWRU_1hp_10.mat'
    data_root = '/home/liki/datasets/small/'
    SUST_list = ['SUST_0.mat',
                 'SUST_20.mat',
                 'SUST_40.mat',
                 'SUST_60.mat',]   
    SUST_speed_list = ['SUST_500.mat',
                 'SUST_1000.mat',
                 'SUST_1500.mat',
                 'SUST_2000.mat',]   
    
#    data_root = '/home/liki/datasets/bearings/'
    
    SBDS_list = ['SUST_500.mat',
                 'SUST_1000.mat',
                 'SUST_1500.mat',
                 'SUST_2000.mat',]
    listtrain = SBDS_list
    
    head_list=['iter','acc_best','source','target']    
    log_5_times = loger(head_list,
                      save_dir='save_dir/'+'SUSTERM'+'/',
                      save_name='SBDSERMlog'+'5_times') 
#    for k in range(5):
#        for i in range(4):
#            for j in range(4):
#                if i!= j:
#                    tdatadir = data_root+listtrain[i]
#                    sdatadir = data_root+listtrain[j]
#                    diag=Diagnosis(n_class=10,lr=0.001,batch_size=64,
#                                   save_dir='save_dir/'+'SUSTERM'+'/'
#                                   ,model_name = 't:'+str(i)+'s:'+str(j))
#                    xs_train,ys_train,xs_test,ys_test=data_reader(sdatadir)
#                    xt_train,yt_train,xt_test,yt_test=data_reader(tdatadir)
#                    diag.fit(xs_train,ys_train,xs_test,ys_test,epoches=201)
#                    _,acc = diag.evaluation(xt_test,yt_test)
#                    log_5_times.add((k,acc,'s:'+str(j),'t:'+str(i)))
#    log_5_times.export_csv()
#%%
    import scipy.io as io    
    for i in range(3,4):
        for j in range(4):
            if i!= j:
                CWRU_list = ['SUST_500.mat',
                 'SUST_1000.mat',
                 'SUST_1500.mat',
                 'SUST_2000.mat',]
                diag=Diagnosis(n_class=10,lr=0.001,batch_size=64,
                               save_dir='save_dir/'+'SUSTERM'+'/'
                               ,model_name = 't:'+str(i)+'s:'+str(j))
                target_name = CWRU_list[i]
                target_domain = data_root+CWRU_list[i]
                CWRU_list.remove(CWRU_list[i])
                
                s1_domain = data_root+CWRU_list[0]
                s2_domain = data_root+CWRU_list[1]
                s3_domain = data_root+CWRU_list[2]
                
                
                xs_train,ys_train,s1,ys_test = data_reader(s1_domain)
                xs_train,ys_train,s2,ys_test = data_reader(s2_domain)
                xs_train,ys_train,s3,ys_test = data_reader(s3_domain)
                xs_train,ys_train,target,ys_test = data_reader(target_domain)
                diag.load()
                E = diag.net.cpu().mid_rep
                
                Fs1 = E(s1,0,7,0).view(2000,-1)
                
                Fs2 = E(s2,0,7,0).view(2000,-1)
                
                Fs3 = E(s3,0,7,0).view(2000,-1)
                Ft = E(target,0,7,0).view(2000,-1)
                
                
                Fs1 = Fs1.detach().numpy()
                Fs2 = Fs2.detach().numpy()
                Fs3 = Fs3.detach().numpy()
                Ft = Ft.detach().numpy()
                
    #            print(Fs1.type)
                
                save_dir2 = 'save_dir/feature/'
                
                if not os.path.exists(save_dir2):
                    os.makedirs(save_dir2)
        
                io.savemat(save_dir2+'ERM single t:'+str(i)+'s:'+str(j)+'SUSTfeatures.mat',
                           {'Fs1': Fs1,
                            'Fs2': Fs2,
                            'Fs3': Fs3,
                            'Ft': Ft,
                            })        

#%%
#    sdatadir = data_root+SUST_speed_list [0]
#    tdatadir = data_root+SUST_speed_list [1]
#    diag=Diagnosis(n_class=10,lr=0.001,batch_size=64,save_dir='save_dir/'+'sbdsERM'+'/'
#                   ，model_name='default')
#    
#    xs_train,ys_train,xs_test,ys_test=data_reader(sdatadir)
#    xt_train,yt_train,xt_test,yt_test=data_reader(tdatadir)
#    diag.fit(xs_train,ys_train,xs_test,ys_test,epoches=10)
##    diag.evaluation(xs_test,ys_test)
#    diag.evaluation(xt_test,yt_test)
#    def to_numpy(tensor):
#        return tensor.cpu().detach().numpy()
#    
#    xshat = to_numpy(diag.net(xs_test.cuda())[0])
#    xthat = to_numpy(diag.net(xt_test.cuda())[0])
#    diag.load() 
    
#    diag.save_prediction(diag.x_test)

#    utils.print_network(diag.net)
    

#%%
