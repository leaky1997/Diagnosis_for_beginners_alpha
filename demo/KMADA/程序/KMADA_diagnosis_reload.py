# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 19:09:41 2019

@author: 李奇
"""
#%%
from cnn_model import CNNmodel
from cnn_model import Dmodel
from os.path import splitext
import pandas as pd
import scipy
#import os
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
import utils
from diagnosis_demo_cnn import Diagnosis

#%%
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

class TL_diagnosis():
    def __init__(self,save_dir='save_dir/',
                 source_name='default_source',
                 TLname='default',
                 batchsize=64,Dlr=0.0001,Flr=0.001):
        print('TL start!!!')
        self.train_hist = {}
        self.train_hist['Dloss'] = []
        self.train_hist['Floss'] = [] #supposed to be 50% 
        self.train_hist['target_acc'] = []
        self.save_dir = save_dir
        self.source_name = source_name
        self.TLname = TLname
        
        self.batch_size = batchsize
        self.Dlr = Dlr
        self.Flr = Flr
        
        self.s_model = CNNmodel()
        self.t_model = CNNmodel()        
        
        self.load(self.s_model,loadname=self.source_name)
        self.load(self.t_model,loadname=self.source_name)
        
        self.s_model.eval()
        utils.set_requires_grad(self.s_model,requires_grad=False)
        self.s_f_extractor = self.s_model.feature_extractor
        self.t_f_extractor = self.t_model.feature_extractor
        
        self.d = Dmodel()
        

        utils.make_cuda(self.d)
        utils.make_cuda(self.s_f_extractor)
        utils.make_cuda(self.t_f_extractor)
        utils.make_cuda(self.t_model)
        
        
    def fit(self,xs,ys,xt,yt,epoches=100):
        '''
        arg:
            xs:train data from source domain
            ys:train label from source domain
            
            xt:from target domain
            yt:from target domain
        '''        

        
        
        s_dataset = TensorDataset(xs,ys)
        s_loader = DataLoader(
                dataset = s_dataset,              # torch TensorDataset format
                batch_size = self.batch_size,      # mini batch size
                shuffle = True,                   # 要不要打乱数据 (打乱比较好)
                num_workers=1,                  # 多线程来读数据，提取xy的时候几个数据一起提取
                )
        t_dataset = TensorDataset(xt,yt)
        t_loader = DataLoader(
                dataset = t_dataset,                  # torch TensorDataset format
                batch_size = self.batch_size,      # mini batch size
                shuffle = True,                       # 要不要打乱数据 (打乱比较好)
                num_workers = 1,                      # 多线程来读数据，提取xy的时候几个数据一起提取
                )
        D_optim = optim.Adam(self.d.parameters(),lr = self.Dlr)
        target_optim = optim.Adam(self.t_f_extractor.parameters(),
                                  lr = self.Flr)
                                #self.BCE_loss=nn.BCELoss()如果模型有sigmoid就用这个,有sigmoid用下面的效率更高
        self.BCElogits_loss = nn.BCEWithLogitsLoss()
        
        
        iteration = int(xs.size(0)//self.batch_size)            #每个epoch中会迭代几次
        batch_iterator = zip(utils.loop_iterable(s_loader),
                             utils.loop_iterable(t_loader))
    
        for epoch in range(1,epoches+1):
            Dloss_epoch = []
            Floss_epoch = []
            acc_epoch = []
            epoch_start_time = time.time()
            for _ in range(iteration):
#训练D
                    
                utils.set_requires_grad(self.t_f_extractor,requires_grad = False)#到底需不需要这一步呢，存疑？？？
                utils.set_requires_grad(self.d,requires_grad = True)             #因为zero_grad和step只对某个优化器有效
                (xs_batch,_),(xt_batch,_) = next(batch_iterator)               #可能可以节省内存，猜的
                # if self.gpu
                xs_batch = xs_batch.cuda()
                xt_batch = xt_batch.cuda()

                
                s_feature = self.s_f_extractor(xs_batch)
                t_feature = self.t_f_extractor(xt_batch)
                
                s_t_feature = torch.cat([s_feature,t_feature])
                
                y_real = torch.ones(xs_batch.size(0),1)
                y_fake = torch.zeros(xt_batch.size(0),1)
                y_real = y_real.cuda()
                y_fake = y_fake.cuda()
                
                s_t_label = torch.cat([y_real,y_fake])
#计算loss并反向传播
                prediction = self.d(s_t_feature)
                Dloss = self.BCElogits_loss(prediction,s_t_label)
                D_optim.zero_grad()
                Dloss.backward()
                D_optim.step()
                Dloss_epoch.append(Dloss.item())
#训练特征提取器
                utils.set_requires_grad(self.t_f_extractor,requires_grad = True)#到底需不需要这一步呢，存疑？？？
                utils.set_requires_grad(self.d,requires_grad = False)            #因为zero_grad和step只对某个优化器有效
                
                _,(xt_batch,yt_batch) = next(batch_iterator)
                
                xt_batch = xt_batch.cuda()
                yt_batch = yt_batch.cuda()
                
                y_real = torch.ones(xt_batch.size(0),1)
                y_real = y_real.cuda()
                
                t_feature = self.t_f_extractor(xt_batch)
                
                prediction = self.d(t_feature)
                Floss = self.BCElogits_loss(prediction,y_real)         #让提取的特征接近y_real
                target_optim.zero_grad()
                Floss.backward()
                target_optim.step()
                Floss_epoch.append(Floss.item())
#计算self.t_f_extractor+clf 
                self.t_model.feature_extractor = self.t_f_extractor
                
                clf_prediction = self.t_model(xt_batch)
                
                clf_prediction = torch.argmax(clf_prediction,1)
                yt_batch = torch.argmax(yt_batch,1)

                acc=utils.caculate_acc(clf_prediction,yt_batch)
                
                acc_epoch.append(acc)
#结束一个epoch计算结果并打印
            Dloss = np.mean(Dloss_epoch)
            Floss = np.mean(Floss_epoch)
            target_acc = np.mean(acc_epoch)
            self.train_hist['Dloss'].append(Dloss)
            self.train_hist['Floss'].append(Floss)
            self.train_hist['target_acc'].append(target_acc)
            epoch_end_time=time.time()
            epoch_time=epoch_end_time-epoch_start_time
            if epoch%1==0:
                print('Epoch:[%1d] Epoch time:[%.4f]***'%(epoch,epoch_time))
                print('   Dloss:[%.4f] Floss:[%.4f]'%(Dloss,Floss))
                print('   targetacc:[%.4f]'%target_acc)
        self.t_model.feature_extractor = self.t_f_extractor
        self.save(self.t_model,savename = self.TLname)
        
        #finalacc = self.evaluation(xt,yt,modeldir = self.TLname)
         
        #self.train_hist['target_acc'].append(finalacc)
        
        self.save_his(his_name=self.TLname)
        
    def evaluation(self,xt,yt,modeldir='trained_target_model'):
        '''
        evaluate the performence of target domain 
        '''
        t_model = CNNmodel()
        self.load(t_model,loadname = modeldir)
        t_model = t_model.cuda()
        xt = xt.cuda()
        yt = yt.cuda()
        prediction = t_model(xt)
        
        prediction = torch.argmax(prediction,1)
        yt = torch.argmax(yt,1)
        acc = utils.caculate_acc(prediction,yt)
        
        print('the accuracy of the target model is %.4f'%acc)
        
        return acc
        
    def save_his(self,his_name='history.csv'):
        '''
        save history
        '''
        data_df = pd.DataFrame(self.train_hist)
        data_df.to_csv(self.save_dir+his_name+'.csv')
        
    
    
    def load(self,net,loadname='net_parameters'):
        '''
        加载，记得要改名字
        '''
        net.load_state_dict(torch.load(self.save_dir +loadname+'.pkl'))
    
    def save(self,net,savename='net_parameters'):
        '''
        save model and its parameters remember change the ！！！savename！！！！
        '''
        torch.save(net.state_dict(), self.save_dir+savename+'.pkl')

#%%
if __name__ == '__main__':
    import scipy.io as io
    def to_numpy(tensor):
        return tensor.cpu().detach().numpy()
#    domains=['datasets/CWRU_0hp_10.mat',
#             'datasets/CWRU_1hp_10.mat',
#             'datasets/CWRU_2hp_10.mat',
#             'datasets/CWRU_3hp_10.mat']
#    name=['CW0','CW1','CW2','CW3']
    
    
    domains=['datasets/SBDS_0K_10.mat',
             'datasets/SBDS_1K_10.mat',
             'datasets/SBDS_2K_10.mat',
             'datasets/SBDS_3K_10.mat']
    name=['SB0','SB1','SB2','SB3']

    for i in range(4):
        print('source domain is %d'%i)
        source_domain = domains[i]
        source_name = name[i]
        
#        diag = Diagnosis(n_class=10,lr=0.001,batch_size=64,
#                 gpu=True,save_dir='save_dir/',model_name = source_name)
       
        
#        xs_train,ys_train,xs_test,ys_test = data_reader(source_domain)

        for j in range(4):
            if i!= j:
                target_domain = domains[j]
                target_name = name[j]
                
                TLname = source_name + 'to' + target_name
   
                diagtl = TL_diagnosis(save_dir='save_dir_BSTOBSMMD/',
                                    source_name = source_name,
                                    TLname = TLname,
                             batchsize = 64,Dlr = 0.0001,Flr = 0.001)

                xs_train,ys_train,xs_test,ys_test = data_reader(source_domain)
                xt_train,yt_train,xt_test,yt_test = data_reader(target_domain)
                xs_test=xs_test.cuda()
                xt_test=xt_test.cuda()
#source|target data through S_model
                diagtl.load(diagtl.s_model,loadname=diagtl.source_name)
                
                SFE = diagtl.s_model
                
#the last layer of model
                SFE.clf = nn.Sequential(*list(SFE.clf.children())[:-1])
                SFE = SFE.cuda()
                sf_in_s_model= SFE(xs_test)
                tf_in_s_model= SFE(xt_test)
                
#source|target data through T_model
                diagtl.load(diagtl.t_model,loadname=diagtl.TLname)
                TFE = diagtl.t_model
#%%
                
#%%                

                