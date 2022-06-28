# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 19:09:41 2019

@author: 李奇
"""
#%%
from cnn_model import CNNmodel
#from SBDS_cnn_model import CNNmodel

#from cnn_model import Dmodel
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
from Log import loger
from AutomaticWeightedLoss import AutomaticWeightedLoss
from data_loader import Loader_single,Loader_unif_sampling,Loader_unif_sampling_DA
from JAN import JAN
#%%
#def data_reader(datadir,gpu=True):
#    '''
#    read data from mat or other file after readdata.py   
#    '''
#    datatype=splitext(datadir)[1]
#    if datatype=='.mat':
#
#        print('datatype is mat')
#
#        data=scipy.io.loadmat(datadir)
#        
#        x_train=data['x_train']
#        x_test=data['x_test']     
#        y_train=data['y_train']     
#        y_test=data['y_test']
#    if datatype=='':
#        pass
#    
#    x_train=torch.from_numpy(x_train)
#    y_train=torch.from_numpy(y_train)
#    x_test=torch.from_numpy(x_test)
#    y_test=torch.from_numpy(y_test)
#    return x_train,y_train,x_test,y_test
#method = 'ADGN'#'JAN''CORAL''ADGN''MMD''WLP0.1''WLP0.5''WLP1''ERM''ERMwln'
class TL_diagnosis():
    def __init__(self,save_dir='save_dir/',
                 target_name='default_source',
                 batchsize=64,Dlr=0.0001,Flr=0.001, class_num=10,domains=3,norm=None):
        print('TL start!!!')
        head_list=['sacc','tacc','closs','dloss','c1','c2']
        self.log_train = loger(head_list,
                          save_dir=save_dir,
                          save_name='train'+target_name)
#        log_test = loger(head_list,
#                         save_dir=save_dir,
#                         save_name='test')        
#to do        
        self.train_hist = {}
        self.train_hist['Dloss'] = []
        self.train_hist['Floss'] = [] #supposed to be 50% 
        self.train_hist['target_acc'] = []
#        
        self.save_dir = save_dir
        self.target_name = target_name
#        self.TLname = TLname
        
        self.batch_size = batchsize
        self.Dlr = Dlr
        self.Flr = Flr
        self.domains = domains
#        self.s_model = CNNmodel()
        self.t_model = CNNmodel(input_size=32, class_num=class_num,
                                domains=self.domains,norm=norm,
                                lambd=1)        
        
#        self.load(self.s_model,loadname=self.source_name)
        self.load(self.t_model,loadname=self.target_name)
        
#        self.s_model.eval()
#        utils.set_requires_grad(self.s_model,requires_grad=False)
#        self.s_f_extractor = self.s_model.feature_extractor
        self.t_f_extractor = self.t_model.feature_extractor
        
        self.CEloss = nn.CrossEntropyLoss()
        self.BCElogits_loss = nn.BCEWithLogitsLoss()
        self.MSE = nn.MSELoss()
        self.balanceloss = AutomaticWeightedLoss(2)
#        self.balanceloss.params=0.3
#        utils.make_cuda(self.d)
#        utils.make_cuda(self.s_f_extractor)
#        utils.make_cuda(self.t_f_extractor)
        utils.make_cuda(self.t_model)
        
    def predict_label(self,prediction):
        pre_sig=torch.t(torch.sigmoid(prediction))
        pre_sig=pre_sig.view(prediction.size()[0],1)
        label_tran = torch.where(pre_sig > 0.5, torch.full_like(pre_sig, 1), torch.full_like(pre_sig, 0))
        label_tran = label_tran.view(prediction.size()[0],1)
        return pre_sig,label_tran        
            
    def fit(self,s_dataset_train,s_dataset_test,t_dataset,method,epoches=100):
        '''
        arg:
            xs:train data from source domain
            ys:train label from source domain
            
            xt:from target domain
            yt:from target domain
        '''        

        timetotal = 0         
        timefirst97 = []         
        
#        s_dataset = TensorDataset(xs,ys)
        s_loader = DataLoader(
                dataset = s_dataset_train,              # torch TensorDataset format
                batch_size = self.batch_size,      # mini batch size
                shuffle = True,                   # 要不要打乱数据 (打乱比较好)
                num_workers=32,                  # 多线程来读数据，提取xy的时候几个数据一起提取
                )
        s_test_loader = DataLoader(
                dataset = s_dataset_test,              # torch TensorDataset format
                batch_size = self.batch_size,      # mini batch size
                shuffle = True,                   # 要不要打乱数据 (打乱比较好)
                num_workers=32,                  # 多线程来读数据，提取xy的时候几个数据一起提取
                )
        
#        t_dataset = TensorDataset(xt,yt)
        t_loader = DataLoader(
                dataset = t_dataset,                  # torch TensorDataset format
                batch_size = self.batch_size,      # mini batch size
                shuffle = True,                       # 要不要打乱数据 (打乱比较好)
                num_workers = 32,                      # 多线程来读数据，提取xy的时候几个数据一起提取
                )
#        D_optim = optim.Adam(self.d.parameters(),lr = self.Dlr)
        target_optim = optim.Adam([{'params':self.t_model.parameters()},
                                   {'params':self.balanceloss.parameters(),
                                    'lr':self.Flr,'weight_decay':0.0001}],
                                    lr = self.Flr,weight_decay=0.0001)
        
        
                                #self.BCE_loss=nn.BCELoss()如果模型有sigmoid就用这个,有sigmoid用下面的效率更高
        
        
        iteration = int(s_dataset_train.length//self.batch_size)            #每个epoch中会迭代几次
        batch_iterator = zip(utils.loop_iterable(s_loader),
                             utils.loop_iterable(s_test_loader))
    
        for epoch in range(1,epoches+1):
            Dloss_epoch = []
            Closs_epoch = []
            acc_epoch = []
            c1_epoch = []
            c2_epoch = []
            epoch_start_time = time.time()
            for _ in range(iteration):
#训练D
                    
#                utils.set_requires_grad(self.t_f_extractor,requires_grad = False)#到底需不需要这一步呢，存疑？？？
#                utils.set_requires_grad(self.d,requires_grad = True)             #因为zero_grad和step只对某个优化器有效
                (x1, x2, x3,y1,y2,y3,d1,d2,d3),(x1_t, x2_t, x3_t,y1_t,y2_t,y3_t,d1_t,d2_t,d3_t) = next(batch_iterator)               #可能可以节省内存，猜的
                # if self.gpu
                xs = torch.cat((x1,x2,x3),dim=0)
                ys = torch.cat((y1,y2,y3),dim=0)
                ds = torch.cat((d1,d2,d3),dim=0)
                
                xs_t = torch.cat((x1_t,x2_t,x3_t),dim=0)
                ys_t = torch.cat((y1_t,y2_t,y3_t),dim=0)
                ds_t = torch.cat((d1_t,d2_t,d3_t),dim=0)

                xs = xs.cuda()
                ys = ys.cuda()
                ds = ds.cuda()#.view(-1,1)
                
                xs_t = xs_t.cuda()
                ys_t = ys_t.cuda()
                ds_t = ds_t.cuda()
                
                x1,x2,x3 = x1.cuda(),x2.cuda(),x3.cuda()#self.to_cuda(x1,x2,x3)

                
                c_pre,d_pre = self.t_model(xs)
#                s_feature = self.s_f_extractor(xs)
#                t_feature = self.t_f_extractor(xt_batch)
                
#                s_t_feature = torch.cat([s_feature,t_feature])
                
#                y_real = torch.ones(xs_batch.size(0),1)
#                y_fake = torch.zeros(xt_batch.size(0),1)
#                y_real = y_real.cuda()
#                y_fake = y_fake.cuda()
                
#                s_t_label = torch.cat([y_real,y_fake])
#计算loss并反向传播
#                prediction = self.d(s_t_feature)
                y1,_ = self.t_model(x1)#.view(xs.size(0),-1)
                y2,_ = self.t_model(x2)#.view(xs2.size(0),-1)
                y3,_ = self.t_model(x3)#.view(xs2.size(0),-1)
                
                Dloss = self.CEloss(d_pre,ds)
                Closs = self.CEloss(c_pre,torch.max(ys, 1)[1])
                
                if method == 'WLP0.5':
                    Total_loss = Closs + 0.5*Dloss
                if method == 'WLP0.1':
                    Total_loss = Closs + 0.1*Dloss
                if method == 'WLP1':
                    Total_loss = Closs + 1*Dloss
                if method == 'ADGN':
                    Total_loss = self.balanceloss(Closs,Dloss)
                if method == 'CADGN':
                    f1 = [y1]
                    f2 = [y2]
                    f3 = [y3]                    
                    DCloss1 = JAN(f1,f2)
                    DCloss2 = JAN(f2,f3)
                    DCloss3 = JAN(f1,f3)                
                    DCloss = DCloss1+DCloss2+DCloss3

                    Dloss = Dloss +DCloss
                    Total_loss = self.balanceloss(Closs,Dloss)
                    
                if method == 'ADGNwln':
                    Total_loss = self.balanceloss(Closs,Dloss)
                    
                if method == 'ERM':                    
                    Total_loss = Closs
                if method == 'ERMwln':
                    Total_loss = Closs
                target_optim.zero_grad()
                Total_loss.backward()
                target_optim.step()
                
                Closs_epoch.append(Closs.item())
                Dloss_epoch.append(Dloss.item())                
#计算self.t_f_extractor+clf 
                
                clf_prediction,_ = self.t_model(xs_t)
                
                clf_prediction = torch.argmax(clf_prediction,1)
                ys_t = torch.argmax(ys_t,1)

                acc=utils.caculate_acc(clf_prediction,ys_t)
                
                acc_epoch.append(acc)
                c1_epoch.append(self.balanceloss.params[0].cpu().detach().numpy())
                c2_epoch.append(self.balanceloss.params[1].cpu().detach().numpy())           
#结束一个epoch计算结果并打印
            target_acc = self.evaluation(t_dataset)   
            Dloss = np.mean(Dloss_epoch)
            Closs = np.mean(Closs_epoch)
            source_acc = np.mean(acc_epoch)
            c1 = np.mean(c1_epoch)
            c2 = np.mean(c2_epoch)            
            print_value = source_acc,target_acc,Closs,Dloss,c1,c2
            self.log_train.add(print_value)
            
#            self.train_hist['Dloss'].append(Dloss)
#            self.train_hist['Floss'].append(Floss)
#            self.train_hist['target_acc'].append(target_acc)
            
            epoch_end_time=time.time()
            epoch_time=epoch_end_time-epoch_start_time
            
            timetotal += epoch_time
            
            if target_acc>0.97:
                timefirst97.append(epoch_time)
            
            if epoch%1==0:
                print('Epoch:[%1d] Epoch time:[%.4f]***'%(epoch,epoch_time))
#                print('   Dloss:[%.4f] Floss:[%.4f]'%(Dloss,Floss))
#                print('   targetacc:[%.4f]'%target_acc)
                print('*source_acc:[%4f],target_acc:[%4f],Closs:[%4f],Dloss:[%4f],c1:[%4f],c2:[%4f]*'%print_value)
            if target_acc>=np.max(self.log_train.train_his['tacc']) :      
                self.save(self.t_model,savename = self.target_name)
            
#        self.t_model.feature_extractor = self.t_f_extractor
#        self.save(self.t_model,savename = self.target_name)
        
        #finalacc = self.evaluation(xt,yt,modeldir = self.TLname)
         
        #self.train_hist['target_acc'].append(finalacc)
        
#        self.save_his(his_name=self.target_name)
        self.log_train.export_csv()
        acc_best = np.max(self.log_train.train_his['tacc'])
        validation_idx =  np.argmax(self.log_train.train_his['sacc'])
        acc_model_selection = self.log_train.train_his['tacc'][validation_idx]
        acc_last = self.evaluation(t_dataset,self.target_name)
        
#        try:
#            timefirst = timefirst97[0]
#        except:
#            timefirst = 0
        return acc_best,acc_model_selection,acc_last
    
    
    def evaluation(self,t_dataset,modeldir='trained_target_model'):
        '''
        evaluate the performence of target domain 
        '''
                
        self.t_model = self.t_model.cuda()
        loadert =DataLoader(
            dataset=t_dataset,     
            batch_size=self.batch_size,      
            shuffle=True,               
            num_workers=32,              
            )
        loss_epoch=[]
        acc_epoch=[]
        with torch.no_grad():
            for iter, (xt,yt) in enumerate(loadert):
#                if self.gpu:
                xt, yt = xt.cuda(), yt.cuda()

                

                y_pre,_ = self.t_model(xt)
            
                loss = self.CEloss(y_pre, torch.max(yt, 1)[1])          
                prediction = torch.argmax(y_pre, 1)
                yt=torch.argmax(yt,1)



                loss_epoch.append(loss.item())

                
                acc=utils.caculate_acc(prediction,yt)
                
                acc_epoch.append(acc)
                    
            loss=np.mean(loss_epoch)
            acc=np.mean(acc_epoch)
        
#        xt = xt.cuda()
#        yt = yt.cuda()
#        prediction = t_model(xt)
#        
#        prediction = torch.argmax(prediction,1)
#        yt = torch.argmax(yt,1)
#        acc = utils.caculate_acc(prediction,yt)
        
        print('the accuracy of the target model is %.4f'%acc)
        
        return acc
        
    def save_his(self,his_name='history.csv'):
        '''
        save history
        '''
        data_df = pd.DataFrame(self.train_hist)
        data_df.to_csv(self.save_dir+'GD'+his_name+'.csv')
        
    
    
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
#experiment 1    
#    domains=['datasets/CWRU_0hp_10.mat',
#             'datasets/CWRU_1hp_10.mat',
#             'datasets/CWRU_2hp_10.mat',
#             'datasets/CWRU_3hp_10.mat']
#    name=['CW0','CW1','CW2','CW3']
    dic = {}     
    dic['targetacc'] = []     
    dic['timefirst'] = []     
    dic['timetotal'] = []     
#experiment 2 

    data_root = '/home/liki/datasets/small/'
    
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
    SBDS_list = ['SBDS_0K_10.mat',
                 'SBDS_1K_10.mat',
                 'SBDS_2K_10.mat',
                 'SBDS_3K_10.mat',
                   ]
    JNU_list = ['JNU_600.mat',
                 'JNU_800.mat',
                 'JNU_1000.mat',
                   ]
    SUST_list = ['SUST_0.mat',
                 'SUST_20.mat',
                 'SUST_40.mat',
                 'SUST_60.mat',]   
    SUST_speed_list = ['SUST_500.mat',
                 'SUST_1000.mat',
                 'SUST_1500.mat',
                 'SUST_2000.mat',]   
#    methods = ['ADGN','WLP0.1','WLP0.5','WLP1','ERM','ERMwln']
#    methods = ['ADGNwln']
    methods = ['ADGN']
    
    for method in methods:
#        method = 'ERMwln'
        if method == 'ADGNwln':#'ERMwln':
            norm_type = 'lnwithout'
        else:
            norm_type = 'ln'
            
        ex_name = '2-0.5INlp-NEW_SUST'+method+'5_times '
        head_list=['iter','acc_best','acc_model_selection','acc_last']
        log_5_times = loger(head_list,
                          save_dir='save_dir/'+ex_name+'/',
                          save_name='5_times')
            
        for j in range(5):
            for i in range(4):
                
                data_root = '/home/liki/datasets/small/'
                
                CWRU_list = ['SUST_500.mat',
                 'SUST_1000.mat',
                 'SUST_1500.mat',
                 'SUST_2000.mat',]
                
                print('target domain is %d'%i)
                
                t_dataset = Loader_single(data_root+CWRU_list[i],mode ='train')
                target_name = CWRU_list[i]
                
                CWRU_list.remove(CWRU_list[i])
                s_dataset_train = Loader_unif_sampling(data_root+CWRU_list[0],
                                                 data_root+CWRU_list[1],
                                                 data_root+CWRU_list[2],
                                                 mode ='train')
                s_dataset_test = Loader_unif_sampling(data_root+CWRU_list[0],
                                                 data_root+CWRU_list[1],
                                                 data_root+CWRU_list[2],
                                                 mode ='test')
                
        
                
                diag = Diagnosis(n_class=10,lr=0.001,batch_size=128,
                         gpu=True,domains=3,save_dir='save_dir/'+ex_name+'/',
                         model_name = target_name,norm=norm_type)
               
                
                
                diag.fit(s_dataset_train,s_dataset_test,t_dataset,epoches = 20,test_flag='t')
                diag.t_evaluation(t_dataset)
        #%%     DG&DA   
                diagtl = TL_diagnosis(save_dir='save_dir/'+ex_name+'/',target_name = target_name,
                                    batchsize = 128,Dlr = 0.0001,
                                    Flr = 0.001, class_num=10,domains=3,norm=norm_type)
                acc_best,acc_model_selection,acc_last = diagtl.fit(s_dataset_train,
                                                                   s_dataset_test,
                                                                   t_dataset,
                                                                   method = method,
                                                                   epoches = 200,
                                                                   )
                log_5_times.add((j,acc_best,acc_model_selection,acc_last))
        log_5_times.export_csv()
#    data_df = pd.DataFrame(dic)     
#    data_df.to_csv('save_dir'+'/'+CWRU_list[0][:4]+'result'+'.csv')         

#%% 
#    dic = {}     
#    dic['targetacc'] = []     
#    dic['timefirst'] = []     
#    dic['timetotal'] = []     
#    
#    for i in range(4):
#        
#        data_root = '/home/liki/datasets/small/'
#        
#        CWRU_list = LB_list
#        
#        print('target domain is %d'%i)
#        
#        t_dataset = Loader_single(data_root+CWRU_list[i],mode ='test')
#        target_name = CWRU_list[i]
#        
#        CWRU_list.remove(CWRU_list[i])
#        s_dataset_train = Loader_unif_sampling(data_root+CWRU_list[0],
#                                         data_root+CWRU_list[1],
#                                         data_root+CWRU_list[2],
#                                         mode ='train')
#        s_dataset_test = Loader_unif_sampling(data_root+CWRU_list[0],
#                                         data_root+CWRU_list[1],
#                                         data_root+CWRU_list[2],
#                                         mode ='test')
#        
#
#        
#        diag = Diagnosis(n_class=10,lr=0.001,batch_size=64,
#                 gpu=True,save_dir='save_dir/',model_name = target_name)
#       
#        
#        
#        diag.fit(s_dataset_train,s_dataset_test,t_dataset,epoches = 10,test_flag='t')
#        diag.t_evaluation(t_dataset)
##%%     DG&DA   
#        diagtl = TL_diagnosis(save_dir='save_dir/',
#                            target_name = target_name,
#                            batchsize = 64,
#                            Dlr = 0.0001,
#                            Flr = 0.001,
#                            domains=3)
#        acc,timefirst,timetotal = diagtl.fit(s_dataset_train,s_dataset_test,t_dataset,epoches = 100)
#        dic['targetacc'].append(acc)         
#        dic['timefirst'].append(timefirst)          
#        dic['timetotal'].append(timetotal)      
#    data_df = pd.DataFrame(dic)     
#    data_df.to_csv('save_dir'+'/'+CWRU_list[0][:4]+'result'+'.csv')   