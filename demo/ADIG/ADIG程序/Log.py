# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 15:13:16 2020

@author: 98133
"""
 
import pandas as pd
import numpy as np
import os
import torch


class loger():
    def __init__(self,head_list,save_dir='./save_dir/',save_name='Default'):
        self.train_his={}

        self.save_dir = save_dir
        self.head_list = head_list
        self.save_name = save_name
        for i in self.head_list:
            self.train_his[i]=[]
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)     
            
    def add(self,value_list):       
        assert len(self.head_list)==len(value_list)
        
        for i,title in enumerate(self.head_list):
            if torch.is_tensor(value_list[i]):
                self.train_his[title].append(value_list[i].item())
            else:
                self.train_his[title].append(value_list[i])
                
    def export_csv(self):
        data_df = pd.DataFrame(self.train_his)

        data_df.to_csv(self.save_dir+self.save_name+'.csv',index=False,header=True)
