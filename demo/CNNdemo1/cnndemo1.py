# %% [markdown]
# # CNN诊断代码Demo
# ## 阅读该代码之后需要习得一下几点：
# 1. 编写智能故障诊断代码的基本逻辑
#     * 数据导入
#     * 数据预处理
#     * 模型建立
#     * 模型训练（本demo，其他的请看相应的函数）
#     * 模型评估与测试 （本demo包括模型评估）
# 2. 编写代码的基本规范
#     * 注释
#     * 空格
#     * 变量命名
#     * 其他请参考 [Google 代码风格](https://github.com/shendeguize/GooglePythonStyleGuideCN)中的第三部分
# 3. 本代码只提供了一个主程序的逻辑，其他相关内容请参考我的[仓库](https://gitee.com/Leaky/diagnosis_for_beginners)
#     * 欢迎师弟师妹一起来贡献仓库，如果觉得麻烦可以把代码发给管理员来更新。
# 4. 本demo文件逻辑
#     - sets.mat **数据集**
#         - 训练集
#         - 测试集
#     - Demo 1 CNN.py or Demo 1 CNN.ipynb **主程序**
#         - 诊断模型类
#             - fit 训练
#             - evaluation 评估
#         - 数据读取
#     - cnn_model.py **模型程序**
#         - 自定义模块，例如1X1卷积
#         - 利用框架构架模型，明确输入输出的size
#         - 主程序随机生成input 测试模型骨架
#     - readdata.py **数据预处理**
#         - shuru 原始数据读取
#         - norm 可以采用sklearn中的库
#         - sampling 从序列中采样样本
#         - readdata类 实现读取-数据预处理-采样的工作流，构建一个样本
#         - dataset 在readdata的基础上重复执行readdate的流，来构建数据集
#         - 主程序 input 数据集地址
#     - utils.py **其他工具**
#         - lossplot 绘制loss
#         - accplot 绘制acc
#         - figscat 绘制散点图
#         - print_network 打印网络结构
#         - initialize_weights 初始化网络参数， xavier 还是kaiming

# %% [markdown]
# - - -
# 1. 首先是在程序最开始时，使用文档字符串注释（'''或"""建议统一为双引号）
# 2. 导入python库
# 3. 其他见备注

# %%
# kaggle 上需要切换工作目录，以下是相关指令
# import os
# os.getcwd()
# os.chdir('/kaggle/input/diagnosis-for-beginner/CNNdemo1')
# os.getcwd()
# ! pip install xlrd

# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一个对CWRU数据集利用CNN进行诊断的Demo

数据集已经处理好，可以直接执行程序，只对该程序详细注释。
根据代码规范不要描述代码,假定阅读代码的人比你更精通Python(他只是不知道你试图做什么)。
这个demo为了师弟师妹更好理解，对代码进行了一定程度的描述，后续的程序只针对功能进行描述。
需要注意的是，测试数据不应该以任何形式在训练阶段出现，这是19年的代码，可能存在部分逻辑问题。

    Enviroment：
        pytorch 0.4 有点老了，后续会更新新版本的demo

    Typical usage example：

    diag=Diagnosis(n_class=10,lr=0.001,batch_size=64,num_train=200)
    datadir='sets.mat'
    x_train,y_train,x_test,y_test=data_reader(datadir)
    diag.fit(x_train,y_train,x_test,y_test,epoches=10)
    diag.evaluation(x_test,y_test)    
    
Created on Tue Jul  9 09:13:34 2019

@author: liqi

Todo：完善各部分功能，但该demo应该不会修改了。
"""
from readdata import dataset  # 数据读取模块 为了提升易读性,行注释应该至少在代码2个空格后,并以#后接至少1个空格开始注释部分.
from cnn_model import CNNmodel  # 模型模块
import pandas as pd
from os.path import splitext
import scipy  # 科学处理库
import os  # os 操作库
import time  # 时间库
import torch  # pytorch 
from torch import nn, optim  # 神经网络， 优化的模块
from torch.utils.data import DataLoader, TensorDataset  # torch 数据导入模块
import numpy as np  
import utils  # 功能

# %% [markdown]
# - - -
# 以下定义了一个诊断的类。

# %%
class Diagnosis():
    """训练诊断模型 #一句话概括这个类的功能 
    
    其他信息
    
    Attributes（可选，可以强调重要的属性）：
        net：选用的诊断模型
        gpu: 
    """
    def __init__(self, n_class=10, lr=0.001, batch_size=64, num_train=100, gpu=True, save_dir='save_dir'):
        """
        初始化函数
        函数应有文档字符串,除非符合以下所有条件:
        外部不可见
        非常短
        简明
        Args:
            n_class:健康状态分类数
            lr:学习率
            batch_size:每次训练样本数量
            num_train:训练数量
            gpu:使用GPU加速
            save_dir:保存训练数据路径

        """
        print('diagnosis begin')  
        self.net = CNNmodel(input_size=32,class_num=10)
        self.gpu = gpu
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if self.gpu:
            self.net.cuda()
        
        self.lr = lr
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=self.lr)
        self.loss_function = nn.CrossEntropyLoss()

        self.batch_size = batch_size

        # 变量命名采用下划线的方式比如该字典代表训练记录 上面的loss_function 代表损失函数

        self.train_hist = {}
        self.train_hist['loss'] = []
        self.train_hist['acc'] = []
        self.train_hist['testloss'] = []
        self.train_hist['testacc'] = []
        # 方法的前后建议空一行

    def caculate_acc(self, prediction, y_):
        """
        计算acc
        Args：
            prediction：模型预测值
            y_：真实标签

        """
        correct = torch.zeros(1).squeeze().cuda()
        total = torch.zeros(1).squeeze().cuda()
        correct += (prediction == y_).sum().float()
        total += len(y_)
        acc = (correct/total).cpu().detach().data.numpy()
        return acc
        
    def fit(self, x_train, y_train, x_test=None, y_test=None, epoches=1):
        """ fit方法（重要的函数需要详细描述功能）
        
        Args:
            x_train: 训练集数据
            y_train: 训练集标签
            x_test:测试数据
            y_test:测试标签
        
        Return：
            训练log ，后续会有专门的log方法，该demo先直接用字典记录
            
        Raises(可选)：
            写error的地方
        
        """

        print('training start!!')        
        if x_test.numpy().any() == None:
            test_flag = None
        else:
            test_flag = True
            
        torch_dataset = TensorDataset(x_train, y_train)  # 读取数据
        loader =DataLoader(
            dataset=torch_dataset,     
            batch_size=self.batch_size,      
            shuffle=True,               
            num_workers=2,              
            )
        whole_time = time.time()
        for epoch in range(epoches):  # 训练开始epoches循环
            loss_epoch = []
            acc_epoch = []
            epoch_start_time = time.time()
            for iter, (x_, y_) in enumerate(loader):  # 一次迭代一个batch
                if self.gpu:
                    x_, y_ = x_.cuda(), y_.cuda()
                # ============================
                # 可以在复杂操作前进行块注释
                # 接下来是进行训练的阶段
                # 前向传播-计算损失-反向传播-优化器更新模型
                # =============================
                self.optimizer.zero_grad()
                y_pre = self.net(x_)
                loss = self.loss_function(y_pre, torch.max(y_, 1)[1])          
                prediction = torch.argmax(y_pre, 1)
                y_ = torch.argmax(y_, 1)
                loss_epoch.append(loss.item())
                loss.backward()
                self.optimizer.step()
                # ============================
                acc = self.caculate_acc(prediction, y_)
                acc_epoch.append(acc)  # 计算acc                
            epoch_end_time = time.time()
            loss = np.mean(loss_epoch)
            acc = np.mean(acc_epoch)
            epoch_time = epoch_end_time-epoch_start_time
            self.train_hist['loss'].append(loss)         
            self.train_hist['acc'].append(acc)
            if test_flag:
                self.evaluation(x_test=x_test,y_test=y_test,train_step=True)
            if epoch % 10 == 0:
                print("Epoch: [%2d] Epoch_time:  [%8f] \n loss: %.8f, acc:%.8f" %  # 打印日志
                              ((epoch + 1), epoch_time, loss, acc))
        total_time = epoch_end_time - whole_time
        print('Total time: %2d'%total_time)
        print('best acc: %4f'%(np.max(self.train_hist['acc'])))
        print('best testacc: %4f'%(np.max(self.train_hist['testacc'])))                
        self.save_his()  # 保存日志
        self.save_prediction(x_test)
        print('=========训练完成=========')        
        return self.train_hist

    def evaluation(self, x_test, y_test, train_step=False):
        """ 评估方法用于验证或者测试
        
        Args:
            x_test:测试数据
            y_test:测试标签
        
        Return：
            训练log ，后续会有专门的log方法，该demo先直接用字典记录
            
        Raises(可选)：
            写error的地方
        
        """
        print('evaluation')
        self.net.eval()  # 进入eval模式，相应的权重更新或者drop的更新关闭
        with torch.no_grad():  # 关闭梯度计算
            if self.gpu:
                x_test = x_test.cuda()
                y_test = y_test.cuda()
            y_test_ori = torch.argmax(y_test, 1)
            y_test_pre = self.net(x_test)
            test_loss = self.loss_function(y_test_pre, y_test_ori)
            y_tset_pre = torch.argmax(y_test_pre, 1)
            acc=self.caculate_acc(y_tset_pre, y_test_ori)
        print("***\ntest result \n loss: %.8f, acc:%.4f\n***" %
              (test_loss.item(),
               acc))
        if train_step:
            self.train_hist['testloss'].append(test_loss.item())
            self.train_hist['testacc'].append(acc)
            if acc >= np.max(self.train_hist['testacc']):
                self.save()
                print(' a better model is saved')
        self.net.train()
        return test_loss.item(), acc

    def save(self):
        """
        save model and its parameters
        """
        torch.save(self.net.state_dict(), self.save_dir+'/net_parameters.pkl')
        
    def save_prediction(self, data, data_name='test'):
        """
        save prediction
        Args:
            data：测试数据
            data_name：保存记录测试结果的文件名称
        """
        if self.gpu:
            data = data.cuda()
        final_pre = self.net(data)
        final_pre = torch.argmax(final_pre, 1).cpu().detach().data.numpy()
        dic = {}
        dic['prediction'] = []
        dic['prediction'] = final_pre
        prediction = pd.DataFrame(dic)
        prediction.to_csv(self.save_dir+'/'+data_name+'prediction.csv')

    def save_his(self):
        """
        save history

        """
        data_df = pd.DataFrame(self.train_hist)
        data_df.to_csv(self.save_dir+'/history.csv')

    def load(self):
        """
        load model
        """
        self.net.load_state_dict(torch.load(self.save_dir +'/net_parameters.pkl'))


def data_reader(datadir,gpu=True):
    """
    read data from mat or other file after readdata.py
    demo只展示了mat的读取
    Args:
        datadir: 加载数据文件名称
    """
    datatype = splitext(datadir)[1]
    if datatype == '.mat':

        print('datatype is mat')

        data = scipy.io.loadmat(datadir)
        
        x_train = data['x_train']
        x_test = data['x_test']
        y_train = data['y_train']
        y_test = data['y_test']
    if datatype == '':
        pass
    
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)
    return x_train, y_train, x_test, y_test

# %% [markdown]
# 下面是主程序，使用`if __name__ == '__main__':`
# 的原因是避免被作为方法导入时执行下面的操作
# 精度较高的原因是CWRU的频谱特征比较明显，在数据预处理后的数据比较好分类

if __name__ == '__main__':

    diag = Diagnosis(n_class=10, lr=0.001, batch_size=64,
                   num_train=200, gpu=True, save_dir='/kaggle/working/save_dir')  # 实例化诊断类 有GPU的请让变量gpu=True
    datadir = 'sets.mat'  # 数据集
    x_train, y_train, x_test, y_test = data_reader(datadir)  # 数据读取
    diag.fit(x_train, y_train, x_test, y_test, epoches=10)  # fit模型
    diag.evaluation(x_test, y_test)  # 评估模型
    utils.print_network(diag.net)  # 展示模型结构




