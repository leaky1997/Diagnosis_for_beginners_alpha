# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 15:20:35 2019

@author: Liqi|leah0o
"""
import numpy as np 
import xlrd
import scipy.io as io
from sklearn.preprocessing import normalize as norm
from scipy.fftpack import fft
from os.path import splitext
from Data_Preprocessing_API_functions import stft_Func, fft_Func, raw_data_Func
 

def shuru(data):
    """
    load data from xlsx
    Args：
        data：加载数据文件名称
    """
    datatype = splitext(data)[1]

    if datatype == '.xlsx':
        excel = xlrd.open_workbook(data)
        sheet = excel.sheets()[1]
        data = sheet.col_values(0)

    if datatype == '.mat':
        matdata = io.loadmat(data)
        fliter_ = filter(lambda x: 'DE_time' in x, matdata.keys())
        fliter_list = [item for item in fliter_]
        idx = fliter_list[0]
        data = matdata[idx][:, 0]

    return data


def meanstd(data):    
    """
    to -1~1不用了，还是用sklearn的比较方便
    Args：
        data：分割好的数据
    """
    for i in range(len(data)):
        datamean = np.mean(data[i])
        datastd = np.std(data[i])
        data[i] = (data[i]-datamean)/datastd
    
    return data


def sampling(data_this_doc, num_each, sample_lenth):
    """
    Args:
        data_this_doc：文件地址
        num_each：训练集的数量
        sample_lenth：采样的长度

    return:
        采样完的数据
    shuru->取长度->归一化
    ------
    note：采用的normalization 真这个话，除以l2范数
    """
    
    temp = shuru(data_this_doc)
    idx = np.random.randint(0, len(temp)-sample_lenth*2, num_each)

    # 数据预处理API
    temp_sample = fft_Func(num_each, sample_lenth=sample_lenth, idx=idx, temp=temp)
    temp_sample = norm(temp_sample)  # 正则化
    
    return temp_sample


class readdata():
    """
    连接数据集、连接标签、输出
    """

    def __init__(self, data_doc, num_each=400, ft=2, sample_lenth=1024):
        """
        Args:
            data_doc:文件集合，即需要连接的文件（97.mat、105.mat等）路径
            num_each:分组数量
            ft:需要连接的文件数量
            sample_lenth:采样长度
        """
        self.data_doc = data_doc
        # 特殊的再计算
        self.num_train = num_each*ft
        
        self.ft = ft
        self.sample_lenth = sample_lenth
        self.row = num_each
    
    def concatx(self):
        """
        连接多个数据
        暂且需要有多少数据写多少数据
        """

        data = np.zeros((self.num_train, self.sample_lenth))
        for i, data_this_doc in enumerate(self.data_doc):
            data[0+i*self.row:(i+1)*self.row] = sampling(data_this_doc, self.row, self.sample_lenth)
        return data

    def labelling(self):   
        """
        根据样本数和故障类型生成样本标签
        one_hot
        """
    
        label = np.zeros((self.num_train, self.ft))
        for i in range(self.ft):
            label[0+i*self.row:self.row+i*self.row, i] = 1

        return label
       
    def output(self):
        """
        输出数据集的数据和标签
        """
        data = self.concatx()
        
        label = self.labelling()
        size = int(float(self.sample_lenth)**0.5)
        data = data.astype('float32').reshape(self.num_train, 1, size, size)
        label = label.astype('float32')
        return data, label
    

def dataset(train_data_name, data_name='sets', num_each=400, sample_lenth=1024, test_rate=0.5):
    """
    根据特定的数据集构建

    Args：
    train_data_name:训练数据文件名称
    num_each：训练数据分组数
    sample_lenth：每次采样点长度
    test_rate：测试率

    """

    test_data_name = train_data_name

    if test_rate == 0:
        testingset = readdata(test_data_name,
                             ft=len(train_data_name),
                             num_each=num_each,
                             sample_lenth=sample_lenth)

        x_test, y_test = testingset.output()
        io.savemat(data_name, {'x_test': x_test, 'y_test': y_test, })
        return x_test, y_test
    else:
        
        trainingset = readdata(train_data_name,
                             ft=len(train_data_name),
                             num_each=num_each,
                             sample_lenth=sample_lenth)
        
        testingset = readdata(test_data_name,
                             ft=len(train_data_name),
                             num_each=int(num_each*test_rate),
                             sample_lenth=sample_lenth)
        
        x_train, y_train = trainingset.output()
        x_test, y_test = testingset.output()
        io.savemat(data_name, {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test, })
        return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    train_data_name=['/home/c/liki/EGAN/自家试验台数据/002/4.xlsx',       
                      '/home/c/liki/EGAN/自家试验台数据/010/4.xlsx',
                      '/home/c/liki/EGAN/自家试验台数据/029/4.xlsx',
                      '/home/c/liki/EGAN/自家试验台数据/053/4.xlsx',
                      '/home/c/liki/EGAN/自家试验台数据/014/4.xlsx',
                      '/home/c/liki/EGAN/自家试验台数据/037/4.xlsx',
                      '/home/c/liki/EGAN/自家试验台数据/061/4.xlsx',
                      '/home/c/liki/EGAN/自家试验台数据/006/4.xlsx',
                      '/home/c/liki/EGAN/自家试验台数据/034/4.xlsx',
                      '/home/c/liki/EGAN/自家试验台数据/057/4.xlsx'
                      ]
    train_data_name_target=['/home/c/liki/EGAN/自家试验台数据/003/4.xlsx',       
                      '/home/c/liki/EGAN/自家试验台数据/011/4.xlsx',
                      '/home/c/liki/EGAN/自家试验台数据/030/4.xlsx',
                      '/home/c/liki/EGAN/自家试验台数据/054/4.xlsx',
                      '/home/c/liki/EGAN/自家试验台数据/015/4.xlsx',
                      '/home/c/liki/EGAN/自家试验台数据/038/4.xlsx',
                      '/home/c/liki/EGAN/自家试验台数据/062/4.xlsx',
                      '/home/c/liki/EGAN/自家试验台数据/007/4.xlsx',
                      '/home/c/liki/EGAN/自家试验台数据/035/4.xlsx',
                      '/home/c/liki/EGAN/自家试验台数据/058/4.xlsx'
                      ]
    CWRUdata=['西储/98.mat',
              '西储/106.mat',
              '西储/170.mat',
              '西储/210.mat',
              '西储/119.mat',
              '西储/186.mat',
              '西储/223.mat',
              '西储/131.mat',
              '西储/198.mat',
              '西储/235.mat'
              ]
    dataset(CWRUdata, data_name='sets_CWRUdata')