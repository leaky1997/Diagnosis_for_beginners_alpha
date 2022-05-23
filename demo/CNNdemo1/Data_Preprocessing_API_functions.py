# -*- coding: utf-8 -*-
"""
Created on: Mob Feb 21 22:48:22 2022

@author: Qitong Chen
"""
import cv2
import numpy as np
from scipy.fftpack import fft
from scipy import signal
import pywt
import matplotlib.pyplot as plt


# =============================================================================================================== #
#                                               1.数据预处理接口函数                                                  #
# =============================================================================================================== #

def raw_data_Func(num_each, sample_lenth, idx, temp):
    """
    原始数据直接输入
    num_each：每个.mat文件所需提取的组数
    sample_lenth：每组包含的数据量
    idx：每组的起始索引
    temp：.mat文件中需要处理的原始数据信号（未按组分割时的数据）

    """
    temp_sample = []
    for i in range(num_each):
        time = temp[idx[i]:idx[i] + sample_lenth]  # time.shape = 2 * sample_length
        temp_sample.append(time)
        pass
    temp_sample = np.array(temp_sample)  # 将temp_sample从列表转换为numpy类型

    return temp_sample


def fft_Func(num_each, sample_lenth, idx, temp):
    """
    快速傅里叶变换FFT
    num_each：每个.mat文件所需提取的组数
    sample_lenth：每组包含的数据量
    idx：每组的起始索引
    temp：.mat文件中需要处理的原始数据信号（未按组分割时的数据）

    """
    temp_sample = []
    for i in range(num_each):
        time = temp[idx[i]:idx[i] + sample_lenth * 2]  # time.shape = 2 * sample_length
        fre = abs(fft(time))[0:sample_lenth]  # 取时间序列的一半进行FFT变换
        temp_sample.append(fre)
        pass
    temp_sample = np.array(temp_sample)  # 将temp_sample从列表转换为numpy类型

    return temp_sample


def  STFT(fl):
    """
    stft函数参数：
    fs:采样频率
    window:窗函数。默认为汉明窗
    nperseg： 每个窗口的长度，默认为256
    noverlap:重叠的点数。指定值时需要满足COLA约束。默认是窗长的一半
    nfft：fft长度
    return: f:采样频率数组；t:段时间数组；Zxx:STFT结果

    """
    f, t, Zxx = signal.stft(fl, nperseg=64)
    img = np.abs(Zxx) / len(Zxx)

    return img


def stft_Func(num_each, sample_lenth, idx, temp):
    """
    短时傅里叶变换STFT
    num_each：每个.mat文件所需提取的组数
    sample_lenth：每组包含的数据量
    idx：每组的起始索引
    temp：.mat文件中需要处理的原始数据信号（未按组分割时的数据）

    """
    temp_sample = []
    for i in range(num_each):
        time = temp[idx[i]:idx[i] + sample_lenth]  # time.shape = 2 * sample_length
        imgs = abs(STFT(time))
        imgs1 = cv2.resize(imgs, (32, 32))
        temp_sample.append(imgs1)
        pass
    temp_sample = np.array(temp_sample)
    temp_sample = temp_sample.reshape(num_each, -1)  # 转换为二维数据，便于后面正则化

    return temp_sample


def CWT(lenth, data, sampling_rate):
    sampling_rate = sampling_rate
    scale = np.arange(1, lenth)
    cwtmatr, freqs = pywt.cwt(data, scale, 'mexh', 1.0 / sampling_rate)

    return cwtmatr


def cwt_Func(num_each, sample_lenth, idx, temp):
    """
    连续小波变换CWT
    num_each：每个.mat文件所需提取的组数
    sample_lenth：每组包含的数据量
    idx：每组的起始索引
    temp：.mat文件中需要处理的原始数据信号（未按组分割时的数据）

    """
    temp_sample = []
    sample_lenth = int(sample_lenth/4)
    for i in range(num_each):
        time = temp[idx[i]:idx[i] + sample_lenth]
        imgs = abs(CWT(sample_lenth+1, time, sampling_rate=sample_lenth))
        imgs1 = cv2.resize(imgs, (32, 32))
        temp_sample.append(imgs1)
        pass

    temp_sample = np.array(temp_sample)
    temp_sample = temp_sample.reshape(num_each, -1)  # 转换为二维数据，便于后面正则化

    return temp_sample


def slice_Func(num_each, sample_lenth, idx, temp):
    """
    由于CNN模型需要四维数据，后面需要先升为2维数据，再升为4维，所以本.py文件中不需使用
    1D---》2D变换
    num_each：每个.mat文件所需提取的组数
    sample_lenth：每组包含的数据量
    idx：每组的起始索引
    temp：.mat文件中需要处理的原始数据信号（未按组分割时的数据）

    """
    temp_sample = []
    for i in range(num_each):
        w = int(np.sqrt(sample_lenth))
        time = temp[idx[i]:idx[i] + sample_lenth]
        imgs = time.reshape(w, w)
        temp_sample.append(imgs)
        pass
    temp_sample = np.array(temp_sample)
    temp_sample = temp_sample.reshape(num_each, -1)  # 转换为二维数据，便于后面正则化
    return temp_sample

# =============================================================================================================== #
#                                                     2.正则化函数                                                  #
# =============================================================================================================== #


# min_max
def min_max_Func(num_each, sample_lenth, idx, temp):
    """
    最大最小值变换
    num_each：每个.mat文件所需提取的组数
    sample_lenth：每组包含的数据量
    idx：每组的起始索引
    temp：.mat文件中需要处理的原始数据信号（未按组分割时的数据）
    """

    temp_sample = []
    for i in range(num_each):
        time = temp[idx[i]:idx[i] + sample_lenth]
        min_data = min(time)
        max_data = max(time)
        new_data = np.zeros(sample_lenth)
        for j in range(sample_lenth):
            new_data[j] = (time[j] - min_data) / (max_data - min_data)
            pass
        print('new_data=', new_data.shape)
        print('new_data=', new_data)
        temp_sample.append(new_data)
        pass
    temp_sample = np.array(temp_sample)
    return temp_sample


# Plus_or_minus_one
def plus_or_minus_one_Func(num_each, sample_lenth, idx, temp):
    """
    [-1, 1]正则化
    num_each：每个.mat文件所需提取的组数
    sample_lenth：每组包含的数据量
    idx：每组的起始索引
    temp：.mat文件中需要处理的原始数据信号（未按组分割时的数据）

    """
    temp_sample = []
    for i in range(num_each):
        time = temp[idx[i]:idx[i] + sample_lenth]
        min_data = min(time)
        max_data = max(time)
        new_data = np.zeros(sample_lenth)
        for j in range(sample_lenth):
            new_data[j] = -1 + 2 * (time[j] - min_data) / (max_data - min_data)
            pass
        temp_sample.append(new_data)
        pass
    temp_sample = np.array(temp_sample)
    return temp_sample


# Z-score正则化
def z_score_Fun(num_each, sample_lenth, idx, temp):
    """
    Z-score Normalization
    num_each：每个.mat文件所需提取的组数
    sample_lenth：每组包含的数据量
    idx：每组的起始索引
    temp：.mat文件中需要处理的原始数据信号（未按组分割时的数据）

    """
    temp_sample = []
    for i in range(num_each):
        time = temp[idx[i]:idx[i] + sample_lenth]
        mean_data = np.mean(time)
        std_data = np.std(time)
        new_data = np.zeros(sample_lenth)
        for j in range(sample_lenth):
            new_data[j] = (time[j] - mean_data) / std_data
            pass
        temp_sample.append(new_data)
        pass
    temp_sample = np.array(temp_sample)
    return temp_sample

