# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 21:22:51 2018

@author: 李奇
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from os.path import splitext
from scipy.io import loadmat

def data_reader(datadir,gpu=True):
    '''
    read data from mat or other file after readdata.py   
    '''
    datatype=splitext(datadir)[1]
    if datatype=='.mat':

        print('datatype is mat')

        data=loadmat(datadir)
        
        x_train=data['x_train']
        x_test=data['x_test']     
        y_train=data['y_train']     
        y_test=data['y_test']
    if datatype=='':
        pass

    return x_train,y_train,x_test,y_test

#def figscat(samples1,samples2,fig_name,types=10,y_=None):
#    font1 = {'family': 'Times New Roman',
#         'weight': 'normal',
#         'size': 10,
#         }
#    markerlist=['.','o','+','*']
#    clist=['r','b','lightskyblue','lemonchiffon']
#    
#    labels=['Right_predict','Wrong_predict']
#    
#    x1=list(range(0,len(samples1)))
#    x2=list(range(0,len(samples2)))
##    yticks=list(range(types))
#    xticks=samples1.shape[0]
#    yticks=['normal','out02','in02','ball02','out04','in04','ball04','out06','in06','ball06']
#
#    plt.scatter(x1,samples1,marker=markerlist[0],c='',edgecolor=clist[0],label=labels[0])
#    plt.scatter(x2,samples2,marker=markerlist[1],c='',edgecolor=clist[1],label=labels[1])
#
#    plt.xticks([x for x in range(len(samples1)+1) if x%(len(samples1)/types)==0])
#    plt.yticks([y for y in range(types)],yticks)
#    plt.title(fig_name)
#    plt.legend() # 显示图例
##        plt.xlabel('iteration times')
##        plt.ylabel('rate')
#    plt.xlabel('sample', font1)
#    plt.ylabel('faulttype', font1)
#    plt.grid(True,linestyle='-')
#    plt.savefig(fig_name+'.png',dpi=512)
#    plt.show()


def pca_(matname):
    def to_1dim(array):
        return np.mean(array,axis=1)
    
    
    sfis=loadmat(matname)['sfis']
    sfit=loadmat(matname)['sfit']
    tfis=loadmat(matname)['tfis']
    tfit=loadmat(matname)['tfit']

    sfis=to_1dim(sfis)
    sfit=to_1dim(sfit)
    tfis=to_1dim(tfis)
    tfit=to_1dim(tfit)


    
    D2shapes = sfis.shape[1]*sfis.shape[2]#*sfis.shape[3]
        
    pca = PCA(n_components=2)
    
    pca_resultss = pca.fit_transform(sfis.reshape([-1,D2shapes]))
#    pca_resultst = pca.fit_transform(sfit.reshape([-1,D2shapes])) 
#    pca_resultts = pca.fit_transform(tfis.reshape([-1,D2shapes])) 
    pca_resulttt = pca.fit_transform(tfit.reshape([-1,D2shapes])) 
    
    pca_dic = {'s_s':pca_resultss,
               's_t':pca_resultst,
               't_s':pca_resultts,
               't_t':pca_resulttt}
    
    
        
    return pca_dic


        
def tSNE_(matname):#,matname2,matname3
    
#    def to_1dim(array):
#        return np.mean(array,axis=1)
    sfis=loadmat(matname)['Fs1'].reshape(2000,-1)
#    sfit=loadmat(matname)['sfit']
#    tfis=loadmat(matname)['tfis']
    tfit=loadmat(matname)['Fs2'].reshape(2000,-1)
    tfit2=loadmat(matname)['Fs3'].reshape(2000,-1)
    tfit3=loadmat(matname)['Ft'].reshape(2000,-1)
    
#    sfis=to_1dim(sfis)
#    sfit=to_1dim(sfit)
#    tfis=to_1dim(tfis)
#    tfit=to_1dim(tfit)
#    print(sfis)
#    print(sfit)
#    print(tfis)
#    print(tfit)
    
    sfis_plus_tfit = np.append(sfis,tfit,axis=0)
    sfis_plus_tfit = np.append(sfis_plus_tfit,tfit2,axis=0)
    sfis_plus_tfit = np.append(sfis_plus_tfit,tfit3,axis=0)
    
#    def half(data):
#        shape0=data.shape[0]
#        shape1=data.shape[1]
#        newdata=np.zeros((int(shape0/2),shape1))
#        for i in range(40):
#            newdata[i*200:200+i*200]=data[400*i:i*400+200]
#        return newdata
#    sfis_plus_tfit=half(sfis_plus_tfit)
    
    tSNE = TSNE(n_components=2,n_iter=400)
#    print(sfis_plus_tfit)
    tSNE_results = tSNE.fit_transform(sfis_plus_tfit)
#    tSNE_resultst = tSNE.fit_transform(sfit.reshape([-1,D2shapes])) 
#    tSNE_resultts = tSNE.fit_transform(tfis.reshape([-1,D2shapes])) 
#    tSNE_resulttt = tSNE.fit_transform(tfit.reshape([-1,D2shapes])) 
    
#    tSNE_dic = np.append(tSNE_resultss,tSNE_resulttt,axis=0)
    
    
    return tSNE_results


def plot2D(dic,setname='SBDS',save_dir='plot_dir/',TLname='default'):
    #for name,dimen_result in dic.items():
        
    tSNEx=dic[:,0]
    tSNEy=dic[:,1]
    
    
    


#        clist = ['r','y','g','b','c','aqua','lawngreen','lightskyblue','limegreen','lemonchiffon','mediumblue'] 
    clist = ['r','lightsalmon','gold','yellow','palegreen',
             'lightseagreen','lightblue','slateblue','violet','purple']
    labelCWRU = ['NOR','I07','I14','I21',
             'B07','B14','B21',
             'O07','O14','O21'
             ]
    labelSBDB = ['Nor','I02','I04','I06',
     'B02','B04','B06',
     'O02','O04','O06'
     ]
    if setname=='CWRU':
        label=labelCWRU
    elif setname=='SBDB':
        label=labelSBDB

#        clist = ['red',
#                 'chocolate',
#                 'darkorange',
#                 'darkgoldenrod',
#                 'gold','limegreen',
#                 'lightseagreen',
#                 'dodgerblue',
#                 'navy',
#                 'darkorchid']
    
    markerlist=['.',',','o','v','^','<','>','D','*','h','x','+']
    
#    num = 202
    num = 602
    
    sample = 40
    target_idx = 2000
    target_idx2 = 4000

    target_idx3 = 6000
#%%for 10
#    for i in range(10):
#        
#        
#        plt.scatter(tSNEx[0+i*num:sample+i*num],tSNEy[0+i*num:sample+i*num],
#                    s=100,marker=markerlist[i],c='none',edgecolor=clist[-3],label='S0'+label[i])
#
#        plt.scatter(tSNEx[target_idx+i*num:target_idx+sample+i*num],tSNEy[target_idx+i*num:target_idx+sample+i*num],
#                    s=100,marker=markerlist[i],c='none',edgecolor=clist[1],label='S1'+label[i])
#
#        plt.scatter(tSNEx[target_idx2+i*num:target_idx2+sample+i*num],tSNEy[target_idx2+i*num:target_idx2+sample+i*num],
#                    s=100,marker=markerlist[i],c='none',edgecolor=clist[3],label='S2'+label[i])
#
#        plt.scatter(tSNEx[target_idx3+i*num:target_idx3+sample+i*num],tSNEy[target_idx3+i*num:target_idx3+sample+i*num],
#                    s=100,marker=markerlist[i],c='none',edgecolor=clist[5],label='T'+label[i])
#%% for 4
    for i in range(4):
        
        
        plt.scatter(tSNEx[0+i*num:sample+i*num],tSNEy[0+i*num:sample+i*num],
                    s=100,marker=markerlist[i],c='none',edgecolor=clist[-3],label='S0'+label[i])

        plt.scatter(tSNEx[target_idx+i*num:target_idx+sample+i*num],tSNEy[target_idx+i*num:target_idx+sample+i*num],
                    s=100,marker=markerlist[i],c='none',edgecolor=clist[1],label='S1'+label[i])

        plt.scatter(tSNEx[target_idx2+i*num:target_idx2+sample+i*num],tSNEy[target_idx2+i*num:target_idx2+sample+i*num],
                    s=100,marker=markerlist[i],c='none',edgecolor=clist[3],label='S2'+label[i])

        plt.scatter(tSNEx[target_idx3+i*num:target_idx3+sample+i*num],tSNEy[target_idx3+i*num:target_idx3+sample+i*num],
                    s=100,marker=markerlist[i],c='none',edgecolor=clist[5],label='T'+label[i])

        
#    plt.legend(loc='best',fontsize=14, fancybox=True, shadow=True)  
    plt.xticks([])
 
    plt.yticks([])
    plt.savefig(save_dir+TLname+'.svg',format='svg')
    plt.show()
    
        
 
    
if __name__ == '__main__': 
    
#    domains=['datasets/CWRU_0hp_10.mat',
#             'datasets/CWRU_1hp_10.mat',
#             'datasets/CWRU_2hp_10.mat',
#             'datasets/CWRU_3hp_10.mat']
#    name=['CW0','CW1','CW2','CW3']
#    CORALname=['0','1','2','3']
#    methods =['3-0.5INlp-NEW_SBDSCADGN5_times ',
#              'SBDS_no balanceCORAL_5_times ',
#              'SBDS_ERM5_times ',
#              'SBDSJAN_5_times ',
#              'SBDS_no balanceMMD_5_times ']
#    
#    domains=['datasets/SBDS_0K_10.mat',
#             'datasets/SBDS_1K_10.mat',
#             'datasets/SBDS_2K_10.mat',
#             'datasets/SBDS_3K_10.mat']
#    name=['SB0','SB1','SB2','SB3']
#
#    CWRU_list = ['SBDS_0K_10.mat',
#         'SBDS_1K_10.mat',
#         'SBDS_2K_10.mat',
#         'SBDS_3K_10.mat',
#           ]
    DIR = './feature2/'
    import os
    filelist = os.listdir(DIR)
    for file in filelist:
        
        tSNEresult = tSNE_(DIR+file)
        plot2D(tSNEresult,setname='SBDB',save_dir=DIR,TLname=file)
        

        
                

