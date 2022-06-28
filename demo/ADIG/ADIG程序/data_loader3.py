import torch.utils.data as data

import os
import scipy.io as sio
import h5py
import torch
import numpy as np
from torchvision import transforms
import torchvision


class Loader_single(data.Dataset):
    def __init__(self, mat_path, mode='train', transform=None):
        self.mat_path = mat_path
        mat = sio.loadmat(self.mat_path)
        self.xmode = 'x_'+mode
        self.ymode = 'y_'+mode
        y = mat[self.ymode]
        self.length = len(y)

    def __getitem__(self, idx):
        mat = sio.loadmat(self.mat_path)
        y_task = mat[self.ymode][idx]
        data = mat[self.xmode][idx]
                
        return torch.from_numpy(data), torch.from_numpy(y_task)

    def __len__(self):
        return self.length


class Loader_unif_sampling(data.Dataset):
    def __init__(self, mat_path1, mat_path2, mode='train', transform=None):
        self.mat_path_1 = mat_path1
        self.mat_path_2 = mat_path2
#        self.mat_path_3 = mat_path3
                
        self.mat_1 = sio.loadmat(self.mat_path_1)
        self.mat_2 = sio.loadmat(self.mat_path_2)
#        self.mat_3 = sio.loadmat(self.mat_path_3)
        self.xmode = 'x_'+mode
        self.ymode = 'y_'+mode

        self.transform = transform
        self.len_1 = len(self.mat_1[self.ymode])
        self.len_2 = len(self.mat_2[self.ymode])
#        self.len_3 = len(self.mat_3[self.ymode])
        
        self.length = np.max([self.len_1, self.len_2])
        

    def __getitem__(self, idx):

        idx_1 = idx % self.len_1
        idx_2 = idx % self.len_2
#        idx_3 = idx % self.len_3

             
        y_task_1 = self.mat_1[self.ymode][idx_1]
        y_domain_1 = 0.0
        data_1_pil = self.mat_1[self.xmode][idx_1]#Image.fromarray(np.reshape(mat_1['x_train'][0],(32,32)), 'RGB')

        
        y_task_2 = self.mat_2[self.ymode][idx_2]
        y_domain_2 = 1.0
        data_2_pil = self.mat_2[self.xmode][idx_2]#data_2_pil = Image.fromarray(np.reshape(mat_2['x_train'][0],(32,32)), 'RGB')

        
#        y_task_3 = self.mat_3[self.ymode][idx_3]
#        y_domain_3 = 2.0
#        data_3_pil = self.mat_3[self.xmode][idx_3]#data_3_pil = Image.fromarray(np.reshape(mat_3['x_train'][0],(32,32)), 'RGB')


        if self.transform is not None:
            data_1 = self.transform(data_1_pil)
            data_2 = self.transform(data_2_pil)
#            data_3 = self.transform(data_3_pil)
        else:
            data_1 = torch.from_numpy(data_1_pil)
            data_2 = torch.from_numpy(data_2_pil)
#            data_3 = torch.from_numpy(data_3_pil)
            
        y_task_1 = torch.from_numpy(y_task_1)
        y_task_2 = torch.from_numpy(y_task_2)
#        y_task_3 = torch.from_numpy(y_task_3)

                
        return (data_1, data_2,
                y_task_1,y_task_2,   
                torch.tensor(y_domain_1).long().squeeze(),
                torch.tensor(y_domain_2).long().squeeze())

    def __len__(self):
        return self.length
        
class Loader_unif_sampling_DA(data.Dataset):
    def __init__(self, mat_path1, mat_path2, mat_path3,mat_path4, mode='train', transform=None):
        self.mat_path_1 = mat_path1
        self.mat_path_2 = mat_path2
        self.mat_path_3 = mat_path3
        self.mat_path_4 = mat_path4
                
        self.mat_1 = sio.loadmat(self.mat_path_1)
        self.mat_2 = sio.loadmat(self.mat_path_2)
        self.mat_3 = sio.loadmat(self.mat_path_3)
        self.mat_4 = sio.loadmat(self.mat_path_4)
        
        self.xmode = 'x_'+mode
        self.ymode = 'y_'+mode
        self.transform = transform
        
        self.len_1 = len(self.mat_1[self.ymode])
        self.len_2 = len(self.mat_2[self.ymode])
        self.len_3 = len(self.mat_3[self.ymode])
        self.len_4 = len(self.mat_4[self.ymode])
        
        self.length = np.max([self.len_1, self.len_2, self.len_3,self.len_4])
        

    def __getitem__(self, idx):

        idx_1 = idx % self.len_1
        idx_2 = idx % self.len_2
        idx_3 = idx % self.len_3
        idx_4 = idx % self.len_4

#        mat_1 = sio.loadmat(self.mat_path_1)
#        mat_2 = sio.loadmat(self.mat_path_2)
#        mat_3 = sio.loadmat(self.mat_path_3)
#        mat_4 = sio.loadmat(self.mat_path_4)
             
        y_task_1 = self.mat_1[self.ymode][idx_1]
        y_domain_1 = 0.0
        data_1_pil = self.mat_1[self.xmode][idx_1]

        
        y_task_2 = self.mat_2[self.ymode][idx_2]
        y_domain_2 = 1.0
        data_2_pil = self.mat_2[self.xmode][idx_2]

        
        y_task_3 = self.mat_3[self.ymode][idx_3]
        y_domain_3 = 2.0
        data_3_pil = self.mat_3[self.xmode][idx_3]
        
        y_task_4 = self.mat_4[self.ymode][idx_4]
        y_domain_4 = 3.0
        data_4_pil = self.mat_4[self.xmode][idx_4]


        if self.transform is not None:
            data_1 = self.transform(data_1_pil)
            data_2 = self.transform(data_2_pil)
            data_3 = self.transform(data_3_pil)
            data_4 = self.transform(data_4_pil)
            
        else:
            data_1 = torch.from_numpy(data_1_pil)
            data_2 = torch.from_numpy(data_2_pil)
            data_3 = torch.from_numpy(data_3_pil)
            data_4 = torch.from_numpy(data_4_pil)
            
        y_task_1 = torch.from_numpy(y_task_1)
        y_task_2 = torch.from_numpy(y_task_2)
        y_task_3 = torch.from_numpy(y_task_3)
        y_task_4 = torch.from_numpy(y_task_4)

                
        return (data_1, data_2, data_3,data_4,
                y_task_1,y_task_2,y_task_3,y_task_4,    
                torch.tensor(y_domain_1).long().squeeze(),
                torch.tensor(y_domain_2).long().squeeze(),
                torch.tensor(y_domain_3).long().squeeze(),
                torch.tensor(y_domain_4).long().squeeze())

    def __len__(self):
        return self.length
        
if __name__ == '__main__':
    data_root = 'E:/01实验室文件/师兄数据集/datasets/'
    
    CWRU_list = ['CWRU_0hp_10.mat',
                 'CWRU_1hp_10.mat',
                 'CWRU_2hp_10.mat',
                 'CWRU_3hp_10.mat',
                   ]
    source_1 = data_root+CWRU_list[0]
    source_2 = data_root+CWRU_list[1]
    source_3 = data_root+CWRU_list[2]
    source_4 = data_root+CWRU_list[3]

#    
#    img_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
#
    source_dataset = Loader_unif_sampling(mat_path1=source_1, mat_path2=source_2, mat_path3=source_3)
    source_loader = torch.utils.data.DataLoader(dataset=source_dataset, batch_size=32, shuffle=True, num_workers=0)

    ax,ay,ad, bx,by,bd, cx,cy,cd = source_dataset.__getitem__(100)
            
    print(ax.size(), ax.min(), ax.max())
#%%    
    source_dataset = Loader_unif_sampling_DA(mat_path1=source_1, mat_path2=source_2, mat_path3=source_3,mat_path4=source_4)
    source_loader = torch.utils.data.DataLoader(dataset=source_dataset, batch_size=32, shuffle=True, num_workers=0)
    ax,ay,ad, bx,by,bd, cx,cy,cd,_,_,_ = source_dataset.__getitem__(100)
            
    print(bx.size(), bx.min(), bx.max())
#    source_3 = './bears/SBDS_2K_10.mat'
#    target_dataset = Loader_validation(source_3)
#    source_loader = torch.utils.data.DataLoader(dataset=target_dataset, batch_size=32, shuffle=True, num_workers=0)
#    a, b, c = target_dataset.__getitem__(100)
#    print(a.size(), a.min(), a.max())
#    source_3 = './bears/SBDS_2K_10.mat'
#    target_dataset = Loader_validation(source_3)
#    source_loader = torch.utils.data.DataLoader(dataset=target_dataset, batch_size=32, shuffle=True, num_workers=0)
#    a, b, c = target_dataset.__getitem__(10)
#    print(a.size(), a.min(), a.max())