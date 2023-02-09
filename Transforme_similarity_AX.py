# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 16:03:39 2023

@author: epick
"""

from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render
from ax.utils.tutorials.cnn_utils import train, evaluate

# i want to write a function taking a lsit of datasets and creating a sample 
#of couples of spectra of the datasets, balanced for negativity and posiivity
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import pickle
from scipy import signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import random
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
import itertools
import support_functions
from support_functions import perturbation_producer
from support_functions import Virus_processer
from support_functions import processer
from support_functions import contrastive_pairs
from models import siamese_transformer
from support_functions import data_extractor
from support_functions import domain_converter
from support_functions import patch_producer
from support_functions import test_siamese
from support_functions import new_test_siamese
from support_functions import tensor_processer
from support_functions import tensor_contrastive_pairs
from support_functions import general_tensor_processer
from support_functions import valid_epoch
from models import Siamese1DCNN
from models import Siamesehybrid

import os
import time
import math
import random

#datasets should arleady be divided by classes
#virus_base=pd.read_csv(r"C:/Users/epick/OneDrive/Desktop/uni/JP/progetto/SARS-CoV-2_omicron_variant/SARS-CoV-2_omicron_variant/{} 0{} 1000.txt".format('BA.1A',1),delimiter='\t',header=None).values[:,0]
leng= 500

batch_size = 1 #256 #1024
printing = False
epochs=5
lr= 0.00005#0.00005
decay =0.12# 0.01 #0.01
dropout=0
dim = 1024 #256 #2048
num_heads =1
num_layers = 1# 9
batch_inference = 32 #256 con 50 copi # 1024
number_elements = 7
simmetry = True



h1=64
ending_size=3
n_classes=1
conv_size= 3
conv_pooling = [2,1,1,1,1]
conv_kernels = [3,3,3,3,3]
embedding_size = 64
m=123
virus_base=np.linspace(600,1750,leng)

folder  = "C:/Users/epick/OneDrive/Desktop/uni/JP/progetto/Data/virus_spectra" 
data =  Virus_processer(folder, task = 'divide by class', base = virus_base)[3]

n=1024
base= np.zeros(5*n)
container=np.zeros(shape=(100,5*n))

#data extraction and preparation

sugars_list=['Sucrose','Fructose','Glucose','Glucitol' ]
windows_list=['450','1100','1625','2900','3300']
for i,sugar in enumerate(sugars_list):
    for j,number in enumerate(windows_list):
        mappa = (pd.read_csv(r"C:/Users/epick/OneDrive/Desktop/uni/JP/progetto/Data/Data/{}/afterbaseline/{} {}-b.txt".format(sugar,sugar,number),delimiter='\t',header=None).values[1:,2:])
        range_map=( pd.read_csv(r"C:/Users/epick/OneDrive/Desktop/uni/JP/progetto/Data/Data/{}/afterbaseline/{} {}-b.txt".format(sugar,sugar,number),delimiter='\t',header=None).values[0,2:])    
        if i == 0:
           base[j*n:(j+1)*n] = range_map
           
        for k in range(25):
                signal= mappa[k]
        
                padder=[max(range_map)+j*(signal[0]-signal[1]) for j in range(int(n-len(range_map)))]

                signal1=np.append(signal,np.zeros(shape=(int(n-len(signal)))))
                
                container[25*i+k,1024*j:1024*(j+1)]=signal1

base= np.sort(base)
sugars=[]
for i in range(len(sugars_list)):
    sugar=[]
    for j in range(25):
        f=scipy.interpolate.CubicSpline(base[0:3000],container[25*i+j,0:3000])(virus_base) 
        sugar.append(f/f.max())
    sugars.append(sugar)

data1_classes = []
for i in range(30): 
    data1_classes.append([])

data1 = pd.read_csv("C:/Users/epick/OneDrive/Desktop/uni/JP/progetto/test_data.csv/test_data/test_data.csv",header = None)

labels_data1 = pd.read_csv("C:/Users/epick/OneDrive/Desktop/uni/JP/progetto/sample_submission.csv",header = None).values[1:,1].astype(int)
    
data1.fillna(0, inplace=True)
base = data1.values [0,2:]

base,idxs=np.unique(base,return_index=True)
data1 = data1.values[1:,2:]
data1 = data1[:,idxs]
data1 = [(element-np.min(element))/np.max(element) for element in data1]

data1 = [(domain_converter((base),signa,virus_base)) for signa in data1]
for i,element in enumerate(data1):
    data1_classes[labels_data1[i]-1].append(element)
data1 = data1_classes

folder  = "C:/Users/epick/OneDrive/Desktop/uni/JP/progetto/Data/all_together" 

data_test = Virus_processer(folder, task = 'divide by class', base = virus_base)[3]
data_class = []
data_map=[]
folder =  "C:/Users/epick/OneDrive/Desktop/uni/JP/progetto/sugars_company" 
elements = os.listdir(folder)
n_classes = len(elements)

output_domain=np.linspace(600,1750,leng)
for j,element in enumerate(elements):
    temp = pd.read_csv(folder + '/' + element,delimiter= ';').apply(pd.to_numeric, errors='coerce').values[2:-2,1:]
    if j == 0:
        base = np.linspace(100,3700,len(temp))
    base_temp = np.linspace(100,3700,len(temp))
    for i in range(np.shape(temp)[1]):
        transformed = np.zeros(shape=np.shape(np.transpose(temp[:,i])))
        transformed = np.transpose(domain_converter(base_temp,temp[:,i],output_domain))#base
        data_class.append((transformed-np.min(transformed))/np.max(transformed))
    data_map.append(data_class)
    data_class=[]
    
data = data_test + data1  + data_map# + sugars

pairs = tensor_contrastive_pairs(data,number_elements,100000,1,oversampling=[True,3])
print(pairs[2],pairs[3])

pairs = [TensorDataset(torch.tensor(pairs[0]),torch.tensor(pairs[1]))]




#preparing wrapper training function for hyperparameter optimization

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
criterion = nn.MSELoss()

num_epochs=epochs
k=5#5 
splits=KFold(n_splits=k,shuffle=True,random_state=42)
foldperf={}

def train_epoch(model,device,dataloader,loss_fn,optimizer,verbose = False):
    train_loss,train_correct=0.0,0
    model.train()
    count=0
    for signal,label in dataloader:
        label=label.view(label.shape[0],1)

        signal,label = signal.to(device),label.to(device)
        optimizer.zero_grad()
        output = (model(signal))
        count+=1
        label = label.double()
        loss = loss_fn(output,label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * signal.size(0)
        output= output.cpu().detach().numpy()
        output[output>0.5]=1
        output[output<=0.5]=0
    
        label =  label.cpu().detach().numpy()
        

        #scores, predictions = torch.max(output.data, 1)
        #print(((np.array_equal(output , label))))
        result =[]
        for i in range(len(output)):
            result.append(int(output[i]==label[i]))
        train_correct += np.sum(result)#((np.array(output == label).astype(int)) )
        for param in model.parameters():
            a=param.grad

    return train_loss,train_correct

    
def train_and_evaluate_kfold( parameters, verbose = False):
    train = []
    test = []
    total_avg_results = []
    batch_size= parameters.get('batch_size')
    lr = parameters.get('lr')
    decay = parameters.get('decay')
    num_layers = parameters.get('num_layers')

    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(pairs[0])))):
        print('Fold {}'.format(fold + 1))
    
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(pairs[0], batch_size=batch_size, sampler=train_sampler)
        test_loader = DataLoader(pairs[0], batch_size=batch_size, sampler=test_sampler)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #model instantiation
        
        #model = Siamesehybrid(1,m=m,h1=h1,conv_size=conv_size,ending_size = ending_size,conv_pooling = conv_pooling, conv_kernels = conv_kernels, embedding_size = embedding_size,dropout = dropout, leng = 2*leng, last_different = True , nhead = num_heads, dim_feedforward = dim, activation = F.relu, verbose=False, num_layers = num_layers, asymmetric = False, concatenate = False, channels = 1)
    
        model = siamese_transformer(1,2*leng,leng,num_heads,dim,dropout,F.relu,asymmetric = simmetry, concatenate = False ,num_layers = num_layers, normalize =False )
        #model = Siamese1DCNN(1,m=m,h1=h1,conv_size=conv_size,ending_size = ending_size,conv_pooling = conv_pooling,conv_kernels = conv_kernels, embedding_size = embedding_size,dropout = dropout, leng = 2*leng, last_different = False, asymmetric = False, concatenate = False, channels = 1) 
        model=model.double()
        model.to(device)
        optimizer = optim.Adam(model.parameters(),lr=lr,weight_decay=decay)#, weight_decay=0) #0.000001
    
        history = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[]}
        candida_results = []
        sugars_results = []
        omicron_results = []
        average_results = []
        for epoch in range(num_epochs):
            train_loss, train_correct=train_epoch(model,device,train_loader,criterion,optimizer)
            test_loss, test_correct=valid_epoch(model,device,test_loader,criterion,similarity= True,tensor = True)
    
            train_loss = train_loss / len(train_loader.sampler)
            train_acc = train_correct / len(train_loader.sampler) * 100
            test_loss = test_loss / len(test_loader.sampler)
            test_acc = test_correct / len(test_loader.sampler) * 100
            if verbose:
                print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f} AVG Training Acc {:.2f} % AVG Test Acc {:.2f} %".format(epoch + 1,
                                                                                                                         num_epochs,
                                                                                                                         train_loss,
                                                                                                                         test_loss,
                                                                                                                         train_acc,
                                                                                                                         test_acc))
    
            
    
            folder  = "C:/Users/epick/OneDrive/Desktop/uni/JP/progetto/SARS-CoV-2_omicron_variant/readable_by_processer" 
            data_virus_nuovi=[]
            for i in range(len(data_test)):
                for j in range(len(data_test[i])):
                    data_virus_nuovi.append([data_test[i][j],torch.tensor(i)])
            data_virus_nuovone= processer(folder, task = 'divide by class', base= virus_base)[3]
            data_virus_classify= processer(folder, task = 'classification', base= virus_base)[3]
    
            database_virus=[]
            for i in range(len(data_virus_nuovone)):
                database_virus.append([data_virus_nuovone[i],torch.tensor(i)] )
            omicron_results.append(new_test_siamese(model,device,data_virus_classify,database_virus,batch= batch_inference,similarity= True,tensor=True,verbose =[False,False]) )
            print('virus inference: ',omicron_results[-1])
            folder = "C:/Users/epick/OneDrive/Desktop/uni/JP/progetto/Candida individual"
    
            candida_classify  = general_tensor_processer(folder,task='classification',base=virus_base)[3]
            test_datone  = general_tensor_processer(folder,task='divide by class',base=virus_base)[3]
            database_candida = []
            for i in range(len(test_datone)):
                    database_candida.append( [test_datone[i],torch.tensor(i)] )
            candida_results.append(new_test_siamese(model,device,candida_classify,database_candida,batch= batch_inference,similarity= True,tensor=True,verbose =[False,False]) )
            print('candida inference: ',candida_results[-1])
    
    
            sugars_classify  = []
            for i,classe in enumerate(sugars):
                for j,element in enumerate(classe):
                    sugars_classify.append([element,torch.tensor(i)])
            test_datone_sugars  = sugars
            database_sugars = []
            for i in range(len(test_datone_sugars)):
                    database_sugars.append( [test_datone_sugars[i],torch.tensor(i)] )
            sugars_results.append(new_test_siamese(model,device,sugars_classify,database_sugars,batch= batch_inference,similarity= True,tensor=True,verbose =[False,False]) )
            print('sugars inference: ',sugars_results[-1])
    
              
            avg = (candida_results[-1] + sugars_results[-1] +omicron_results[-1])/3
            average_results.append(avg)
            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            train.append(train_acc)
            test.append(test_acc)
        '''
        folder = "C:/Users/epick/OneDrive/Desktop/uni/JP/progetto/Candida data/Candida data/1000"
        
        test_datone  = processer(folder,task='divide by class',base=virus_base)[3]
        test_datone_loader = DataLoader(contrastive_pairs(test_datone,10,9400,1,inference = True)[0],batch_size =1 )
        print(valid_epoch(model,device,test_datone_loader,criterion)[1]/len(test_datone_loader))
        '''
        total_avg_results.append(max(average_results))
        del model 
        del data_virus_nuovone 
        del data_virus_classify
        del database_sugars
        return total_avg_results[-1]


best_parameters, values, experiment, model = optimize(
    parameters=[
        {"name": "lr", "type": "range", "bounds": [1e-6, 0.4], "log_scale": True},
        {"name": "batch_size", "type": "range", "bounds": [256, 2048]},
        {"name": "num_layers", "type": "range", "bounds": [1, 12]},
        {"name": "decay", "type": "range", "bounds": [0.01,0.15], "log_scale": True},
        #{"name": "max_epoch", "type": "range", "bounds": [1, 30]},
        #{"name": "stepsize", "type": "range", "bounds": [20, 40]},        
    ],
  
    evaluation_function= train_and_evaluate_kfold,
    objective_name='accuracy',
)
print(best_parameters)
np.save('best_parameters_transformer',best_parameters)
