# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 02:53:50 2022

@author: epick
"""
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
from sklearn.preprocessing import MinMaxScaler
from support_functions import data_extractor
from support_functions import domain_converter
from support_functions import patch_producer
from support_functions import test_siamese
from support_functions import intra_class_oversampling

import time
import math
import random
#models:
    
class Transformer1d(nn.Module):
    """
    
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)
        
    Output:
        out: (n_samples, n_classes)
        
    Pararmetes:
        
    """

    def __init__(self, n_classes, n_length, d_model, nhead, dim_feedforward, dropout, activation,num_layers, verbose=False):
        super(Transformer1d, self).__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.n_length = n_length
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.n_classes = n_classes
        self.verbose = verbose
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=self.nhead, 
            dim_feedforward=self.dim_feedforward, 
            dropout=self.dropout,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)#6
        self.dense = nn.Linear(self.d_model, self.n_classes)
        #self.softmax = nn.Softmax(dim=0)
        
    def forward(self, x):
        #x = x.view(x.shape[0],1,self.n_length)

        out = x
        if self.verbose:
            print('input (n_samples, n_channel, n_length)', out.shape)
        #out = out.permute( 0,1)
        if self.verbose:
            print('transpose (n_length, n_samples, n_channel)', out.shape)

        out = self.transformer_encoder(out)
        if self.verbose:
            print('transformer_encoder', out.shape)

        #out = out.mean(0)
        if self.verbose:
            print('global pooling', out.shape)

        out = self.dense(out)
        if self.verbose:
            print('dense', out.shape)

       # out = self.softmax(out)
        if self.verbose:
            print('softmax', out.shape)
        return out    
    
    
    
    

class Conv(nn.Module):
    def __init__(self,n_classes,h1=256,embedding_size = 256,m =266,dropout = 0, conv_size = 3, conv_pooling = [2,2,4,1,2],last_different = False,channels = 1,ending_size = 1, batch_norm = True, conv_kernels = [32,32,32],softmax = False  ): #256, 128
        super(Conv, self).__init__()
        self.batch_norm = batch_norm
        self.conv_size = conv_size
        self.conv_pooling=conv_pooling
        self.convs = nn.ModuleList()
        self.batchnorm = nn.ModuleList()
        self.channels = channels
        self.softmax = softmax
        for k in range((conv_size)):
            if(k==0):
               self.convs.append(nn.Conv1d(channels, embedding_size, kernel_size = conv_kernels[k])) 
               self.add_module("conv"+str(k+1), self.convs[-1])
               self.batchnorm.append(nn.BatchNorm1d(embedding_size))

            elif(k==(conv_size-1) and last_different):
                self.convs.append(nn.Conv1d(embedding_size, ending_size, kernel_size = conv_kernels[k])) 
                self.add_module("conv"+str(k+1), self.convs[-1])
                self.batchnorm.append(nn.BatchNorm1d(ending_size))
            else:
                self.convs.append(nn.Conv1d(embedding_size, embedding_size, kernel_size = conv_kernels[k]))
                self.add_module("conv"+str(k+1), self.convs[-1])
                self.batchnorm.append(nn.BatchNorm1d(embedding_size))
            
        self.drop1=nn.Dropout(p=dropout) 
        if last_different:
            self.fc1 = nn.Linear(ending_size*m, h1)
        else:
            self.fc1 = nn.Linear(embedding_size*m, h1)

        self.drop2=nn.Dropout(p=dropout)

        self.fc2 = nn.Linear(h1, n_classes)
        if(softmax):
            self.softmaxlayer = torch.nn.Softmax(1)
       # self.logsoftmax= torch.nn.LogSoftmax(1)
    
    def forward(self, x):
        x = x.view(x.size(dim=0),self.channels,int( x.nelement()/(self.channels*x.shape[0])))
       
        for i in range(self.conv_size):
            x = F.relu(self.convs[i](x))
            
            x = F.max_pool1d(x, kernel_size= self.conv_pooling[i]) 
            
            x = self.drop1(x)

            if (self.batch_norm):
                x = self.batchnorm[i](x)
                


        x = x.view(x.size(0),-1)

        x = F.relu(self.fc1(x))
        x = self.drop2(x)

        x = self.fc2(x)
        if self.softmax:
            x = self.softmaxlayer(x)
        return x
    
class Conv2D(nn.Module):
    def __init__(self,n_classes,h1=256,embedding_size = 256,m =266, width = 20, dropout = 0, conv_size = 3, conv_pooling = [2,2,4,1,2],last_different = False,channels = 1,ending_size = 1, batch_norm = True, conv_kernels = [32,32,32],softmax = False  ): #256, 128
        super(Conv2D, self).__init__()
        self.batch_norm = batch_norm
        self.conv_size = conv_size
        self.conv_pooling=conv_pooling
        self.convs = nn.ModuleList()
        self.batchnorm = nn.ModuleList()
        self.channels = channels
        self.softmax = softmax
        self.width = width
        for k in range((conv_size)):
            if(k==0):
               self.convs.append(nn.Conv2d(channels, embedding_size, kernel_size = conv_kernels[k])) 
               self.add_module("conv"+str(k+1), self.convs[-1])
               self.batchnorm.append(nn.BatchNorm2d(embedding_size)) 

            elif(k==(conv_size-1) and last_different):
                self.convs.append(nn.Conv2d(embedding_size, ending_size, kernel_size = conv_kernels[k])) 
                self.add_module("conv"+str(k+1), self.convs[-1])
                self.batchnorm.append(nn.BatchNorm2d(ending_size))
            else:
                self.convs.append(nn.Conv2d(embedding_size, embedding_size, kernel_size = conv_kernels[k]))
                self.add_module("conv"+str(k+1), self.convs[-1])
                self.batchnorm.append(nn.BatchNorm2d(embedding_size))
            
        self.drop1=nn.Dropout(p=dropout) 
        if last_different:
            self.fc1 = nn.Linear(ending_size*m, h1)
        else:
            self.fc1 = nn.Linear(embedding_size*m, h1)

        self.drop2=nn.Dropout(p=dropout)

        self.fc2 = nn.Linear(h1, n_classes)
        if(softmax):
            self.softmaxlayer = torch.nn.Softmax(1)
       # self.logsoftmax= torch.nn.LogSoftmax(1)
    
    def forward(self, x):
        x = x.view(x.size(dim=0),self.channels,self.width-1,int( x.nelement()/(self.channels*x.shape[0]*(self.width-1))))
       
        for i in range(self.conv_size):
            x = F.relu(self.convs[i](x))
            
            x = F.max_pool2d(x, kernel_size= self.conv_pooling[i]) 
            
            x = self.drop1(x)

            if (self.batch_norm):
                x = self.batchnorm[i](x)
                


        x = x.view(x.size(0),-1)

        x = F.relu(self.fc1(x))
        x = self.drop2(x)

        x = self.fc2(x)
        if self.softmax:
            x = self.softmaxlayer(x)
        return x    
    
class siamese_transformer(nn.Module):
    def __init__(self, n_classes, n_length, d_model, nhead, dim_feedforward, dropout, activation, verbose=False, num_layers = 6,concatenate = False, asymmetric = False, normalize = False):
        super(siamese_transformer, self).__init__()
        self.num_layers=num_layers
        self.concatenate = concatenate
        self.asymmetric = asymmetric 
        self.d_model = d_model
        self.nhead = nhead
        self.normalize =normalize
        self.n_length = n_length
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.n_classes = n_classes
        self.verbose = verbose
        self.batchnorm = nn.BatchNorm1d(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=self.nhead, 
            dim_feedforward=self.dim_feedforward, 
            dropout=self.dropout,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)#6
        if asymmetric:
            self.transformer_encoder2 = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dense = nn.Linear(self.d_model+int(concatenate)*self.d_model, self.n_classes)
        #self.softmax = nn.Softmax(dim=0)
        
    def forward(self, x):
        x = x.view(x.shape[0],1,self.n_length)
        x1=x[:,0,:int(self.n_length/2)]
        x2=x[:,0,int(self.n_length/2):]

        out1 = x1
        if self.verbose:
            print('input (n_samples, n_channel, n_length)', out1.shape)
        #out1 = out1.permute( 0,1)
        if self.verbose:
            print('transpose (n_length, n_samples, n_channel)', out1.shape)

        out1 = self.transformer_encoder(out1)
        if self.verbose:
            print('transformer_encoder', out1.shape)

        #out1 = out1.mean(0)
        if self.verbose:
            print('global pooling', out1.shape)
        
        out2 = x2
        if self.verbose:
            print('input (n_samples, n_channel, n_length)', out2.shape)
        #out2 = out2.permute( 0,1)
        if self.verbose:
            print('transpose (n_length, n_samples, n_channel)', out2.shape)
        if self.asymmetric:
            out2 = self.transformer_encoder2(out2)
        else:
            out2 = self.transformer_encoder(out2)
        if self.verbose:
            print('transformer_encoder', out2.shape)

        #out2 = out2.mean(0)
        if self.verbose:
            print('global pooling', out2.shape)
        if (self.concatenate == True):
            output = torch.cat((out1, out2), 1)
        else:
            #print(out1.shape)
            output = torch.abs(out1 - out2) #torch.norm(out1- out2, dim=1)
            #output = torch.norm(out1- out2, dim=1)
            #print(output.shape)
        if self.normalize:
            output = (output - output.min()) / output.max()
       # output = output.view(output.shape[0],1, leng)
        #output = self.batchnorm(output)
        output = self.dense(output)
        if not self.concatenate:
            output = torch.sigmoid(output)

        if self.verbose:
            print('dense', output.shape)

       # out = self.softmax(out)

        return output
    
class Siamese2DCNN(nn.Module):
    def __init__(self,n_classes,h1=256,conv_size=3,ending_size = 1,width=20,conv_pooling = [2,2,1,1,1],conv_kernels = [3,3,3], embedding_size = 256,dropout = 0, leng = 485, last_different = False, asymmetric = True, concatenate = True, channels = 1): #256, 128
        super(Siamese2DCNN, self).__init__()
        self.convs = nn.ModuleList()
        self.batchnorm = nn.ModuleList()
        self.conv_size = conv_size
        self.width = width
        self.channels = channels
        self.leng = leng
        self.ending_size = ending_size
        self.conv_kernels = conv_kernels
        self.leng = 45
        m= 117
        self.asymmetric = asymmetric
        for k in range((conv_size)):
            if(k==0):
               self.convs.append(nn.Conv2d(channels, embedding_size, kernel_size = conv_kernels[k])) 
               self.add_module("conv"+str(k+1), self.convs[-1])
               self.batchnorm.append(nn.BatchNorm2d(embedding_size))

            elif(k==(conv_size-1) and last_different):
                self.convs.append(nn.Conv2d(embedding_size, ending_size, kernel_size = conv_kernels[k])) 
                self.add_module("conv"+str(k+1), self.convs[-1])
                self.batchnorm.append(nn.BatchNorm2d(ending_size))
            else:
                self.convs.append(nn.Conv2d(embedding_size, embedding_size, kernel_size = conv_kernels[k]))
                self.add_module("conv"+str(k+1), self.convs[-1])
                self.batchnorm.append(nn.BatchNorm2d(embedding_size))
                
            
        self.drop1=nn.Dropout(p=dropout) 
        if (last_different):
            self.fc1 = nn.Linear(ending_size*m, h1)       
        else:
            self.fc1 = nn.Linear(embedding_size*m, h1)
        self.drop2=nn.Dropout(p=dropout)
        self.convs_second = nn.ModuleList()
        self.batchnorm_second = nn.ModuleList()

        for k in range((conv_size)):
            if(k==0):
                if(asymmetric):
                    self.convs_second.append(nn.Conv2d(channels, embedding_size, kernel_size = conv_kernels[k])) 
                    self.add_module("conv_second"+str(k+1), self.convs[-1])
                    self.batchnorm_second.append(nn.BatchNorm2d(embedding_size))
            elif(k==(conv_size-1) and last_different):
                if(asymmetric):
                    self.convs_second.append(nn.Conv2d(embedding_size, ending_size, kernel_size = conv_kernels[k])) 
                    self.add_module("conv_second"+str(k+1), self.convs[-1])
                    self.batchnorm_second.append(nn.BatchNorm2d(ending_size))

            else:
                if(asymmetric):
                    self.convs_second.append(nn.Conv2d(embedding_size, embedding_size, kernel_size = conv_kernels[k]))
                    self.add_module("conv_second"+str(k+1), self.convs[-1])
                    self.batchnorm_second.append(nn.BatchNorm2d(embedding_size))

        
        if(asymmetric):
        
            if (last_different):
    
                self.fc1_second = nn.Linear(ending_size*m, h1)       
            else:
                self.fc1_second = nn.Linear(embedding_size*m, h1)

        
        self.concatenate = concatenate
        self.fc2 = nn.Linear(h1+ int(concatenate)*h1, n_classes)

    
    def forward(self, x, z ):
        
        x = x.view(x.shape[0],self.channels,self.width-1,self.leng)
       
        for i in range(self.conv_size):
            x = F.relu(self.convs[i](x))
            
            x = F.max_pool2d(x, kernel_size= self.conv_pooling[i])
            
            x = self.batchnorm[i](x)


        x = self.drop1(x)

        x = x.view(x.size(0),-1)

        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        
        z = z.view(z.shape[0],self.channels,self.width-1,self.leng)
        
        if(self.asymmetric):
            for i in range(self.conv_size):
                z = F.relu(self.convs_second[i](z))
                
                z = F.max_pool2d(z, kernel_size= self.conv_pooling[i]) 
                z = self.batchnorm[i](z)
        else:
            for i in range(self.conv_size):
                z = F.relu(self.convs[i](z))
                
                z = F.max_pool2d(z, kernel_size= self.conv_pooling[i]) 
                
                z = self.batchnorm_second[i](z)

        z = self.drop1(z)

        z = z.view(z.size(0),-1)
        if(self.asymmetric):
            z = F.relu(self.fc1_second(z))
        else:
           z = F.relu(self.fc1(z)) 
        z = self.drop2(z)
        
        if self.concatenate:
            x_tot = torch.cat((x,z),1)
        else: 
            x_tot = x+z
        
        x_tot = (x_tot-x_tot.min())/x_tot.max()

        x_tot = self.fc2(x_tot)
       # x = self.softmax(x)
        return x_tot
    
    
    
    
    
class Siamese1DCNN(nn.Module):
    def __init__(self,n_classes,m=117,h1=64,conv_size=3,ending_size = 1,conv_pooling = [2,2,1,1,1],conv_kernels = [3,3,3], embedding_size = 64,dropout = 0, leng = 485, last_different = False, asymmetric = False, concatenate = False, channels = 1): #256, 128
        super(Siamese1DCNN, self).__init__()
        self.convs = nn.ModuleList()
        self.batchnorm = nn.ModuleList()
        self.conv_size = conv_size
        self.channels = channels
        self.conv_pooling = conv_pooling
        self.leng = leng
        self.ending_size = ending_size
        self.conv_kernels = conv_kernels
        self.leng = leng
        self.m=m
        self.asymmetric = asymmetric
        for k in range((conv_size)):
            if(k==0):
               self.convs.append(nn.Conv1d(channels, embedding_size, kernel_size = conv_kernels[k])) 
               self.add_module("conv"+str(k+1), self.convs[-1])
               self.batchnorm.append(nn.BatchNorm1d(embedding_size))

            elif(k==(conv_size-1) and last_different):
                self.convs.append(nn.Conv1d(embedding_size, ending_size, kernel_size = conv_kernels[k])) 
                self.add_module("conv"+str(k+1), self.convs[-1])
                self.batchnorm.append(nn.BatchNorm1d(ending_size))
            else:
                self.convs.append(nn.Conv1d(embedding_size, embedding_size, kernel_size = conv_kernels[k]))
                self.add_module("conv"+str(k+1), self.convs[-1])
                self.batchnorm.append(nn.BatchNorm1d(embedding_size))
                
            
        self.drop1=nn.Dropout(p=dropout) 
        if (last_different):
            self.fc1 = nn.Linear(ending_size*m, h1)       
        else:
            self.fc1 = nn.Linear(embedding_size*m, h1)
        self.drop2=nn.Dropout(p=dropout)
        self.convs_second = nn.ModuleList()
        self.batchnorm_second = nn.ModuleList()

        for k in range((conv_size)):
            if(k==0):
                if(asymmetric):
                    self.convs_second.append(nn.Conv1d(channels, embedding_size, kernel_size = conv_kernels[k])) 
                    self.add_module("conv_second"+str(k+1), self.convs[-1])
                    self.batchnorm_second.append(nn.BatchNorm1d(embedding_size))
            elif(k==(conv_size-1) and last_different):
                if(asymmetric):
                    self.convs_second.append(nn.Conv1d(embedding_size, ending_size, kernel_size = conv_kernels[k])) 
                    self.add_module("conv_second"+str(k+1), self.convs[-1])
                    self.batchnorm_second.append(nn.BatchNorm1d(ending_size))

            else:
                if(asymmetric):
                    self.convs_second.append(nn.Conv1d(embedding_size, embedding_size, kernel_size = conv_kernels[k]))
                    self.add_module("conv_second"+str(k+1), self.convs[-1])
                    self.batchnorm_second.append(nn.BatchNorm1d(embedding_size))

        
        if(asymmetric):
        
            if (last_different):
    
                self.fc1_second = nn.Linear(ending_size*m, h1)       
            else:
                self.fc1_second = nn.Linear(embedding_size*m, h1)

        
        self.concatenate = concatenate
        self.fc2 = nn.Linear(h1+ int(concatenate)*h1, n_classes)

    
    def forward(self, x_tot ):
        x_tot = x_tot.view(x_tot.shape[0],self.channels,self.leng)
        x=x_tot[:,0,:int(self.leng/2)]
        x= x.view(x.shape[0],1,x.shape[1])
        z= x_tot[:,0,int(self.leng/2):]
        z= z.view(z.shape[0],1,z.shape[1])
        for i in range(self.conv_size):
            x = F.relu(self.convs[i](x))
            
            x = F.max_pool1d(x, kernel_size= self.conv_pooling[i])
            
            x = self.batchnorm[i](x)


        x = self.drop1(x)

        x = x.view(x.size(0),-1)

        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        
        #z = z.view(z.shape[0],self.channels,self.leng)
        
        if(self.asymmetric):
            for i in range(self.conv_size):
                z = F.relu(self.convs_second[i](z))
                
                z = F.max_pool1d(z, kernel_size= self.conv_pooling[i]) 
                z = self.batchnorm_second[i](z)
        else:
            for i in range(self.conv_size):
                z = F.relu(self.convs[i](z))
                
                z = F.max_pool1d(z, kernel_size= self.conv_pooling[i]) 
                
                z = self.batchnorm[i](z)

        z = self.drop1(z)

        z = z.view(z.size(0),-1)
        if(self.asymmetric):
            z = F.relu(self.fc1_second(z))
        else:
           z = F.relu(self.fc1(z)) 
        z = self.drop2(z)
        
        if self.concatenate:
            output = torch.cat((x,z),1)
        else: 
            output = torch.abs(x-z)
        
        #x_tot = (x_tot-x_tot.min())/x_tot.max()

        output = self.fc2(output)
        if not self.concatenate:
            output = torch.sigmoid(output)
       # x = self.softmax(x)
        return output