import ax
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import pickle
from scipy import fftpack as f
#from scipy import signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import sklearn
from sklearn.model_selection import KFold
import random
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
import itertools
import os
import models
from models import Conv
from models import Transformer1d
from models import Conv2D
from support_functions import domain_converter
from support_functions import processer
import plotly
from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render
from ax.utils.tutorials.cnn_utils import train, evaluate
from support_functions import intra_class_oversampling
from support_functions import valid_epoch
from support_functions import data_extractor
import seaborn as sns
#number of points per signal (window)
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
pio.renderers.default='browser'
#finds the width of the spectral window 
def windowfinder(signal):
    return min(signal[:,0]),max(signal[:,0]) 
#data loading

folder =  "C:/Users/epick/OneDrive/Desktop/uni/JP/progetto/sugars_company" 
elements = os.listdir(folder)
data_map = []
'''
lr = 0.00005
dim = 64
decay = 0
dropout = 0
n_heads = 1
'''
#outer parameter initialization (some are overwritten by baesyan optimization)
embedding_size = 64
m=242
leng= 485
width= 40
#batch_size = 7
printing = False
epochs=100
lr= 0.00005 
decay =0
dropout=0
dim = 128
num_heads = 1
num_layers = 1#3
n=1024
conv_pooling = [2,1,1,1,1]
conv_kernels = [3,3,3,3,3]
ending_size = 1
num_epochs = 40
dataset_number = 0

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)
#augmentation function copy (also present in support_functions module)
def augmentation(n_classes,training_data_map,oversampling_number=10):
    divided_by_class = []
    for i in range(n_classes): 
        divided_by_class.append([])
    for i,element in enumerate(training_data_map):
        j = int(element[1])
        divided_by_class[j].append(element[0])
    augmented_training = []
    for i,classe in enumerate(divided_by_class):
        augmented_training.append(intra_class_oversampling(classe,len(classe),oversampling_number))
    augmented_training_final = []
    for i,element in enumerate(augmented_training):
        for j,spectrum in enumerate(element):
            augmented_training_final.append([spectrum,i])
    return augmented_training_final
#training function (validation is pulled from support_functions)
def train_epoch(model,device,dataloader,loss_fn,optimizer):
    train_loss,train_correct=0.0,0
    model.train()
    count=0
    for signals, labels in dataloader:
        signals,labels = signals.to(device),labels.to(device)
        optimizer.zero_grad()
        output = (model(signals))
        count+=1
        loss = loss_fn(output,labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * signals.size(0)
        scores, predictions = torch.max(output.data, 1)

        output=F.softmax(output,dim=0)
        train_correct +=  int(((output.argmax(1) == (labels))).sum()) 

        for param in model.parameters():
            a=param.grad

    return train_loss,train_correct
#outer wrapper for hyperparameter optimization
def train_and_evaluate_kfold( parameters):
    batch_size=parameters.get('batch_size')
    train = []
    test = []
    results=[]
    #inner testing acc

    
    result_total1 = []
    #inner split for nested cross-validation
    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(data_map)))):
        print('Fold {}'.format(fold + 1))
        result_test1 = []
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)
        if augment[0]:
            train_data = [data_map[idx] for idx in train_idx]
            training_set = augmentation(n_classes,train_data,augment[1])
            training_set_data,training_labels = data_extractor(training_set)
            training_set = TensorDataset(torch.tensor(training_set_data),torch.tensor(training_labels))
        train_loader = DataLoader(training_set, batch_size=batch_size, shuffle = False)
        test_loader = DataLoader(data_map, batch_size=batch_size, sampler=test_sampler)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        
        model = Conv( n_classes = n_classes,leng=leng,conv_size=parameters.get('conv_size'),conv_kernels=conv_kernels,conv_pooling=conv_pooling, m = m, embedding_size = embedding_size, dropout = dropout,last_different = True, ending_size = ending_size)
       
        #model = Transformer1d( n_classes, leng, leng, num_heads, dim, dropout, F.relu,num_layers, verbose=False)
      
        #model = Conv2D( n_classes = n_classes,conv_kernels=conv_kernels,width=width, m = m, embedding_size = 64, dropout = dropout,last_different = True, ending_size = 1)
        model=model.double()
        model.to(device)
        lr = parameters.get('lr')
        decay = parameters.get('decay')
        
        optimizer = optim.Adam(model.parameters(),lr=lr,weight_decay = decay) #0.01
    
        history = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[]}
    
        for epoch in range(num_epochs):
            train_loss, train_correct=train_epoch(model,device,train_loader,criterion,optimizer)
            test_loss, test_correct=valid_epoch(model,device,test_loader,criterion,similarity=False, split = False)
    
            train_loss = train_loss / len(train_loader.sampler)
            train_acc = train_correct / len(train_loader.sampler) * 100
            test_loss = test_loss / len(test_loader.sampler)
            test_acc = test_correct / len(test_loader.sampler) * 100
            result_test1.append(test_acc)
            if verbose:
                print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f} AVG Training Acc {:.2f} % AVG Test Acc {:.2f} %".format(epoch + 1,
                                                                                                                         num_epochs,
                                                                                                                         train_loss,
                                                                                                                         test_loss,
                                                                                                                         train_acc,
                                                                                                                         test_acc))
            
            
            
            
                
            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            train.append(train_acc)
            test.append(test_acc)
    
        if verbose:
           # foldperf['fold{}'.format(fold+1)] = history  
            plt.figure(fold+1)
            plt.plot(range((fold+1)*num_epochs),train)
            plt.plot(range((fold+1)*num_epochs),test)
            plt.xlabel("N. Epochs")
            plt.ylabel("Accuracy")
            plt.legend(['Training set accurcy', 'Testing set accuracy'],loc = 'lower left')
        result_total1.append(max(result_test1))
        
    #torch.save(model,'k_cross_tran.pt') 
    avg  = np.mean(result_total1)
    std = np.std(result_total1)
    print(avg)
    
    print(std)
    return(avg)
#sugars loading
base= np.zeros(5*n)

container=np.zeros(shape=(100,5*n))
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
            #signal1=signal
            #range_map=np.append(range_map,padder, axis= 0)
                signal1=np.append(signal,np.zeros(shape=(int(n-len(signal)))))
                container[25*i+k,1024*j:1024*(j+1)]=signal1

        #padder_map = padder=[max(range_map)+j*abs(range_map[0]- range_map[1]) for j in range(int(n-len(range_map)))]#(np.linspace(windowfinder(signal)[0],windowfinder(signal)[1],1024))[int(len(signal[:,0])):]
signal=pd.read_csv(r"C:/Users/epick/OneDrive/Desktop/uni/JP/progetto/SARS-CoV-2_omicron_variant/SARS-CoV-2_omicron_variant/{} 0{} 1000.txt".format('BA.1A',1),delimiter='\t',header=None).values
virus_domain = signal[:,0]  
from scipy import signal
for i in range(100):
   f=container[i,0:3000]
   f=(f-f.min())/f.max()
   data_map.append([f,int(i/25)])

   #data_map.append([signal.cwt(f, signal.ricker, np.arange(1, width)),int(i/25)])
base= np.sort(base)

data_map = []
for i in range(100):
    f = container[i, 0:3000]
    f = (f - f.min()) / f.max()
    data_map.append([f, int(i / 25)])


virus_list=['BA.1A','BA.1B','BA.2','BA.2.75','BA.4','BA.5','XE' ]


windows_list=['450','1100','1625','2900','3300']

data=[]
#virus loading
data_map_virus=[]
labels=[]
from scipy import signal

for i,virus in enumerate(virus_list):
    for j in range(9):
        signa=pd.read_csv(r"C:/Users/epick/OneDrive/Desktop/uni/JP/progetto/SARS-CoV-2_omicron_variant/SARS-CoV-2_omicron_variant/{} 0{} 1000.txt".format(virus,j+1),delimiter='\t',header=None).values
        data_map_virus.append([signa[:,1],torch.tensor(i)])
    signa=pd.read_csv(r"C:/Users/epick/OneDrive/Desktop/uni/JP/progetto/SARS-CoV-2_omicron_variant/SARS-CoV-2_omicron_variant/{} 10 1000.txt".format(virus),delimiter='\t',header=None).values
    data_map_virus.append([signa[:,1],torch.tensor(i)])
    base = signa[:,0]

candida_list = ['albicans','asahii','auris','duobushaemulonii','glabrata','haemulonii','iusitanae','krusei','Minuta','neoforams','pseudohaemulonii']
signal=pd.read_csv(r"C:/Users/epick/OneDrive/Desktop/uni/JP/progetto/SARS-CoV-2_omicron_variant/SARS-CoV-2_omicron_variant/{} 0{} 1000.txt".format('BA.1A',1),delimiter='\t',header=None).values
virus_base = signal[:,0]  
base = virus_base
#candida loading
folder = "C:/Users/epick/OneDrive/Desktop/uni/JP/progetto/Candida individual"
n_classes, n, base, data_map_candida = processer(folder, task = 'classification', base = base,  CWT= [False,width], positional_encoding = False )


label_names = []
label_names.append(sugars_list)
label_names.append(virus_list)
label_names.append(candida_list)

class_names = label_names[dataset_number]
k=5#5 
splits=KFold(n_splits=k,shuffle=True,random_state=42)
ratio = 1/k
result_total = []
conf_matrices_total = []

data_maps = [data_map,data_map_virus,data_map_candida]
data_map = data_maps[dataset_number]

n_classes = len(class_names)

if (dataset_number==0):
    leng = 3000
    m=1500
for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(data_map)))):
    result_test = []
    conf_matrices = []
    divided_by_class = []
    #divide by class
    for i in range(n_classes): 
        divided_by_class.append([])
    for i,element in enumerate(data_map):
        j = int(element[1])
        divided_by_class[j].append(element[0])
    data_map = []
    data_map_test = []
    #separate testing and training (data_map is train/val)
    for i,element in enumerate(divided_by_class):
        idxs=np.arange(len(element))
        #sample ratio of each class
        idxs = np.random.choice(idxs,int(ratio*len(element)))
        for j,spectrum in enumerate(element):
          
            #add to test those sampled
            if np.isin(j,idxs):
                data_map_test.append([spectrum,i])
            else:
                data_map.append([spectrum,i])
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(42)
    criterion = nn.CrossEntropyLoss()

    verbose = False
    augment = [True,10]
    
    best_parameters, values, experiment, model = optimize(
        parameters=[
            {"name": "lr", "type": "range", "bounds": [1e-6, 0.01], "log_scale": True},
            {"name": "batch_size", "type": "range", "bounds": [16, 128]},
            {"name": "conv_size", "type": "range", "bounds": [1, 5]},
            {"name": "decay", "type": "range", "bounds": [0.01,0.15], "log_scale": True},
            #{"name": "max_epoch", "type": "range", "bounds": [1, 30]},
            #{"name": "stepsize", "type": "range", "bounds": [20, 40]},        
        ],
      
        evaluation_function= train_and_evaluate_kfold,
        objective_name='accuracy',
    )
    print(best_parameters)
    best_objectives = np.array([[trial.objective_mean*100 for trial in experiment.trials.values()]])
    
    best_objective_plot = optimization_trace_single_method(
        y=np.maximum.accumulate(best_objectives, axis=1),
        title="Model performance vs. # of iterations",
        ylabel="Classification Accuracy, %",
    )
    
    #plotting hyperparameters search maps
    '''
    render(best_objective_plot)
    render(plot_contour(model=model, param_x='batch_size', param_y='lr', metric_name='accuracy'))
    render(plot_contour(model=model, param_x='batch_size', param_y='conv_size', metric_name='accuracy'))
    render(plot_contour(model=model, param_x='batch_size', param_y='decay', metric_name='accuracy'))
    render(plot_contour(model=model, param_x='decay', param_y='lr', metric_name='accuracy'))
    render(plot_contour(model=model, param_x='decay', param_y='conv_size', metric_name='accuracy'))
    render(plot_contour(model=model, param_x='conv_size', param_y='lr', metric_name='accuracy'))
    '''
    num_epochs = 40
    
    print(np.shape(data_map_test))
    parameters = best_parameters
    batch_size=parameters.get('batch_size')
    #treating train
    train_data = data_map
    training_set = augmentation(n_classes,train_data,augment[1])
    training_set_data,training_labels = data_extractor(training_set)
    training_set = TensorDataset(torch.tensor(training_set_data),torch.tensor(training_labels))
    #treating test
    test_set = data_map_test
    test_set_data,test_labels = data_extractor(test_set)
    test_set = TensorDataset(torch.tensor(test_set_data),torch.tensor(test_labels))
    
    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle = True)
    test_loader = DataLoader(data_map_test, batch_size=batch_size,shuffle = True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = Conv( n_classes = n_classes,leng=leng,conv_size=parameters.get('conv_size'),conv_kernels=conv_kernels,conv_pooling=conv_pooling, m = m, embedding_size = embedding_size, dropout = dropout,last_different = True, ending_size = ending_size)
    
    #model = Transformer1d( n_classes, leng, leng, num_heads, dim, dropout, F.relu,num_layers, verbose=False)
    
    #model = Conv2D( n_classes = n_classes,conv_kernels=conv_kernels,width=width, m = m, embedding_size = 64, dropout = dropout,last_different = True, ending_size = 1)
    model=model.double()
    model.to(device)
    lr = parameters.get('lr')
    decay = parameters.get('decay')
    
    optimizer = optim.Adam(model.parameters(),lr=lr,weight_decay = decay) 
    
    history = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[]}
    
    for epoch in range(num_epochs):
        train_loss, train_correct=train_epoch(model,device,train_loader,criterion,optimizer)
        test_loss, test_correct=valid_epoch(model,device,test_loader,criterion,similarity=False, split = False,confusion_matrices = [True,conf_matrices])
    
        train_loss = train_loss / len(train_loader.sampler)
        train_acc = train_correct / len(train_loader.sampler) * 100
        test_loss = test_loss / len(test_loader.sampler)
        test_acc = test_correct / len(test_loader.sampler) * 100
        result_test.append(test_acc)

        print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f} AVG Training Acc {:.2f} % AVG Test Acc {:.2f} %".format(epoch + 1,
                                                                                                                     num_epochs,
                                                                                                                     train_loss,
                                                                                                                     test_loss,
                                                                                                                     train_acc,
                                                                                                                     test_acc))
    result_total.append(max(result_test))
    sklearn.metrics.ConfusionMatrixDisplay(conf_matrices[np.argmax(result_test)])
    conf_matrices_total.append(conf_matrices[np.argmax(result_test)]) 
    

    print(class_names)
    cm = sum(conf_matrices_total)
    # Plot confusion matrix in a beautiful manner
    fig = plt.figure(figsize=(16, 14))
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, fmt = 'g'); #annot=True to annotate cells
    # labels, title and ticks
    ax.set_xlabel('Predicted', fontsize=20)
    ax.xaxis.set_label_position('bottom')
    plt.xticks(rotation=90)
    ax.xaxis.set_ticklabels(class_names, fontsize = 10)
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True', fontsize=20)
    ax.yaxis.set_ticklabels(class_names, fontsize = 10)
    plt.yticks(rotation=0)

    plt.title('Refined Confusion Matrix', fontsize=20)