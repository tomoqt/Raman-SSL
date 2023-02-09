
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import pickle
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
import os
from scipy import signal
import math
from sklearn.preprocessing import MinMaxScaler
import sklearn
from os import listdir
signa=pd.read_csv(r"C:/Users/epick/OneDrive/Desktop/uni/JP/progetto/SARS-CoV-2_omicron_variant/SARS-CoV-2_omicron_variant/{} 0{} 1000.txt".format('BA.1A',1),delimiter='\t',header=None).values
virus_domain = signa[:,0]
#converts domain via spline interpolation
def domain_converter(base,spectrum,output_domain, normalize = True):
    signa = scipy.interpolate.CubicSpline(np.sort(base,axis=0),spectrum)(output_domain)
    return (signa- np.min(signa))/np.max(signa)
'''
function built for general pre-processing of a specific dataset of viruses, in a list-based shape
'''
def Virus_processer (data_folder, task = 'classification', base = np.array([False,False]), CWT= [False,20], positional_encoding = False):
    folders = os.listdir(data_folder)
    if task == 'classification':
        n_classes = len(folders)
        data_map = []
        n = (pd.read_csv(data_folder + '/' + folders[0] + '/' + os.listdir(data_folder + '/' + folders[0])[0]  ,delimiter='\t',header=None).values[:,2:4]).shape[0]
        if  not base.all() :
            base =  pd.read_csv(data_folder + '/' + folders[0]+ '/' + os.listdir(data_folder + '/' + folders[0])[0] ,delimiter='\t',header=None).values[:,2:4][:,0]
        for i,folder in enumerate(folders):
            spectra=os.listdir(data_folder + '/' + folder)

            for j,spectrum in enumerate(spectra):
                signa = pd.read_csv(data_folder + '/' + folder + '/' + spectrum,delimiter='\t',header=None).values[:,2:4]
                signal_new = np.zeros(shape=(len(base),2))
                temp_0,idx = np.unique(signa[:,0],return_index=True)
                signal_new[:,0] = (base-np.min(base))/np.max(base)
                signal_new[:,1] = domain_converter(temp_0,signa[idx,1],base)
                signal_new[:,1] = (signal_new[:,1]-np.min(signal_new[:,1]))/np.max(signal_new[:,1])
                if(CWT[0]):
                        data_map.append([signal.cwt(signal_new[:,1], signal.ricker, np.arange(1, CWT[1])),torch.tensor(i)])

                else:
                    if (not positional_encoding):
                        data_map.append([(signal_new[:,1]),torch.tensor(i)])
                    else:
                        data_map.append([(signal_new),torch.tensor(i)])

        return n_classes, n, base, data_map
    elif task == 'reconstruction':    

        data_map = []
        n = (pd.read_csv(data_folder + '/' + folders[0] + '/' + os.listdir(data_folder + '/' + folders[0])[0] ,delimiter='\t',header=None).values[:,2:4]).shape[0]
        n_classes=n
        for i,folder in enumerate(folders):
            spectra=os.listdir(data_folder + '/' + folder) 
            for j,spectrum in enumerate(spectra):
                signa = pd.read_csv(data_folder + '/' + folder + '/' + spectrum,delimiter='\t',header=None).values[:,2:4]
                signal_new = np.zeros(shape=(len(base),2))
                temp_0,idx = np.unique(signa[:,0],return_index=True)
                signal_new[:,0] = (base-np.min(base))/np.max(base)
                signal_new[:,1] = domain_converter(temp_0,signa[idx,1],base)
                signal_new[:,1] = (signal_new[:,1]-np.min(signal_new[:,1]))/np.max(signal_new[:,1])
                if(CWT[0]):
                    data_map.append([signal.cwt(signal_new[:,1], signal.ricker, np.arange(1, CWT[1])),torch.tensor(signal.cwt(signal_new[:,1], signal.ricker, CWT[1]))])
                else:
                    if (not positional_encoding):
                        data_map.append([(signal_new[:,1]),(signal_new[:,1])])
                    else:
                        data_map.append([(signal_new),(signal_new[:,1])])

    elif task == 'divide by class':    
        data_class = []
        data_map = []
        n = (pd.read_csv(data_folder + '/' + folders[0] + '/' + os.listdir(data_folder + '/' + folders[0])[0] ,delimiter='\t',header=None).values[:,2:4]).shape[0]
        n_classes = len(folders)
        for i,folder in enumerate(folders):
            spectra=os.listdir(data_folder + '/' + folder)
            for j,spectrum in enumerate(spectra):
                signa = pd.read_csv(data_folder + '/' + folder + '/' + spectrum,delimiter='\t',header=None).values[:,2:4]
                signal_new = np.zeros(shape=(len(base),2))
                temp_0,idx = np.unique(signa[:,0],return_index=True)
                signal_new[:,0] = (base-np.min(base))/np.max(base)
                signal_new[:,1] = domain_converter(temp_0,signa[idx,1],base)
                signal_new[:,1] = (signal_new[:,1]-np.min(signal_new[:,1]))/np.max(signal_new[:,1])
                if(CWT[0]):
                    data_class.append(signal.cwt(signal_new[:,1], signal.ricker, np.arange(1, CWT[1])))
                else:
                    if (not positional_encoding):
                        data_class.append((signal_new[:,1]))  
                    else:
                        data_class.append((signal_new))  
            data_map.append(data_class)
            data_class = []


        
    
        return n_classes, n, base, data_map
    raise Exception("Warning: task'" + task + "'not recognized")
    return 0

'''
perturbator function: spits out perturbed elements couple with originals. 
3 perturbations are available, perturbation type indicates:
    gaussian, baseline, shift and the number of repetitions of the dataset to produce of such perturbations
'''
def patch_masking(data, n_pert, patch_width):
    if patch_width % 2 !=0 :
        patch_width = patch_width +1
        
    k = len(data)
    l = len(data[0])

    patch_data = np.zeros(shape=(n_pert*k,l))
    patch_labels= np.zeros(shape=(n_pert*k,patch_width))
    half_width= int(patch_width/2)
    maximum = int(l-half_width)
    minimum = int(half_width)
    for i in range(n_pert*k):
        center_idz = int(np.random.randint(minimum, maximum,1))
        patch_data[i] = data[i%k]
        patch_labels[i] = data[i%k][center_idz-half_width:center_idz+half_width]
        temp = np.zeros(len( patch_data[i]))
        temp[center_idz-half_width:center_idz+half_width] = patch_data[i][center_idz-half_width:center_idz+half_width]
        patch_data[i] = patch_data[i] - temp
        #patch_data[i][center_idz-half_width:center_idz+half_width] = np.zeros(patch_width)
    return patch_data,patch_labels
def gaussian_perturbation(data, gaus_mean, gaus_std, n_pert):#data deve entrare come np array con SOLO i segnali
    k=len(data)
    perturbations = np.random.multivariate_normal(gaus_mean,gaus_std,size=n_pert*k)
    
    return  perturbations

def baseline_perturbation(data, n_pert, baseline_distribution): 
    k = len(data)
    l = len(data[0])
    baseline_pert = np.zeros(shape=(n_pert*k,l))
    for j,signal in enumerate(baseline_pert):
        baseline_pert[j] = data[j%k] - abs(baseline_distribution(1)*np.ones(l))#if you want to shift upwards you have toa ctually subtract due to renormalization
        baseline_pert[j] = baseline_pert[j]/max(baseline_pert[j])
    return baseline_pert

def shift_perturbation(data, n_pert, shift_distribution ):
    k = len(data)
    l = len(data[0])
    shift_pert = np.zeros(shape=(n_pert*k,l))
    for j,signal in enumerate(shift_pert):
        shift_pert[j] = np.roll(data[j%k],int(l*shift_distribution(1)*0.5))
    return shift_pert

def perturbation_producer (data_map, gaus_mean, gaus_std, baseline_distribution = np.random.random, shift_distribution = np.random.random, perturbation_type = [1,1,1]): 
    n=len(data_map)

    container = np.zeros(shape=(int(np.dot(n*np.ones(len(perturbation_type)),perturbation_type)),len(data_map[0][0])))
    data=[]
    labels=[]
    for j,element in enumerate(data_map):
        data.append(element[0])
        labels.append(element[1])
    if perturbation_type[0]:
        gaus = gaussian_perturbation(data, gaus_mean, gaus_std, perturbation_type[0])
        container[:1*perturbation_type[0]*n] = np.repeat(data,perturbation_type[0],axis=0) + gaus
    if perturbation_type[1]:
        baseline = baseline_perturbation(data, perturbation_type[1],baseline_distribution)
        container[1*perturbation_type[0]*n:2*perturbation_type[1]*n] = baseline
    if perturbation_type[2]:
        shift = shift_perturbation(data, perturbation_type[2],shift_distribution)       
        container[2*perturbation_type[1]*n:3*perturbation_type[2]*n] = shift
  
    
    perturbed_data_map = []
    for j,element in enumerate(container):
        perturbed_data_map.append([element,labels[j%n]])
        

    return perturbed_data_map

def patch_producer(data_map,number, patch_width = 20):
    data,labels = data_extractor(data_map)
    patch_data,patch_labels = patch_masking(data,number,patch_width)
    patch_data_map = []
    for j,element in enumerate(patch_data):
        patch_data_map.append([element,patch_labels[j]])
    return patch_data_map

def data_extractor(data_map):    
    data=[]
    labels=[]
    for j,element in enumerate(data_map):
        data.append(element[0])
        labels.append(element[1])
    return data,labels
'''
function constructing combinatorily pairs of spectra for siamese contrastive learning
'''
def contrastive_pairs(classes, number_of_elements_per_class, number_of_pairs, n_negatives_per_positive,similarity=False, inference = False, oversampling = [True,2], separate = False):
    n_classes = len(classes)


    len_smallest_class = min([len(classe) for classe in classes])
    if len_smallest_class < 2 and inference == False:
        raise Exception("Class dimension has to be of at least 2")

    samples = []
    #sample a number number_of_elements_per_class from each class and append it as a list in samples, 
    #each element is a list containing samples for a class
    for i,classe in enumerate(classes):
        idxes = np.arange(0,number_of_elements_per_class)
        if (oversampling[0]):
            classe = intra_class_oversampling(classe, oversampling[1], number_of_elements_per_class-len(classe))
            idxes = np.random.choice(idxes,number_of_elements_per_class,replace=False)
        else:
            idxes = np.random.choice(idxes,min(number_of_elements_per_class,len(classe)),replace=False)
        samples.append([classe[idx] for idx in idxes])
    positives = []
    negatives = []

    #we make all possible positive couples
    for i,sample in enumerate(samples):
        if(similarity):
            positives = positives + [[a, b, torch.tensor(1).double()] for a in samples[i] for b in samples[i]]

        else:
            positives = positives + [[a, b, 1] for a in samples[i] for b in samples[i]]
        if (not separate):
            for l in range(n_negatives_per_positive):
                for j in range(len(sample)**2):
                    ranges = np.arange(0,n_classes)
                    ranges = np.delete(ranges,np.where(ranges == i))
                    k = np.random.choice(ranges,1)[0]
                    idx_k = int(np.random.randint(0,len(classes[k]))) 
                    negatives.append([samples[i][j%len(sample)],classes[k][idx_k],0])
    if separate:
        counter = 0
        # for the number of required elements
        while(counter <= n_negatives_per_positive*(number_of_pairs - len(positives))):
            #we cycle over classes
            for i in range(n_classes): 
                #and sample index k, representing a random class to get an element from
                k = 0
                #retake k if i==k, since must be negative
                while(k==i):
                    k = int(np.random.randint(0,n_classes-1))  
                #sample one element from position 1, in class i
                if(len(samples[i])!=1):
                    idx_sample = int(np.random.randint(0,len(samples[i])-1))
                else:
                    idx_sample = 0
                #sample one element from class k for position 2
                if(len(classes[k])!=1):
                    idx_k = int(np.random.randint(0,len(classes[k])-1)) 
                else:
                    idx_k = 0
                if(similarity):
                    negatives.append([samples[i][idx_sample],classes[k][idx_k],torch.tensor(0).double()])
                else:
                    negatives.append([samples[i][idx_sample],classes[k][idx_k],torch.tensor(0)])
                counter +=1
            
    '''  
    if n_negatives_per_positive ==1:
        return positives + negatives[:len(positives)],len(positives),len(negatives[:len(positives)]),positives, negatives
    else:
    '''
    return positives + negatives,len(positives),len(negatives), positives, negatives

#to be implemented, if redundancy=true throw exception regarding dimension of class
# with respect to required number of elements.
def intra_class_oversampling(classe,number_of_elements_per_sum, number_of_elements,redundancy = True, renormalize=True):   
    container = classe
    # for the number of elements, sample a number_of_elements_per_sum-tuple of elements in the class and take a 
    #linear combination of spectra in the sample, with uniformally distributed weights.
    for i in range(number_of_elements): 
        sum_elements = []
        idxes = np.arange(0,len(classe))
        idxes = np.random.choice(idxes,min(len(classe),number_of_elements_per_sum),replace=False)
        sum_elements = [classe[idx] for idx in idxes]
        weights = np.random.random(len(sum_elements))
        new_signal = np.zeros(shape = np.shape(classe[0])) 
        for j in range(len(sum_elements)):
            new_signal = new_signal + weights[j]*sum_elements[j]
        if renormalize:
            container.append((new_signal-np.min(new_signal))/(np.max(new_signal)))
        else:
            container.append(new_signal)
    return container
'''
general txt spectra processer, in a list-based shape
'''
def processer (data_folder, task = 'classification', base = np.array([False,False]),  CWT= [False,20], positional_encoding = False,delimiter = '\t' ):
    folders = os.listdir(data_folder)
    if task == 'classification':
        n_classes = len(folders)
        data_map = []
        n = (pd.read_csv(data_folder + '/' + folders[0] + '/' + os.listdir(data_folder + '/' + folders[0])[0]  ,delimiter=delimiter,header=None).values).shape[0]
        if  not base.all() :
            base =  pd.read_csv(data_folder + '/' + folders[0]+ '/' + os.listdir(data_folder + '/' + folders[0])[0] ,delimiter=delimiter,header=None).values[:,0]
        for i,folder in enumerate(folders):
            spectra=os.listdir(data_folder + '/' + folder)

            for j,spectrum in enumerate(spectra):
                signa = pd.read_csv(data_folder + '/' + folder + '/' + spectrum,delimiter=delimiter,header=None).values
                signal_new = np.zeros(shape=(len(base),2))
                temp_0,idx = np.unique(signa[:,0],return_index=True)
                signal_new[:,0] = (base-np.min(base))/np.max(base)
                signal_new[:,1] = domain_converter(temp_0,signa[idx,1],base)
                signal_new[:,1] = (signal_new[:,1]-np.min(signal_new[:,1]))/np.max(signal_new[:,1])
                if(CWT[0]):
                    data_map.append([signal.cwt(signal_new[:,1], signal.ricker, np.arange(1, CWT[1])),torch.tensor(i)])
                else:
                    if (not positional_encoding):
                       data_map.append([(signal_new[:,1]),torch.tensor(i)])
                    else:
                        data_map.append([(signal_new),torch.tensor(i)])

        return n_classes, n, base, data_map
    elif task == 'reconstruction':    
        
        data_map = []
        n = (pd.read_csv(data_folder + '/' + folders[0] + '/' + os.listdir(data_folder + '/' + folders[0])[0] ,delimiter=delimiter,header=None).values).shape[0]
        n_classes=n
        for i,folder in enumerate(folders):
            spectra=os.listdir(data_folder + '/' + folder) 
            for j,spectrum in enumerate(spectra):
                signa = pd.read_csv(data_folder + '/' + folder + '/' + spectrum,delimiter=delimiter,header=None).values
                signal_new = np.zeros(shape=(len(base),2))
                temp_0,idx = np.unique(signa[:,0],return_index=True)
                signal_new[:,0] = (base-np.min(base))/np.max(base)
                signal_new[:,1] = domain_converter(temp_0,signa[idx,1],base)
                signal_new[:,1] = (signal_new[:,1]-np.min(signal_new[:,1]))/np.max(signal_new[:,1])
                if(CWT[0]):
                    data_map.append([signal.cwt(signal_new[:,1], signal.ricker, np.arange(1, CWT[1])),torch.tensor(signal.cwt(signal_new[:,1], signal.ricker, CWT[1]))])
                else:
                    if (not positional_encoding):
                        data_map.append([(signal_new[:,1]),(signal_new[:,1])])
                    else:
                        data_map.append([(signal_new),(signal_new[:,1])])

    elif task == 'divide by class':    
        data_class = []
        data_map = []
        n = (pd.read_csv(data_folder + '/' + folders[0] + '/' + os.listdir(data_folder + '/' + folders[0])[0] ,delimiter=delimiter,header=None).values).shape[0]
        n_classes=n
        for i,folder in enumerate(folders):
            spectra=os.listdir(data_folder + '/' + folder)

            for j,spectrum in enumerate(spectra):
                signa = pd.read_csv(data_folder + '/' + folder + '/' + spectrum,delimiter=delimiter,header=None).values
                signal_new = np.zeros(shape=(len(base),2))
                temp_0,idx = np.unique(signa[:,0],return_index=True)
                signal_new[:,0] = (base-np.min(base))/np.max(base)
                signal_new[:,1] = domain_converter(temp_0,signa[idx,1],base)
                signal_new[:,1] = (signal_new[:,1]-np.min(signal_new[:,1]))/np.max(signal_new[:,1])
                if(CWT[0]):
                    data_class.append(signal.cwt(signal_new[:,1], signal.ricker, np.arange(1, CWT[1])))
                else:
                    if (not positional_encoding):
                        data_class.append((signal_new[:,1]))    
                    else:
                        data_class.append((signal_new))    
            data_map.append(data_class)
            data_class = []                        
        

        return n_classes, n, base, data_map

    raise Exception("Warning: task'" + task + "'not recognized")
    return 0
'''
obsolete siamese zero-shot testing function
'''
def test_siamese(model,device,test_data,database, similarity = False):
        model.eval()
        with torch.no_grad():

            test_data,test_labels = data_extractor(test_data)
            database,labels = data_extractor(database)
            count_correct = 0
            test_data_loader = DataLoader(((test_data)),batch_size = 1,shuffle = False)
            database_loader = DataLoader(((database)),batch_size = 1,shuffle = False)
            for i,signal1 in enumerate(test_data_loader):
                temp = []
                signal1 = (signal1).to(device)
        
                for j,signal2 in enumerate(database_loader):
                    signal2 = (signal2).to(device)
        
                    output = model(signal1,signal2)
                    
                    if (not similarity):
                        output=F.softmax(output, dim = 1)
                        #print(output)

                        temp.append(output.cpu().detach()[0][1])
                    
                    else:
                        temp.append(output.cpu().detach().numpy()[0])#.cpu().detach().numpy()
                    

                idx=np.argmax((temp),axis = 0)
                print(temp,idx)
                #print(test_labels[i],int(idx))


                if (test_labels[i] == labels[int(idx)]):
                    count_correct += 1
    
        return count_correct/len(test_data_loader.sampler)
    
'''
working zero-shot testing function for siamese networks 
'''
def new_test_siamese(model1,device,test_data,database,batch=1, similarity = False,verbose =[False,False],tensor = True):
        model1.eval()
        with torch.no_grad():
            if not similarity:
                criterion = nn.CrossEntropyLoss()
            else:
                criterion = nn.MSELoss()

            base = np.linspace(0,1,970)
            test_data,test_labels = data_extractor(test_data)
            database,labels = data_extractor(database)
            count_correct = 0
            tester = []
           # print((database[0:2]))
            if not tensor:
                for i, element in enumerate(test_data):
                    for j, datum in enumerate(database):
                        if (test_labels[i]==labels[j]):
                            tester.append([element, datum,1])
                        else:
                            tester.append([element, datum,0])
            else:
                labelline = []
                datini = []
                for i, element in enumerate(test_data):
                    for j, classe in enumerate(database):
                        for t, datum in enumerate(database[j]):
                            if (test_labels[i].item()==labels[j].item()):
                                #print(np.shape(database[j]))
                                datini.append(np.concatenate((element, datum),axis=None))
                                labelline.append(1)
    
                            else:
                                datini.append(np.concatenate((element, datum),axis=None))
                                labelline.append(0)
                            
                tester  = TensorDataset(torch.tensor(datini),torch.tensor(labelline))




            tester_loader = DataLoader(tester,batch_size = batch, shuffle = False)
            #print('ooooooo1',valid_epoch(model,device,pairs,criterion,verbose = verbose[0],tensor=tensor,similarity=similarity)[1]/len(pairs.sampler))
            res1=valid_epoch(model1,device,tester_loader,criterion,verbose = verbose[0],tensor=tensor,similarity=similarity)[1]/len(tester_loader.sampler)
            #print('vediamo',all([(model.state_dict()[key] == state_dict[key]).all() for key in model.state_dict().keys()]))

            total=[]
            if tensor:

                for signal,label in tester_loader:
                    signal = (signal).to(device)
                    label = label.to(device)
                    signal = signal.view(signal.shape[0],1,signal.shape[1])
                    model1.eval()

                    label=label.view(label.shape[0],1)
                    output = model1(signal)
                    #output = torch.norm(signal[:,:int(signal.shape[1]/2)] - signal[:,int(signal.shape[1]/2):], dim=1) per test con dist quadratica
                    if not similarity:
                        output = F.softmax(output, dim = 1)
                    if(verbose[1]):
                        for i,outpu in enumerate(output):
                           print('output and label: ',outpu,label[i])
                    for i in range(output.shape[0]):
                        if not similarity:
                            total.append(output.cpu().detach().numpy()[i][1])
                        else:
                            total.append(output[i][0])
            else:
                for signal1,signal2,label in tester_loader:
                    signal1 = (signal1).to(device)
                    signal2 = (signal2).to(device)

                    label = label.to(device)
                    label=label.view(label.shape[0],1)

                    output = model1(signal1,signal2)
                    if not similarity:
                        output = F.softmax(output, dim = 1)

                    if(verbose[1]):
                        for i,outpu in enumerate(output):
                           print('output and label: ',outpu,label[i])
                    for i in range(output.shape[0]):
                        if not similarity:
                            total.append(output.cpu().detach().numpy()[i][1])
                        else:
                            total.append(output.cpu().detach().numpy()[i][0])

            for k in range(len(test_data)):
                temp = []
                summone = 0
                for m in range(len(database)):
                    summone+=len(database[m])
                    
                for l in range(len(database)):
                    average  = 0
                    summino = 0
                    for m in range(l):
                        summino+=len(database[m])
                        
                    for s in range(len(database[l])):


                        output1=total[summino+s + k*summone ]
                        average += output1
                    temp.append(average/len(database[l]))
                idx=torch.tensor(temp).argmax()
                #idx=torch.tensor(temp).argmin() per test con dist quadratica
                if (test_labels[k].item() == labels[(idx)].item()):
                    count_correct += 1             
                    
        print('tester:',res1)
        del tester
        del tester_loader
        return count_correct/(len(test_data))
'''              
general purpose validation function     
'''                                        
def valid_epoch(model,device,dataloader,loss_fn,verbose = False,tensor = True,similarity = False,split = True,confusion_matrices=[False,[]]):
    valid_loss, val_correct = 0.0, 0
    counter_wrong_negatives = 0
    counter_wrong_positives = 0
    outputs=[]
    labels_tot=[]
    model.eval()
    with torch.no_grad():
        if tensor:
            for signal,label in dataloader:
                signal,label = signal.to(device),label.to(device)
                output = model(signal)
                if similarity:
                    label=label.view(label.shape[0],1)
                outputs = outputs + [out for out in output.argmax(1).cpu().detach().numpy()]
                labels_tot = labels_tot + [lab for lab in label.cpu().detach().numpy()]
                loss=loss_fn(output,label)
                if not similarity:
                    output=F.softmax(output,dim=1)
        
                   # print(label)
                    valid_loss+=loss.item()*signal.size(0)
                    scores, predictions = torch.max(output.data,1)
                   
                    val_correct += int(((output.argmax(1) == (label))).sum(dim=0)) 
                else:
                    output= output.cpu().detach().numpy()
                    if verbose:
                        for j in range(len(output)):
                            print('output:',output[j],label[j])
                    output[output>0.5]=1
                    output[output<=0.5]=0
                    valid_loss+=loss.item()*signal.size(0)

                    result =[]
                    for i in range(len(output)):
                        if split:
                            if output[i].item()!=label[i] and label[i] == 0:
                                counter_wrong_negatives += 1
                            if output[i].item()!=label[i] and label[i] == 1:   
                                counter_wrong_positives += 1
                        result.append(int(output[i].item()==label[i]))
                    val_correct += np.sum(result)
            if   split:
                print('wrong negatives: ',counter_wrong_negatives)
                print('wrong positives: ',counter_wrong_positives)
                if counter_wrong_positives!= 0:
                    print('ratio n/p:',counter_wrong_negatives/counter_wrong_positives )
            if confusion_matrices[0]:
                confusion_matrices[1].append(confusion_matrix(outputs,labels_tot))

            return valid_loss,val_correct

        else:
            
            for signal1, signal2,label in dataloader:
                signal1, signal2,label = signal1.to(device),signal2.to(device),label.to(device)
                output = (model(signal1,signal2))
                label=label.view(label.shape[0],1)

                loss=loss_fn(output,label)
                #label =  label.double().cpu().detach().numpy()

               # print(label)
                valid_loss+=loss.item()*signal1.size(0)
                output= output.cpu().detach().numpy()
                if verbose:
                    for outpu in range(len(output)):
                        print(outpu)
                output[output>0.5]=1
                output[output<=0.5]=0

                result =[]

                for i in range(len(output)):
                    if   split:
                        if output[i]!=label[i] and label[i]== 0:
                            counter_wrong_negatives += 1
                        if output[i]!=label[i] and label[i]== 1:   
                            counter_wrong_positives += 1

                    result.append(int(output[i]==label[i]))
                val_correct += np.sum(result)#((np.array(output == label).astype(int)) )
            if   split:
                print('wrong negatives: ',counter_wrong_negatives)
                print('wrong positives: ',counter_wrong_positives)
            return valid_loss,val_correct
            
'''
 txt spectra processer for  viruses ,TensorDataset-shaped initialization, analogous to virus_processor
'''           
def tensor_processer (data_folder, task = 'classification', base = np.array([False,False]), CWT= [False,20], positional_encoding = False):
    folders = os.listdir(data_folder)
    if task == 'classification':
        n_classes = len(folders)
        data = []
        labels = []
        n = (pd.read_csv(data_folder + '/' + folders[0] + '/' + os.listdir(data_folder + '/' + folders[0])[0]  ,delimiter='\t',header=None).values[:,2:4]).shape[0]
        if  not base.all() :
            base =  pd.read_csv(data_folder + '/' + folders[0]+ '/' + os.listdir(data_folder + '/' + folders[0])[0] ,delimiter='\t',header=None).values[:,2:4][:,0]
        for i,folder in enumerate(folders):
            spectra=os.listdir(data_folder + '/' + folder)

            for j,spectrum in enumerate(spectra):
                signa = pd.read_csv(data_folder + '/' + folder + '/' + spectrum,delimiter='\t',header=None).values[:,2:4]
                signal_new = np.zeros(shape=(len(base),2))
                temp_0,idx = np.unique(signa[:,0],return_index=True)
                signal_new[:,0] = (base-np.min(base))/np.max(base)
                signal_new[:,1] = domain_converter(temp_0,signa[idx,1],base)
                signal_new[:,1] = (signal_new[:,1]-np.min(signal_new[:,1]))/np.max(signal_new[:,1])
                if(CWT[0]):
                        data.append(signal.cwt(signal_new[:,1], signal.ricker, np.arange(1, CWT[1])))
                        labels.append(torch.tensor(i))
                else:
                    if (not positional_encoding):
                        data.append(signal_new[:,1])
                        labels.append(torch.tensor(i))
                    else:
                        data.append(signal_new)
                        labels.append(torch.tensor(i))

        return n_classes, n, base, TensorDataset(torch.Tensor(data),torch.Tensor(labels))
    elif task == 'reconstruction':    

        data = []
        labels = []
        n = (pd.read_csv(data_folder + '/' + folders[0] + '/' + os.listdir(data_folder + '/' + folders[0])[0] ,delimiter='\t',header=None).values[:,2:4]).shape[0]
        n_classes=n
        for i,folder in enumerate(folders):
            spectra=os.listdir(data_folder + '/' + folder) 
            for j,spectrum in enumerate(spectra):
                signa = pd.read_csv(data_folder + '/' + folder + '/' + spectrum,delimiter='\t',header=None).values[:,2:4]
                signal_new = np.zeros(shape=(len(base),2))
                temp_0,idx = np.unique(signa[:,0],return_index=True)
                signal_new[:,0] = (base-np.min(base))/np.max(base)
                signal_new[:,1] = domain_converter(temp_0,signa[idx,1],base)
                signal_new[:,1] = (signal_new[:,1]-np.min(signal_new[:,1]))/np.max(signal_new[:,1])
                if(CWT[0]):
                    data.append(signal.cwt(signal_new[:,1], signal.ricker, np.arange(1, CWT[1])))
                    labels.append(torch.tensor(signal.cwt(signal_new[:,1], signal.ricker, CWT[1])))
                else:
                    if (not positional_encoding):
                        data.append(signal_new[:,1])
                        labels.append(signal_new[:,1])
                    else:
                        data.append(signal_new)
                        labels.append(signal_new[:,1])
        return n_classes, n, base, TensorDataset(torch.Tensor(data),torch.Tensor(labels))

    elif task == 'divide by class':    
        data_class = []
        data_map = []
        n = (pd.read_csv(data_folder + '/' + folders[0] + '/' + os.listdir(data_folder + '/' + folders[0])[0] ,delimiter='\t',header=None).values[:,2:4]).shape[0]
        n_classes = len(folders)
        for i,folder in enumerate(folders):
            spectra=os.listdir(data_folder + '/' + folder)
            for j,spectrum in enumerate(spectra):
                signa = pd.read_csv(data_folder + '/' + folder + '/' + spectrum,delimiter='\t',header=None).values[:,2:4]
                signal_new = np.zeros(shape=(len(base),2))
                temp_0,idx = np.unique(signa[:,0],return_index=True)
                signal_new[:,0] = (base-np.min(base))/np.max(base)
                signal_new[:,1] = domain_converter(temp_0,signa[idx,1],base)
                signal_new[:,1] = (signal_new[:,1]-np.min(signal_new[:,1]))/np.max(signal_new[:,1])
                if(CWT[0]):
                    data_class.append(signal.cwt(signal_new[:,1], signal.ricker, np.arange(1, CWT[1])))
                else:
                    if (not positional_encoding):
                        data_class.append(signal_new[:,1])  
                    else:
                        data_class.append(signal_new)  
            data_map.append(data_class)
            data_class = []


        
    
        return n_classes, n, base, data_map
    raise Exception("Warning: task'" + task + "'not recognized")
    return 0
    ''' 
    analogous to contrastive pairs, but with TensorDataset-like instantiation in mind
    '''
def tensor_contrastive_pairs(classes, number_of_elements_per_class, number_of_pairs, n_negatives_per_positive,similarity=False, inference = False, oversampling = [True,2], separate = False):
    n_classes = len(classes)
    data=[]
    labels=[]

    len_smallest_class = min([len(classe) for classe in classes])
    if len_smallest_class < 2 and inference == False:
        raise Exception("Class dimension has to be of at least 2")

    samples = []
    #sample a number number_of_elements_per_class from each class and append it as a list in samples, 
    #each element is a list containing samples for a class
    for i,classe in enumerate(classes):
        idxes = np.arange(0,number_of_elements_per_class)
        if (oversampling[0]):
            classe = intra_class_oversampling(classe, oversampling[1], number_of_elements_per_class-len(classe))
            idxes = np.random.choice(idxes,number_of_elements_per_class,replace=True)
        else:
            idxes = np.random.choice(idxes,min(number_of_elements_per_class,len(classe)),replace=False)
        samples.append([classe[idx] for idx in idxes])
    positives = []
    negatives = []

    #we make all possible positive couples
    for i,sample in enumerate(samples):
        if(similarity):
            positives = positives + [np.concatenate((a, b),axis=None) for a in samples[i] for b in samples[i]]

        else:
            positives = positives + [np.concatenate((a, b),axis=None) for a in samples[i] for b in samples[i]]
            
        if (not separate):

            for l in range(n_negatives_per_positive):
                for j in range(len(sample)**2):
                    ranges = np.arange(0,n_classes)
                    ranges = np.delete(ranges,np.where(ranges == i))
                    k = np.random.choice(ranges,1)[0]

                    idx_k = int(np.random.randint(0,len(classes[k]))) 

                    negatives.append(np.concatenate((samples[i][j%len(sample)],classes[k][idx_k]),axis=None))
    if separate:
        counter = 0
        # for the number of required elements
        while(counter <= n_negatives_per_positive*(number_of_pairs - len(positives))):
            #we cycle over classes
            for i in range(n_classes): 
                #and sample index k, representing a random class to get an element from
                k = 0
                #retake k if i==k, since must be negative
                while(k==i):
                    k = int(np.random.randint(0,n_classes-1))  
                #sample one element from position 1, in class i
                if(len(samples[i])!=1):
                    idx_sample = int(np.random.randint(0,len(samples[i])-1))
                else:
                    idx_sample = 0
                #sample one element from class k for position 2
                if(len(classes[k])!=1):
                    idx_k = int(np.random.randint(0,len(classes[k])-1)) 
                else:
                    idx_k = 0
                if(similarity):
                    negatives.append([samples[i][idx_sample],classes[k][idx_k],torch.tensor(0).double()])
                else:
                    negatives.append([samples[i][idx_sample],classes[k][idx_k],torch.tensor(0)])
                counter +=1
            
    '''  
    if n_negatives_per_positive ==1:
        return positives + negatives[:len(positives)],len(positives),len(negatives[:len(positives)]),positives, negatives
    else:
    '''
    for i in range(len(positives)):
        labels.append(torch.tensor(1))
    for i in range(len(negatives)):
        labels.append(torch.tensor(0))
    return positives + negatives,labels,len(positives),len(negatives), positives, negatives
'''
general processor of raman spectra from txt, analogous to processor, but with TensorData-like instantiation in mind
'''
def general_tensor_processer (data_folder, task = 'classification', base = np.array([False,False]), CWT= [False,20], positional_encoding = False):
    folders = os.listdir(data_folder)
    if task == 'classification':
        n_classes = len(folders)
        data = []
        labels = []
        n = (pd.read_csv(data_folder + '/' + folders[0] + '/' + os.listdir(data_folder + '/' + folders[0])[0]  ,delimiter='\t',header=None).values).shape[0]
        if  not base.all() :
            base =  pd.read_csv(data_folder + '/' + folders[0]+ '/' + os.listdir(data_folder + '/' + folders[0])[0] ,delimiter='\t',header=None).values[:,0]
        for i,folder in enumerate(folders):
            spectra=os.listdir(data_folder + '/' + folder)

            for j,spectrum in enumerate(spectra):
                signa = pd.read_csv(data_folder + '/' + folder + '/' + spectrum,delimiter='\t',header=None).values
                signal_new = np.zeros(shape=(len(base),2))
                temp_0,idx = np.unique(signa[:,0],return_index=True)
                signal_new[:,0] = (base-np.min(base))/np.max(base)
                signal_new[:,1] = domain_converter(temp_0,signa[idx,1],base)
                signal_new[:,1] = (signal_new[:,1]-np.min(signal_new[:,1]))/np.max(signal_new[:,1])
                if(CWT[0]):
                        data.append(signal.cwt(signal_new[:,1], signal.ricker, np.arange(1, CWT[1])))
                        labels.append(torch.tensor(i))
                else:
                    if (not positional_encoding):
                        data.append(signal_new[:,1])
                        labels.append(torch.tensor(i))
                    else:
                        data.append(signal_new)
                        labels.append(torch.tensor(i))

        return n_classes, n, base, TensorDataset(torch.Tensor(data),torch.Tensor(labels))
    elif task == 'reconstruction':    

        data = []
        labels = []
        n = (pd.read_csv(data_folder + '/' + folders[0] + '/' + os.listdir(data_folder + '/' + folders[0])[0] ,delimiter='\t',header=None).values).shape[0]
        n_classes=n
        for i,folder in enumerate(folders):
            spectra=os.listdir(data_folder + '/' + folder) 
            for j,spectrum in enumerate(spectra):
                signa = pd.read_csv(data_folder + '/' + folder + '/' + spectrum,delimiter='\t',header=None).values
                signal_new = np.zeros(shape=(len(base),2))
                temp_0,idx = np.unique(signa[:,0],return_index=True)
                signal_new[:,0] = (base-np.min(base))/np.max(base)
                signal_new[:,1] = domain_converter(temp_0,signa[idx,1],base)
                signal_new[:,1] = (signal_new[:,1]-np.min(signal_new[:,1]))/np.max(signal_new[:,1])
                if(CWT[0]):
                    data.append(signal.cwt(signal_new[:,1], signal.ricker, np.arange(1, CWT[1])))
                    labels.append(torch.tensor(signal.cwt(signal_new[:,1], signal.ricker, CWT[1])))
                else:
                    if (not positional_encoding):
                        data.append(signal_new[:,1])
                        labels.append(signal_new[:,1])
                    else:
                        data.append(signal_new)
                        labels.append(signal_new[:,1])
        return n_classes, n, base, TensorDataset(torch.Tensor(data),torch.Tensor(labels))

    elif task == 'divide by class':    
        data_class = []
        data_map = []
        n = (pd.read_csv(data_folder + '/' + folders[0] + '/' + os.listdir(data_folder + '/' + folders[0])[0] ,delimiter='\t',header=None).values).shape[0]
        n_classes = len(folders)
        for i,folder in enumerate(folders):
            spectra=os.listdir(data_folder + '/' + folder)
            for j,spectrum in enumerate(spectra):
                signa = pd.read_csv(data_folder + '/' + folder + '/' + spectrum,delimiter='\t',header=None).values
                signal_new = np.zeros(shape=(len(base),2))
                temp_0,idx = np.unique(signa[:,0],return_index=True)
                signal_new[:,0] = (base-np.min(base))/np.max(base)
                signal_new[:,1] = domain_converter(temp_0,signa[idx,1],base)
                signal_new[:,1] = (signal_new[:,1]-np.min(signal_new[:,1]))/np.max(signal_new[:,1])
                if(CWT[0]):
                    data_class.append(signal.cwt(signal_new[:,1], signal.ricker, np.arange(1, CWT[1])))
                else:
                    if (not positional_encoding):
                        data_class.append(signal_new[:,1])  
                    else:
                        data_class.append(signal_new)  
            data_map.append(data_class)
            data_class = []


        
    
        return n_classes, n, base, data_map
    raise Exception("Warning: task'" + task + "'not recognized")
    return 0

'''
generates confusion matrix
'''
def confusion_matrix(output, labels,show = False): #we assume output and labels batched
    matrix= sklearn.metrics.confusion_matrix(labels,output)
    if(show):
        sklearn.metrics.ConfusionMatrixDisplay(matrix)
    return(matrix)
 
'''
augments data in a list of arrays
'''    
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