# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 15:40:47 2017

@author: damodara
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 15:48:47 2016

@author: damodara
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 14:50:12 2016

@author: damodara
test PRFF_Romain
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import datasets
import time
import copy
import sys
from time import clock
from revrand.basis_functions import RandomRBF, RandomLaplace, RandomCauchy, RandomMatern32, RandomMatern52, \
    FastFoodRBF, OrthogonalRBF, FastFoodGM, BasisCat
from revrand import Parameter, Positive
#import matplotlib as mpl
#mpl.use('Agg')

#%%
#Dataset Load
#====================================
##census Data
# from DatasetLoad import census_dataload
# TrainData, train_label, TestData, test_label=census_dataload()

## cpu Data
#from DatasetLoad import cpu_dataload
#TrainData, train_label, TestData, test_label = cpu_dataload()

## YearPredictionMSD Data
# from DatasetLoad import YearPredictionMSD_dataload
# TrainData, train_label, TestData, test_label = YearPredictionMSD_dataload()

#from DatasetLoad import cadata_dataload
#Data, label = cadata_dataload()

#==============================================================================
#from DatasetLoad import iris_dataload
#Data, label=iris_dataload()

# digits dataset
from DatasetLoad import digits_dataload
Data, label=digits_dataload()
#==============================================================================

# Adult Data
#from DatasetLoad import adult_dataload
#TrainData, train_label, TestData, test_label=adult_dataload()
##
#MNIST Data
from DatasetLoad import MNIST_dataload
Data, label=MNIST_dataload()
#
#from DatasetLoad import MNIST_official_split_dataload
#TrainData, train_label, TestData, test_label = MNIST_official_split_dataload()
###
##CIFAR-10 Dataset
#from DatasetLoad import cifar10_dataload
#TrainData, train_label, TestData, test_label=cifar10_dataload()

###
##Forest Data
#from DatasetLoad import forest_dataload
#Data, label= forest_dataload()

#%%
# Train and Test Data split
NData, Nfeat=np.shape(Data)
Nclass=len(np.unique(label))
from sklearn.cross_validation import train_test_split
indicies=np.arange(NData)
TrainData,TestData,train_label,test_label,tr_index,test_index=train_test_split(Data,label,indicies,test_size=0.14285,random_state=42)
# For extraction of validation data, to estimate bandwidth parameter, for error approximation, and CV for SGD classifier
indicies=np.arange(np.shape(TrainData)[0])
from sklearn.cross_validation import train_test_split
#TrainData, ValData, train_label, val_label,tr_index,val_index = train_test_split(TrainData,train_label,indicies,test_size=0.16667,random_state=42)
#TrainData, ValData, train_label, val_label,tr_index,val_index = train_test_split(TrainData,train_label,indicies,test_size=0.909,random_state=42)
TrainData, ValData, train_label, val_label,tr_index,val_index = train_test_split(TrainData,train_label,indicies,test_size=0.1,random_state=42)
## if already split, comment above
Ntrain, Nfeat=np.shape(TrainData)
Ntest=np.shape(TestData)[0]
Nval = np.shape(ValData)[0]
classnames=np.unique(train_label)
Nclass=len(np.unique(train_label))
train_label=np.squeeze(train_label)
test_label=np.squeeze(test_label)
val_label = np.squeeze(val_label)
#%%
normalize_bf_rff = True

# Data Normalization
if normalize_bf_rff:
   from sklearn import preprocessing
   NormParam=preprocessing.StandardScaler().fit(TrainData)
   TrainData = NormParam.transform(TrainData)
   TestData = NormParam.transform(TestData)
   ValData = NormParam.transform(ValData)
    
    #TrainData = preprocessing.maxabs_scale(TrainData, axis=1)
    #TestData = preprocessing.maxabs_scale(TestData, axis=1)
    #ValData = preprocessing.maxabs_scale(ValData, axis=1)
    
    #TrainData=preprocessing.minmax_scale(TrainData, feature_range=(-1, 1), axis=0)
    #TestData = preprocessing.minmax_scale(TestData, axis=1)
    #ValData = preprocessing.minmax_scale(ValData, axis=1)
    
    #TrainData = preprocessing.normalize(TrainData)
    #TestData = preprocessing.normalize(TestData)
    #ValData = preprocessing.normalize(ValData)
#%% Take a Subset of Data for the computation of PRFF approximation Error 
if Nval>=5000:
   Nsamples = 500
else:
   Nsamples = Nval
       

Rshuffle = np.random.permutation(Nval)
Data = copy.copy(ValData[Rshuffle,])
Data = Data[0:Nsamples,]

#%%
# Calculation of sigma using 5 percentile
from scipy.spatial import distance
Pdist=distance.pdist(Data,metric='euclidean')
nsigma=np.percentile(Pdist,5)
#nsigma = Nfeat
bestgamma = 1/(2*nsigma**2)
#sigma=2*bestgamma

del Pdist
#bestgamma = 1e-15 
#%%
import numpy
from sklearn.metrics.pairwise import rbf_kernel
from prff import PRFF
from prff_bharath import PRFF_Bharath
from sklearn.kernel_approximation import RBFSampler, Nystroem
from PRFGradient import LossfunctionForm2
from Nystroem_Method import Nystroem_ErrorApprox
from RFF_Classification import RFF_Form2_Classification
from Nystroem_Method import Nystroem_Classification
from orff import ORFF
#%%
gamma= copy.copy(bestgamma)
exact_gram = rbf_kernel(Data, gamma= gamma)
no_runs =1
n_components = 50
#alpha_range =[5.0] 
alpha_range = [50, 25, 10, 5, 1, 0.1, 0.5, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005, 0.00001]
#alpha_range = [10.0]
alpha_range = [0.1]
lamda_range =[100, 50, 25, 10, 5, 0.9, 0.1, 0.01, 0.001, 0.0001, 0.0]
lamda_range =[0.001]
n_components_range = [10, 25, 50, 100, 200, 300, 400, 500]#np.arange(1, n_components+1, 10)
n_components_range = [10,25]
prff = True
if prff:
    prff_p_error = np.zeros((no_runs,len(alpha_range), len(n_components_range), len(lamda_range)))
    prff_b_error = np.zeros((no_runs,len(alpha_range), len(n_components_range), len(lamda_range)))
    prff_p_sgd_accuracy = np.zeros((no_runs,len(alpha_range), len(n_components_range), len(lamda_range)))
    prff_b_sgd_accuracy = np.zeros((no_runs,len(alpha_range), len(n_components_range), len(lamda_range)))
    prff_p_ridge_accuracy = np.zeros((no_runs,len(alpha_range), len(n_components_range), len(lamda_range)))
    prff_b_ridge_accuracy = np.zeros((no_runs,len(alpha_range), len(n_components_range), len(lamda_range)))

rff = True
if rff:
    rff_error = np.zeros((no_runs,len(n_components_range)))
    rff_sgd_accuracy=np.zeros((no_runs,len(n_components_range)))
    rff_ridge_accuracy=np.zeros((no_runs,len(n_components_range)))
    nyst_error =np.zeros((no_runs,len(n_components_range)))
    nyst_sgd_accuracy=np.zeros((no_runs,len(n_components_range)))
    nyst_ridge_accuracy=np.zeros((no_runs,len(n_components_range)))
    orff_error = np.zeros((no_runs,len(n_components_range)))
    orff_sgd_accuracy=np.zeros((no_runs,len(n_components_range)))
    orff_ridge_accuracy=np.zeros((no_runs,len(n_components_range)))  

normalize = False    
for niter in range(no_runs):
    RFF = RBFSampler(gamma=gamma, n_components=n_components, random_state=None)
    RFF_Sampler = RFF.fit(TrainData)
    RFF_W = np.concatenate((RFF_Sampler.random_weights_, RFF_Sampler.random_offset_.reshape((1,-1))), axis=0)
    #ORF
    ORF = ORFF(gamma=gamma, n_components=n_components,random_state=None)
#    ORF = OrthogonalRBF(Xdim=Nfeat, nbases=n_components,lenscale = Parameter(nsigma, Positive()))
    ORF_Sampler = ORF.fit(TrainData)
    ORF_W = np.concatenate((ORF_Sampler.random_weights_, ORF_Sampler.random_offset_.reshape((1,-1))), axis=0)
#    ORF_W = np.concatenate((ORF.W/nsigma, RFF_Sampler.random_offset_.reshape((1,-1))), axis=0)
    



    b=0
    
    sampler_iter =0
    n_comp_iter=0
#    option =1 # for classification
    option =2 # for regression
    #classifier_opt =1 # for logistic regression
    classifier_opt =2 # for ridge regression
#    classifier_opt =3 # for logistic and ridge regression
    lamda_iter =0
    for lbda in lamda_range:
        alpha_iter=0
        for alpha in alpha_range:
            # Caution with PRFF: if you generate figures (gen_fig=True), you should use update_b=False because otherwise
            # the objective function on w changes as b changes from one iteration to the next
           
            
            INDEX_OF_PRFF = 0
            INDEX_OF_PRFF_Bharath= 1
            if prff:
                st_time = time.time()
                PRFF_sampler = PRFF_Bharath(gamma=gamma, n_components=n_components, random_state=0,
                            alpha=alpha, lbda=lbda, n_pass=1, minibatch_size=128, 
                            max_iter=1000,update_b=True)
                PRFF_sampler.fit(TrainData)
                PRFF_Bh_W = np.concatenate((PRFF_sampler.random_weights_,
                                       PRFF_sampler.random_offset_.reshape((1,-1))), axis=0)
                end_time = time.time()
                print('Time required to compute %d PRFF is =', n_components, end_time-st_time)                     
                print('Computation of PRFF Coefficients Completed')
                
#                PRFF_sampler_pupdate = PRFF_Bharath(gamma=gamma, n_components=n_components, random_state=None,
#                            alpha=alpha, lbda=lbda, n_pass=1, minibatch_size=128,
#                            max_iter=1000,update_b=True, philippe_update=True)
#                PRFF_sampler_pupdate.fit(TrainData)
#                PRFF_W_pupdate = np.concatenate((PRFF_sampler_pupdate.random_weights_,
#                                       PRFF_sampler_pupdate.random_offset_.reshape((1,-1))), axis=0)
#                print('Computation of PRFF Coefficients Completed')
                #%% 
            n_comp_iter=0
            
            for n_comp in n_components_range:
                
                if prff:
                    prff_b_error[niter,alpha_iter, n_comp_iter, lamda_iter] =LossfunctionForm2(Data, PRFF_Bh_W[:,range(n_comp)], gamma, exact_gram)        
                    prff_b_sgd_accuracy[niter,alpha_iter,n_comp_iter,lamda_iter],prff_b_ridge_accuracy[niter,alpha_iter,n_comp_iter, lamda_iter], cvparam = RFF_Form2_Classification(PRFF_Bh_W[:,range(n_comp)],b,TrainData,
                                        ValData, TestData, train_label, val_label, test_label, option,classifier_opt= classifier_opt, normalize=normalize, loss ="hinge")
                                      
#                    prff_p_error[niter,alpha_iter, n_comp_iter, lamda_iter] =LossfunctionForm2(Data, PRFF_W_pupdate[:,range(n_comp)], gamma, exact_gram)        
#                    prff_p_sgd_accuracy[niter,alpha_iter,n_comp_iter,lamda_iter],prff_p_ridge_accuracy[niter,alpha_iter,n_comp_iter, lamda_iter], cvparam = RFF_Form2_Classification(PRFF_W_pupdate[:,range(n_comp)],b,TrainData,
#                                        ValData, TestData, train_label, val_label, test_label, option,classifier_opt= classifier_opt, normalize=normalize,loss ="hinge")
                                    
                if lamda_iter==0 and alpha_iter==0 and rff:
                    rff_error[niter,n_comp_iter]= LossfunctionForm2(Data,RFF_W[:,range(n_comp)],gamma,exact_gram)
                    nyst_error[niter,n_comp_iter] = Nystroem_ErrorApprox(Data.copy(), n_components=n_comp, gamma=gamma, random_state=None,  K=exact_gram)
                    orff_error[niter,n_comp_iter]= LossfunctionForm2(Data,ORF_W[:,range(n_comp)],gamma,exact_gram)
                    rff_sgd_accuracy[niter,n_comp_iter],rff_ridge_accuracy[niter,n_comp_iter], RFcvparam = RFF_Form2_Classification(RFF_W[:,range(n_comp)],b,TrainData,
                                        ValData, TestData, train_label, val_label, test_label,option, classifier_opt= classifier_opt, normalize=normalize,loss ="hinge")
                    orff_sgd_accuracy[niter,n_comp_iter],orff_ridge_accuracy[niter,n_comp_iter], ORFcvparam = RFF_Form2_Classification(ORF_W[:,range(n_comp)],b,TrainData,
                                        ValData, TestData, train_label, val_label, test_label,option,classifier_opt= classifier_opt, normalize=normalize,loss ="hinge")
                    nyst_sgd_accuracy[niter,n_comp_iter], nyst_ridge_accuracy[niter,n_comp_iter], Nyst_cvparam =  Nystroem_Classification(TrainData, 
                                    ValData, TestData, train_label, val_label, test_label, n_components= n_comp, gamma=gamma, random_state=None,option=option,
                                    classifier_opt= classifier_opt, normalize=normalize,loss ="hinge") 
                    
                            
                n_comp_iter = n_comp_iter+1
                print('components iteration', n_comp_iter)    
       #%%                    
            
            alpha_iter = alpha_iter+1
        lamda_iter = lamda_iter+1   
        print('Lamda Interation', lamda_iter)
        print('================================')
    print('number of iterations', niter)
#%% Plot of error
filesave =False
pathname = 'D:\PostDocWork\LSML\RandomFourierFeatures\Results\check\cadata_oldgrad'
colours = ['b','g','r','y','k','b','g','r','m','y','k','b','g','r','m']
linestyle =['--','-','-','-','-','--','-.','-.','-.', '-','-','-','-','-','-','-']

#%% Save file
if filesave:
    filename = 'MNIST_PRFF_B_approxError.npz'
    fname = os.path.join(pathname,filename)
    Info = ['MNIST PRFF_B_ApproxError with different reg (lamda) parameter and learning rate (alpha)']
    np.savez(fname, prff_b_error= prff_b_error, prff_p_error= prff_b_error, lamda = lamda_range,
             alpha = alpha_range, n_components = n_components_range, Info = Info) 

    filename= 'MNIST_PRFF_B_Diff_Lamda_alpha_SGD_Ridge_Accuracy.npz'
    fname = os.path.join(pathname, filename)
    Info = ['MNIST PRFF_B_Classification Accuracy SGD and Ridge with different reg (lamda) parameter and learning rate (alpha)']
    np.savez(fname, prff_b_sgd_accuracy= prff_b_sgd_accuracy, prff_b_ridge_accuracy= prff_b_ridge_accuracy,
             prff_p_sgd_accuracy= prff_p_sgd_accuracy, prff_p_ridge_accuracy= prff_p_ridge_accuracy,
             lamda = lamda_range, alpha = alpha_range, n_components = n_components_range, Info = Info)
             
    if rff:
        filename= 'MNIST_RF_ORF_Nyst_SGD_Ridge_Accuracy.npz'
        fname = os.path.join(pathname, filename)
        Info = ['MNIST RF_ORF_Nyst_SGD_Ridge_Accuracy']
        np.savez(fname, rff_sgd_accuracy= rff_sgd_accuracy, rff_ridge_accuracy= rff_ridge_accuracy,
                 orff_sgd_accuracy= orff_sgd_accuracy, orff_ridge_accuracy= orff_ridge_accuracy,
                 nyst_ridge_accuracy=nyst_ridge_accuracy, nyst_sgd_accuracy=nyst_sgd_accuracy,
                 n_components = n_components_range, Info = Info) 
                 
        filename='MNIST_RF_ORF_Nyst_ApproxError.npz'
        fname = os.path.join(pathname, filename)
        Info = ['MNIST RF_Nyst_SGD_ApproxError']
        np.savez(fname, rff_error= rff_error, orff_error= orff_error, nyst_error= nyst_error,
                 n_components = n_components_range, Nsamples =Nsamples, Info = Info) 

#%% Plot of error
for i in range(len(lamda_range)):
    for j in range(len(alpha_range)):
        if prff:
            plt.figure(num= i, figsize = (10,12), dpi=100)
            Xaxis = n_components_range
            prffb_mean_score = np.mean(prff_b_error[:,j,:,i],axis=0)   
            prffb_std_score = np.std(prff_b_error[:,j,:,i],axis=0) 
            
            plt.errorbar(Xaxis, prffb_mean_score,prffb_std_score,linestyle=linestyle[j], color=colours[j],
                     linewidth=3.0, label= 'PRFF_%s' % alpha_range[j])
#            prffp_mean_score = np.mean(prff_p_error[:,j,:,i],axis=0)   
#            prffp_std_score = np.std(prff_p_error[:,j,:,i],axis=0)
#            plt.errorbar(Xaxis, prffp_mean_score,prffp_std_score,linestyle=linestyle[-j-1], color=colours[-j-1],
#                     linewidth=3.0, label= 'PRFF_P_%s' % alpha_range[j])
            plt.xlabel('Feature Expansions')
            plt.ylabel('Approximation Error')
    if rff:
       rff_mean = np.mean(rff_error, axis=0) 
       rff_std = np.std(rff_error, axis=0)
       Xaxis = n_components_range
       plt.errorbar(Xaxis, np.mean(rff_error, axis=0), np.std(rff_error, axis=0),marker ='s',
                   color ='b',linewidth=0.5, markersize=10.0,markeredgewidth=0.5,label = 'RFF')
       plt.errorbar(Xaxis, np.mean(orff_error,axis=0),np.std(orff_error,axis=0), marker ='s',
                   color ='r',linewidth=0.5, markersize=10.0,markeredgewidth=0.5,label = 'ORFF')
       plt.errorbar(Xaxis, np.mean(nyst_error,axis=0),np.std(nyst_error,axis=0), marker ='^',
                   color='g',linewidth=0.5, markersize=10.0,markeredgewidth=0.5,label = 'Nyst')
    plt.title('PRFF_B_Error_Approx_lamda_%s' %lamda_range[i])
    plt.legend()
    plt.show()
    if filesave:
        filename = 'MNIST_PRFF_B_Approx_Error_lamda_%s' %lamda_range[i]
        plt.savefig(os.path.join(pathname,filename+'.png'))
#        plt.savefig(os.path.join(pathname,filename+'.eps'))
        plt.close()

#%% Classification Accuracy    

 
for i in range(len(lamda_range)):
     for j in range(len(alpha_range)):
         if prff:
            plt.figure(num= 2, figsize = (10,12), dpi=100)
            Xaxis = n_components_range
            
            prff_b_sgd_mean = np.mean(prff_b_sgd_accuracy[:,j,:,i], axis=0)
            prff_b_sgd_std = np.std(prff_b_sgd_accuracy[:,j,:,i], axis=0)
#            prff_p_sgd_mean = np.mean(prff_p_sgd_accuracy[:,j,:,i], axis=0)
#            prff_p_sgd_std = np.std(prff_p_sgd_accuracy[:,j,:,i], axis=0)
            plt.errorbar(Xaxis, prff_b_sgd_mean,prff_b_sgd_std,linestyle=linestyle[j], 
                     color=colours[j], marker ='o',linewidth=3.0, markersize=10.0, label= 'PRFF_%s ' % alpha_range[j])
#            plt.errorbar(Xaxis, prff_p_sgd_mean,prff_p_sgd_std,linestyle=linestyle[-j-1], 
#                     color=colours[-j-1], marker ='p',linewidth=3.0,markersize=10.0, label= 'PRFF_P_%s' % alpha_range[j])
            plt.xlabel('Feature Expansions')
             

     if rff:
        Xaxis = n_components_range
        rff_sgd_acc_mean = np.mean(rff_sgd_accuracy, axis=0)
        rff_sgd_acc_std = np.std(rff_sgd_accuracy, axis=0)
        orff_sgd_acc_mean = np.mean(orff_sgd_accuracy, axis=0)
        orff_sgd_acc_std = np.std(orff_sgd_accuracy, axis=0)
        nyst_sgd_acc_mean = np.mean(nyst_sgd_accuracy, axis=0)
        nyst_sgd_acc_std = np.std(nyst_sgd_accuracy, axis=0)
        plt.errorbar(Xaxis, rff_sgd_acc_mean,rff_sgd_acc_std, marker ='s',
                   color ='r',linewidth=0.5, markersize=10.0,markeredgewidth=0.5,label = 'RFF')
        plt.errorbar(Xaxis, orff_sgd_acc_mean,orff_sgd_acc_std, marker ='*',
                   color ='r',linewidth=0.5, markersize=10.0,markeredgewidth=0.5,label = 'ORFF')
        plt.errorbar(Xaxis, nyst_sgd_acc_mean,nyst_sgd_acc_std, marker ='^',
                   color ='g',linewidth=0.5, markersize=10.0,markeredgewidth=0.5,label = 'nyst')    

     if option ==1:
         plt.ylabel('Accuracy')
         plt.legend(loc=4)
         plt.title('PRFF_B_SGD_Classification_lamda_%s' %lamda_range[i])
     else:
         plt.ylabel('Mean square error')
         plt.legend(loc=1)
         plt.title('PRFF_B_SGD_Regression_lamda_%s' %lamda_range[i])
     plt.show()
     if filesave:
         filename = 'PRFF_B_SGD_Classification_lamda_%s' %lamda_range[i]
         plt.savefig(os.path.join(pathname, filename+'.png'))
 #        plt.savefig(os.path.join(filename+'.eps'))
         plt.close()

#%% Ridge Classifier or Regression
for i in range(len(lamda_range)):
    for j in range(len(alpha_range)):
        if prff:
            plt.figure(num= i, figsize = (10,12), dpi=100)
            Xaxis = n_components_range
            prff_b_ridge_mean = np.mean(prff_b_ridge_accuracy[:,j,:,i], axis=0)
            prff_b_ridge_std = np.std(prff_b_ridge_accuracy[:,j,:,i], axis=0)
            prff_p_ridge_mean = np.mean(prff_p_ridge_accuracy[:,j,:,i], axis=0)
            prff_p_ridge_std = np.std(prff_p_ridge_accuracy[:,j,:,i], axis=0)
            plt.errorbar(Xaxis, prff_b_ridge_mean,prff_b_ridge_std,linestyle=linestyle[j], 
                     color=colours[j], marker ='o',linewidth=3.0, markersize=10.0, label= 'PRFF_%s ' % alpha_range[j])
#            plt.errorbar(Xaxis, prff_p_ridge_mean,prff_p_ridge_std,linestyle=linestyle[-j-1], 
#                     color=colours[-j-1], marker ='p',linewidth=3.0,markersize=10.0, label= 'PRFF_P_%s' % alpha_range[j])
            plt.xlabel('Feature Expansions')

    if rff:
        Xaxis = n_components_range
        rff_ridge_acc_mean = np.mean(rff_ridge_accuracy, axis=0)
        rff_ridge_acc_std = np.std(rff_ridge_accuracy, axis=0)
        orff_ridge_acc_mean = np.mean(orff_ridge_accuracy, axis=0)
        orff_ridge_acc_std = np.std(orff_ridge_accuracy, axis=0)
        nyst_ridge_acc_mean = np.mean(nyst_ridge_accuracy, axis=0)
        nyst_ridge_acc_std = np.std(nyst_ridge_accuracy, axis=0)
        plt.errorbar(Xaxis, rff_ridge_acc_mean,rff_ridge_acc_std, marker ='s',
                   color ='r',linewidth=0.5, markersize=10.0,markeredgewidth=0.5,label = 'RFF')
        plt.errorbar(Xaxis, orff_ridge_acc_mean,orff_ridge_acc_std, marker ='*',
                   color ='r',linewidth=0.5, markersize=10.0,markeredgewidth=0.5,label = 'ORFF')
        plt.errorbar(Xaxis, nyst_ridge_acc_mean,nyst_ridge_acc_std, marker ='^',
                   color ='g',linewidth=0.5, markersize=10.0,markeredgewidth=0.5,label = 'nyst')    
    
    if option ==1:
        plt.ylabel('Accuracy')
        plt.legend(loc=4)
        plt.title('PRFF_B_Ridge_Classification_lamda_%s' %lamda_range[i])
    else:
        plt.ylabel('Mean square error')
        plt.legend(loc=1)
        plt.title('PRFF_B_Ridge_Regression_lamda_%s' %lamda_range[i])
    plt.show()
    
    if filesave:
        filename = 'PRFF_B_Ridge_Classification_lamda_%s' %lamda_range[i]
        plt.savefig(os.path.join(pathname, filename+'.png'))
#        plt.savefig(os.path.join(filename+'.eps'))
        plt.close()
  

