# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 14:44:40 2016

@author: damodara
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 13:40:24 2016

@author: damodara
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import datasets
import time
import copy
import sys
from time import clock
#import matplotlib as mpl
#mpl.use('Agg')

#%%
#Dataset Load
#====================================
##census Data
#from DatasetLoad import census_dataload
#TrainData, train_label, TestData, test_label=census_dataload()
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
#from DatasetLoad import MNIST_dataload
#Data, label=MNIST_dataload()
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
TrainData, ValData, train_label, val_label,tr_index,val_index = train_test_split(TrainData,train_label,indicies,test_size=0.16667,random_state=42)
## if already split, comment above
Ntrain, Nfeat=np.shape(TrainData)
Ntest=np.shape(TestData)[0]
classnames=np.unique(train_label)
Nclass=len(np.unique(train_label))
train_label=np.squeeze(train_label)
test_label=np.squeeze(test_label)
val_label = np.squeeze(val_label)
#%%
# Data Normalization
from sklearn import preprocessing
NormParam=preprocessing.StandardScaler().fit(TrainData)
TrainData = NormParam.transform(TrainData)
TestData = NormParam.transform(TestData)
ValData = NormParam.transform(ValData)

TrainData = np.concatenate((TrainData, np.ones((np.shape(TrainData)[0],1), dtype= np.float64)), axis=1)
TestData = np.concatenate((TestData, np.ones((np.shape(TestData)[0],1), dtype= np.float64)), axis=1)
ValData = np.concatenate((ValData, np.ones((np.shape(ValData)[0],1), dtype= np.float64)), axis=1)
#%% Take a Subset of Data for the computation of PRFF approximation Error 
Nsamples = 2000
Rshuffle = np.random.permutation(Ntrain)
Data = copy.copy(TrainData[Rshuffle,])
Data = Data[0:Nsamples,]

#%% finding gamma parameter based on classification random fourier features
'''
from RFFCrossValidation import RFFCrossValidation
gamma = 2.0**(np.concatenate((np.arange(-25,-5,0.5),np.arange(-4,-0.5,0.5),np.arange(1,7,1))))
bestgamma, gamma_accuracy, gamma = RFFCrossValidation(ValData, val_label, NFexp=500, nfold= 5, 
                                                     gamma, option=1, C=0)
'''
#%%
#bestgamma=0.01105  # for ADULT Data
#bestgamma =0.00138  # for MNIST Data
#bestgamma = 0.000244141 # for CIFAR-10 Data
#%%
# Calculation of sigma using 5 percentile
from scipy.spatial import distance
Pdist=distance.pdist(Data,metric='euclidean')
nsigma=np.percentile(Pdist,5)
bestgamma = 1/(2*nsigma**2)
#%%
sigma=np.sqrt(2*bestgamma)
#%% Random Fourier Feature Coeficients
NFexp = 101 # Number of feature expansions to learn by PRFF
Ndata, Nfeat = np.shape(TrainData)
s=np.random.seed(0)
#WRD = np.random.normal(0,sigma,(Nfeat,NFexp))
### if bias has to included in RFF, PRFF, uncomment the following lines
s=np.random.seed(0)
WRD = np.random.normal(0,sigma,(Nfeat-1,NFexp))
s=np.random.seed(0)
b = np.random.rand(NFexp)
b= b[:,np.newaxis]*(2*np.pi)
WRD = np.concatenate((WRD,b.T), axis=0)

#%% Pseudo Random Fourier Coefficient Generation
from PseudoRFFGeneration import PRFF_GenerationForm2
batchsize=128 # batch size to learn the feature expansion
eta = 0.1 # learning rate in gradient descent
RegW = 0.1 # regularization parameter

W2 = PRFF_GenerationForm2(WRD, NFexp, bestgamma, batchsize, copy.copy(TrainData), eta, RegW)
#%% Romain Implementation

 
#%% Approximation Error based on a subset of samples 
from PRFGradient import LossfunctionForm2
D=np.shape(W2)[1]
#Ndata,Nfeat=np.shape(Data)
#WRD = np.random.normal(0,sigma,(Nfeat,D))
#
L2=[]*NFexp
LR2=[]*NFexp 
# True Kernel
from sklearn.metrics import pairwise
K=pairwise.rbf_kernel(Data,gamma=sigma**2/2)
#
for i in np.arange(0,D,50):
    if (i==0):
        i=i+1
        i2 = 1
    else:
        i2= int (i/2)
        
    L2.append(LossfunctionForm2(Data,W2[:,range(i)],sigma,K))
    LR2.append(LossfunctionForm2(Data,WRD[:,range(i)],sigma,K))
    print('Feature Expansion', i, 'is over')
#%% save the approximation errors in the file (as npz file)
'''  
pathname = os.path.join('D:\PostDocWork\LSML\RandomFourierFeatures\Results\KernelErrorApproxandclassification\MNIST\withdatanormalization\RegW0.1_alpha0.1')
filename='MNIST_RFF_PRFF_Approx_Error_W2_Reg0.1.npz'
fname = os.path.join(pathname,filename)
Info = ['MNIST Approx Error of Pusedo RFF and RF using form 2 (cos),']
np.savez(fname,PRFApproxErrorForm2=L2, RFApproxErrorForm2=L2, Info=Info)   
'''      
#%%
from PRFGradient import ApproxKernelLossfunction1, ApproxKernelLossfunction2 
ErrRFW2, KRFW2 = ApproxKernelLossfunction2(Data,WRD,sigma,K)
ErrPRFW2, KPRFW2 = ApproxKernelLossfunction2(Data,W2,sigma,K)
 
plt.figure(num=1)
plt.imshow(K,cmap="hot",interpolation="nearest")
plt.title('Orignal Kernel'+ ' gamma = '+ str(round(bestgamma,4)))
plt.colorbar()
plt.figure(num=2, figsize=(12, 10), dpi=100)
ax1=plt.subplot(211)
plt.imshow(K,cmap="hot",interpolation="nearest")
plt.title('OrgKernel'+ ' gamma = '+ str(round(bestgamma,4)))
plt.colorbar()
 
ax2=plt.subplot(223)
plt.imshow(KRFW2,cmap="hot",interpolation="nearest")
plt.title('RFKernelW2'+ ' App.Error = '+ str(round(ErrRFW2,4)))
plt.colorbar()
 
ax4=plt.subplot(224)
plt.imshow(KPRFW2,cmap="hot",interpolation="nearest")
plt.title('PRFKernel W2'+ ' App.Error = '+ str(round(ErrPRFW2,4)))
plt.colorbar()
'''
#pathname = os.path.join('D:\PostDocWork\LSML\RandomFourierFeatures\Results' +
#'\KernelErrorApproxandclassification\MNIST\withdatanormalization')
filename = 'MNIST_OrgKernel_RFApproximations_RegW0.1'
plt.savefig(os.path.join(pathname,filename+'.png'))
plt.savefig(os.path.join(pathname,filename+'.eps'))
'''
#%%
plt.figure(num=3,figsize=(12, 10), dpi=100)
Xaxis = np.arange(0,D,50)

ax2=plt.subplot(111)
PRF2,= plt.plot(Xaxis,L2,label='PsudeoRF-Form2') 
RF2,= plt.plot(Xaxis,LR2,label='RF-Form2')
plt.title('Kernel Approximation-Form 2 batchsize 128')
plt.xlabel('Feature Expansions')
plt.ylabel('Approximation Error')
plt.legend(handles=[PRF2,RF2],loc=1)

'''
filename = 'MNIST_RF_PRFF_Approx_Error_W1_W2'
plt.savefig(os.path.join(pathname,filename+'.png'))
plt.savefig(os.path.join(pathname,filename+'.eps'))
'''
#%% Classification
D=np.shape(W2)[1]
 
b=np.zeros((D,1))
bn=copy.copy(b)
 
 
PRFSGDAccuracy2=np.zeros((D,1)) 
PRFRidgeAccuracy2=np.zeros((D,1)) 
RFOrgRidgeAccuracy2 =np.zeros((D,1))
RFOrgSGDAccuracy2 =np.zeros((D,1))
#%% SGD Classifier (logistic regression) on original features
## Org SGD
from minibatchSGDCV import minibatchSGDCV
from minibatchSGD import minibatchSGD
st_time = clock()
cvparam, bestbatchparam, meanscore = minibatchSGDCV(ValData,val_label,ncv=3, option=1)
end_time = clock()
print('Time required for OrgSGD CV is =', end_time-st_time)
print('SGD Cross Validation Completed')
clf = minibatchSGD(TrainData, train_label, option=1, batchsize=cvparam['batchsize'], 
                               alpha=cvparam['alpha'], eta0=cvparam['eta0'])
SGDClassifiedlabel=clf.predict(TestData)
#SCDconfMat=confusion_matrix(test_label,SGDClassifiedlabel)
SGDAccuracy=sum(test_label==SGDClassifiedlabel)/(float(len(test_label)))
print('SGD Completed')
print("The classification accuracy with OrgFeatSGD =", SGDAccuracy)
#%% SGD Classification on Pseudo Random Fourier Features (and Random FF)

for k in np.arange(0,D,50):
    if (k==0):
        k=k+1
        k2 = k+1
    else:
        k2 = int (k/2)
        
        
   
   # Random Fourier Feature classification
    from RFF_Classification import RFF_Form2_Classification
    RFOrgSGDAccuracy2[k],RFOrgRidgeAccuracy2[k], RFcvparam = RFF_Form2_Classification(WRD[:,range(k)],b,TrainData,
                                    ValData, TestData, train_label, val_label, test_label)
     
    # Psudeo Random Fourier Feature classification
    PRFSGDAccuracy2[k],PRFRidgeAccuracy2[k], RFcvparam = RFF_Form2_Classification(W2[:,range(k)],b,TrainData,
                                    ValData, TestData, train_label, val_label, test_label)
                                    
    print('+++++++++++++++++++++++++++++++++++++++++')
    print('Classification Results with formulation W2')
    print('The classification accuracy with PsuedoRFSGD =', PRFSGDAccuracy2[k])                               
    print('The classification accuracy with RFSGD =', RFOrgSGDAccuracy2[k]) 
    print('The classification accuracy with PsuedoRFRidge =', PRFRidgeAccuracy2[k]) 
    print('The classification accuracy with RFRidge =', RFOrgRidgeAccuracy2[k])
    print('Feature expansion', k, 'is over')
    print('===================================================')
#%%
ind = PRFSGDAccuracy2>0 
PRFSGDAccuracy2 = PRFSGDAccuracy2[ind] 
PRFRidgeAccuracy2 = PRFRidgeAccuracy2[ind] 
RFOrgSGDAccuracy2 = RFOrgSGDAccuracy2[ind] 
RFOrgRidgeAccuracy2 = RFOrgRidgeAccuracy2[ind]
#%%
'''
filename='MNIST_RFF_PRFF_ClassificationResults_Form2.npz'
fname = os.path.join(pathname,filename)
Info = ['MNIST Classification Results of Pusedo RFF and RF using formulation 2 (cos),'+
'classifiers =minibatchSGD, Ridge Regression']
np.savez(fname,RFOrgSGDAccuracy2=RFOrgSGDAccuracy2,RFOrgRidgeAccuracy2=RFOrgRidgeAccuracy2,
         PRFSGDAccuracy2=PRFSGDAccuracy2, PRFRidgeAccuracy2= PRFRidgeAccuracy2,
         OrgFeatSGDAccuracy=SGDAccuracy, Info=Info)
'''
#%%

xaxis = np.arange(0,D,50)
plt.figure(num=4,figsize=(12, 10), dpi=100)
ax2=plt.subplot(111)
RFSGD2, = plt.plot(xaxis,PRFSGDAccuracy2,label='PsudeoRFSGD2')
RFOrgSGD2, =plt.plot(xaxis,RFOrgSGDAccuracy2, label='OrgRFSGD2')
plt.title('MinibatchSGD-Formulation-W2')
plt.xlabel('Feature Expansions'),plt.ylabel('Classification rate')
plt.legend(handles=[RFSGD2,RFOrgSGD2],loc=4)
#plt.ylim((0,1))
'''
filename = 'MNIST_RF_PRFF_MinibatchSGDAccuracy_W2'
plt.savefig(os.path.join(pathname,filename+'.png'))
plt.savefig(os.path.join(pathname,filename+'.eps'))
'''
##
plt.figure(num=5,figsize=(12, 10), dpi=100)
ax2=plt.subplot(111)
RFRidge2, = plt.plot(xaxis,PRFRidgeAccuracy2,label='PsudeoRFRidge2')
RFOrgRidge2, =plt.plot(xaxis,RFOrgRidgeAccuracy2, label='OrgRFRidge2')
plt.title('MinibatchRidge-Formulation-W2')
plt.xlabel('Feature Expansions'),plt.ylabel('Classification rate')
plt.legend(handles=[RFRidge2,RFOrgRidge2],loc=4)
#plt.ylim((0,1))
'''
filename = 'MNIST_RF_PRFF_MinibatchRidgeAccuracy_W2'
plt.savefig(os.path.join(pathname,filename+'.png'))
plt.savefig(os.path.join(pathname,filename+'.eps'))
'''
 

