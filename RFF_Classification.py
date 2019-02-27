# -*- coding: utf-8 -*-
"""
Created on Wed May 25 16:58:52 2016

@author: damodara
"""

"""
Psuedo Random fourier features classification using formulation 1 and formulation 2
formulation 1  = cos(w(x-y)) 
formulation 2  = cos(wx)cos(wy)

RFF_Form1_Classification -- Classification of Pseudo Random Fourier Features (Random Fourier Features)
by SGD classifier (logistic regression) and Ridge regression - Formulation 1
    Input:
        WN  -- Pseudo RFF (or RFF) coefficient matrix
        b   -- already included in WN (so it is vector of zeros)
        TrainData, ValData, TestData -- Training, Validation, and Testing data
    Output:
        PRFSGDAccuracy, PRFRidgeAccuracy -- Classification accuracy by SGD classifier and Ridge regression classifier


RFF_Form2_Classification --Classification of Pseudo Random Fourier Features (Random Fourier Features)
by SGD classifier (logistic regression) and Ridge regression- Formulation 2

    Input: Same as above
        RFcvparam -- if the hyperparameters of the SGD classifier is already known
                        (to reuse the parameters computed from PRFF for RFF or vice-versa)
    Output : Same as above
       RFcvparam  -- computed hyperparameters for SGD classifier
"""

def RFF_Form1_Classification(WN, b,TrainData, ValData, TestData, train_label, val_label, test_label, option=1):
    import numpy as np 
    import time
    import copy
    import sys
    from time import clock
    
    D = np.shape(WN)[1]
    b=np.zeros((D,1))
    bn=copy.copy(b)
    PRFSGDAccuracy=np.zeros((D,1))    
    PRFRidgeAccuracy=np.zeros((D,1))
    #interval = np.arange(0,D,10)
    interval = [D]
    
    for k in interval:
        if (k==0):
            k=k+1
        W=WN[:,range(k)]
        b=bn[range(k)]
        RFTrainData= FeaturemapTransformation_Form1(W,TrainData)           
        RFTestData= FeaturemapTransformation_Form1(W,TestData)
        
        ## Psuedo RF SGD
        from minibatchSGDCV import minibatchRFSGDCV
        from minibatchSGD import RFminibatchSGD
        st_time = clock()
        ncv=3
        RFcvparam, RFbestbatchparam, RFmeanscore = minibatchRFSGDCV(ValData,val_label,ncv,W,b,
                                                                    option=1, RFoption=1)
        end_time = clock()
        print('PRFSGD Cross Validation Completed')
        print('Time required for PRFSGD CV is =', end_time-st_time)
        PRFclf = RFminibatchSGD(TrainData,train_label,W,b,option=1,batchsize=RFcvparam['batchsize'], 
                                           alpha=RFcvparam['alpha'], eta0=RFcvparam['eta0'], RFoption=1)
        PRFSGDClassifiedlabel=PRFclf.predict(RFTestData)
        #SCDconfMat=confusion_matrix(test_label,SGDClassifiedlabel)
        PRFSGDAccuracy[k-1]=sum(test_label==PRFSGDClassifiedlabel)/(float(len(test_label)))
#        print('RFSGD Completed')    
#        print("The classification accuracy with PsudeoRFSGD =",PRFSGDAccuracy[k-1])      
         
        
        ## Ridge regression Psuedo RF
        from sklearn.linear_model import RidgeClassifier
        from sklearn.metrics import confusion_matrix
        clf = RidgeClassifier(alpha=0.1)
        clf.fit(RFTrainData, train_label)
        RFRidgeClassifiedlabel = clf.predict(RFTestData)
        RFRidgeConfMat=confusion_matrix(test_label,RFRidgeClassifiedlabel)
        PRFRidgeAccuracy[k-1]=sum(test_label==RFRidgeClassifiedlabel)/(float(len(test_label)))
#        print("The classification accuracy with PsudeoRFRidge =",PRFRidgeAccuracy[k-1])
#        print("The feature expansion",k, "is over")
#        print('+++++++++++++++++++++++++++++++++++++++')
        
    ind = PRFSGDAccuracy>0
    PRFSGDAccuracy = PRFSGDAccuracy[ind]
    PRFRidgeAccuracy = PRFRidgeAccuracy[ind]    
    return PRFSGDAccuracy, PRFRidgeAccuracy
#%%
def FeaturemapTransformation_Form1(W,Data):
    import numpy as np
    D = np.shape(W)[1]
    Featuremap = np.sqrt(1/D)*np.concatenate((np.cos(np.dot(Data,W)), np.sin(np.dot(Data,W))), axis=1)
    return Featuremap
    
#%%
def RFF_Form2_Classification(NWN, b,TrainData, ValData, TestData, train_label, val_label, test_label,option=1, 
                             classifier_opt =3, RFcvparam =None, normalize = True,loss ="log"):
    import numpy as np 
    import time
    import copy
    import sys
    from time import clock
    from sklearn import metrics
    
    WN = NWN.copy()
    Np, D = np.shape(WN)
    Ntrain, Nfeat = np.shape(TrainData)
    
    if (Np !=Nfeat):
        b = WN[Np-1,:]
        WN = np.delete(WN,Np-1, axis=0)
        
    bn=copy.copy(b)
    PRFSGDAccuracy=np.zeros((D,1))    
    PRFRidgeAccuracy=np.zeros((D,1))
    option =int(option)
    Ntest = np.shape(TestData)[0]
    Nval = np.shape(ValData)[0]
    #interval = np.arange(0,D,10)
    interval = [D]
    for k in interval:
        if (k==0):
            k=k+1
        W=WN[:,range(k)]
        b=bn[range(k)]
        RFTrainData= np.sqrt(2.0/(k))*(np.cos(np.dot(TrainData,W)+np.tile(b.T,(Ntrain,1))))           
        RFTestData=np.sqrt(2.0/(k))*(np.cos(np.dot(TestData,W)+np.tile(b.T,(Ntest,1)))) 
        RFValData = np.sqrt(2.0/(k))*(np.cos(np.dot(ValData,W)+np.tile(b.T,(Nval,1))))
        
        if normalize:          
           from sklearn import preprocessing
           NormParam=preprocessing.StandardScaler().fit(RFTrainData)
           RFTrainData = NormParam.transform(RFTrainData)
           RFTestData = NormParam.transform(RFTestData)
           RFValData = NormParam.transform(RFValData)
        
        
        
        if (classifier_opt ==3 or classifier_opt ==1):
        ## Psuedo RF SGD
            from minibatchSGDCV import minibatchRFSGDCV, minibatchSGDCV
            from minibatchSGD import RFminibatchSGD, minibatchSGD
            if RFcvparam is None:
                st_time = clock()
                ncv=3
#                RFcvparam, RFbestbatchparam, RFmeanscore = minibatchRFSGDCV(ValData,val_label,ncv,W,b,
#                                                                        option, RFoption=2)
                RFcvparam, RFbestbatchparam, RFmeanscore = minibatchSGDCV(RFValData,val_label,ncv,option, loss=loss)
                end_time = clock()
                print('PRFSGD Cross Validation Completed')
                print('Time required for PRFSGD CV is =', end_time-st_time)
                print('CVParam' , RFcvparam['alpha'], RFcvparam['eta0'])
               #
#            PRFclf = RFminibatchSGD(TrainData,train_label,W,b,option,batchsize=RFcvparam['batchsize'], 
#                                               alpha=RFcvparam['alpha'], eta0=RFcvparam['eta0'], RFoption=2)
            PRFclf = minibatchSGD(RFTrainData,train_label,option,batchsize=RFcvparam['batchsize'], 
                                               alpha=RFcvparam['alpha'], eta0=RFcvparam['eta0'],loss=loss)
            
            PRFSGDClassifiedlabel=PRFclf.predict(RFTestData)
            #SCDconfMat=confusion_matrix(test_label,SGDClassifiedlabel)
            
            if (option==1):
                PRFSGDAccuracy[k-1]=sum(test_label==PRFSGDClassifiedlabel)/(float(len(test_label)))
            else:
                PRFSGDAccuracy[k-1]=metrics.mean_squared_error(test_label,PRFSGDClassifiedlabel)
    #        print('RFSGD Completed')    
    #        print("The classification accuracy with PsudeoRFSGD =",PRFSGDAccuracy[k-1])      
            ind = PRFSGDAccuracy>0
             
        if (classifier_opt==3 or classifier_opt==2):
            ## Ridge regression Psuedo RF
            from sklearn.linear_model import RidgeClassifier, Ridge, RidgeClassifierCV, RidgeCV
            
            if (classifier_opt!=3):
                RFcvparam = 0.0
                 
    
            if (option==1):
                clf = RidgeClassifier(alpha=0.01)#RFcvparam['alpha'])
                clf.fit(RFTrainData, train_label)
                RFRidgeClassifiedlabel = clf.predict(RFTestData)
                RFRidgeConfMat=metrics.confusion_matrix(test_label,RFRidgeClassifiedlabel)
                PRFRidgeAccuracy[k-1]=sum(test_label==RFRidgeClassifiedlabel)/(float(len(test_label)))
                
            else:
                clf = Ridge(alpha=0.01)#RFcvparam['alpha'])
                clf.fit(RFTrainData, train_label)
                RFRidgeClassifiedlabel = clf.predict(RFTestData)            
                PRFRidgeAccuracy[k-1]=metrics.mean_squared_error(test_label, RFRidgeClassifiedlabel)
    #        print("The classification accuracy with PsudeoRFRidge =",PRFRidgeAccuracy[k-1])
    #        print("The feature expansion",k, "is over")
    #        print('+++++++++++++++++++++++++++++++++++++++') 
            ind = PRFRidgeAccuracy>0
        
     
     
    PRFSGDAccuracy = PRFSGDAccuracy[ind]
    PRFRidgeAccuracy = PRFRidgeAccuracy[ind]
     
    return PRFSGDAccuracy, PRFRidgeAccuracy, RFcvparam
    
#%%

