# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 21:35:35 2016

@author: damodara
Nystroem - Feature Map Generation, Classification and Error Approximation
"""


def Nystroem_FMAPGenration(Data, n_components= 50, gamma = 2., kernel='rbf', random_state=0):
    from sklearn.kernel_approximation import Nystroem

    NystInt = Nystroem(kernel='rbf', gamma=gamma, n_components=n_components, random_state=random_state)
    Nyst_Sampler = NystInt.fit(Data)
    return Nyst_Sampler


def Nystroem_ErrorApprox(Data, n_components=50, gamma = 2., random_state=0, K=None):
    import numpy as np
    from sklearn.metrics import pairwise
 
    Nyst_Sampler = Nystroem_FMAPGenration(Data, n_components=n_components, gamma=gamma, random_state=random_state)
    
    Ndata = np.shape(Data)[0]
    
    if K is None:
        K=pairwise.rbf_kernel(Data,gamma=gamma) 
    
    Phi = Nyst_Sampler.transform(Data)
    ApproxK = np.dot(Phi, Phi.T)
    #Error = np.linalg.norm(K-ApproxK)/np.linalg.norm(K)
    Error = np.sum((K-ApproxK)**2)/(Ndata**2)
    return Error
    
    
    
def Nystroem_Classification(NTrainData, NValData, NTestData, train_label, val_label, test_label,
                            n_components=50, gamma= 2., random_state=0,option=1,
                            classifier_opt=3, normalize=True, loss="log"):
                                
    
    from time import clock
    from minibatchSGDCV import minibatchSGDCV
    from minibatchSGD import minibatchSGD
    from sklearn.linear_model import RidgeClassifier, Ridge
    from sklearn import metrics
    import numpy as np
    
    Nyst_Sampler = Nystroem_FMAPGenration(NTrainData, n_components=n_components, gamma=gamma, random_state=random_state)
    
    TrainData = Nyst_Sampler.transform(NTrainData).copy()
    ValData = Nyst_Sampler.transform(NValData).copy()
    TestData = Nyst_Sampler.transform(NTestData).copy()   
    if normalize:
       from sklearn import preprocessing
       NormParam=preprocessing.StandardScaler().fit(TrainData)
       TrainData = NormParam.transform(TrainData)
       TestData = NormParam.transform(TestData)
       ValData = NormParam.transform(ValData)
        
    Ntrain, Nfeat = np.shape(TrainData)
    SGDAccuracy = 0.0
    RidgeAccuracy =0.0
    cvparam =0.0
    
    if (classifier_opt==3 or classifier_opt==1):
        st_time = clock()
        ncv=3
        cvparam, bestbatchparam, meanscore = minibatchSGDCV(ValData,val_label,ncv, option, loss=loss)
        end_time = clock()
        print('Time required for NystSGD CV is =', end_time-st_time)
        print('Nyst_SGD Cross Validation Completed')
        
        clf = minibatchSGD(TrainData, train_label, option, batchsize=cvparam['batchsize'], 
                                   alpha=cvparam['alpha'], eta0=cvparam['eta0'], loss=loss)
                                   
        SGDClassifiedlabel=clf.predict(TestData)
        if option==1:
            #SCDconfMat=confusion_matrix(test_label,SGDClassifiedlabel)
            SGDAccuracy=sum(test_label==SGDClassifiedlabel)/(float(len(test_label)))
        else:
            SGDAccuracy = metrics.mean_squared_error(test_label,SGDClassifiedlabel)
    #    print('SGD Completed')
    #    print("The classification accuracy with OrgFeatSGD =", SGDAccuracy)
    
    if (classifier_opt==3 or classifier_opt==2):
        if (classifier_opt!=3):
            cvparam = 0.0
            
        if option==1:
            clf = RidgeClassifier(alpha=0.01)#cvparam['alpha'])
            clf.fit(TrainData, train_label)
            RidgeClassifiedlabel = clf.predict(TestData)
            RidgeConfMat=metrics.confusion_matrix(test_label,RidgeClassifiedlabel)
            RidgeAccuracy=sum(test_label==RidgeClassifiedlabel)/(float(len(test_label)))       
        else:
            clf = Ridge(alpha=0.01)#cvparam['alpha'])
            clf.fit(TrainData, train_label)
            RidgeClassifiedlabel = clf.predict(TestData)
            RidgeAccuracy= metrics.mean_squared_error(test_label,RidgeClassifiedlabel)
    #        print("The classification accuracy with PsudeoRFRidge =",PRFRidgeAccuracy)
    
    return SGDAccuracy, RidgeAccuracy, cvparam
    