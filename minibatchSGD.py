# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 16:24:22 2016

@author: damodara
This files performs mini-batch SGD classification (logistic regression) 
minibatchSGD --Minibatch SGD classifier for original features
    Input:
        TrainData, trainlabel
        
    Output:
        zclf  -- trained model of SGD classifier

RFminibatchSGD --Minibatch SGD classifier for Pseudo RFF (or RFF) features
    Input:
        TrainData, trainlabel
        W   -- PRFF or RFF coefficient matrix
        b   -- bias term (if it is already included in W, then it is vector of zeros)
        RFoption -- 1 for formulation 1
                    2 for formulation 2
    Output:
        zclf  -- trained model of SGD classifier
"""

def minibatchSGD(NTrainData,Ntrainlabel, option=1, loss="log", batchsize=64, alpha=0.1, eta0=0.1):
    import numpy as np
    
    TrainData = NTrainData.copy()
    trainlabel = Ntrainlabel.copy()
    #batchsize=50
    Ndata=np.shape(TrainData)[0]
    nloops=int(np.round(Ndata/batchsize))
    #SGD
    if (option==1):
        from sklearn.linear_model import SGDClassifier
        from sklearn.metrics import confusion_matrix
        zclf = SGDClassifier(loss=loss, penalty="l2", learning_rate ='optimal', alpha=alpha, eta0=eta0,average=True,n_jobs=1)
        classnames=np.unique(trainlabel)
    else:
        from sklearn.linear_model import SGDRegressor
        zclf = SGDRegressor(loss="squared_loss", penalty="l2",learning_rate='optimal',alpha=alpha, eta0=eta0,average=False)
    #
    n_epo = 5
    for n in range(n_epo):
        Rshuffle=np.random.permutation(Ndata)
        Data=TrainData[Rshuffle,]
        label=trainlabel[Rshuffle]
        if (option==1):
            for i in range(nloops):
                if i<(nloops-1):
                    st=int(i*batchsize)
                    last=int(((i+1)*batchsize))
                else:
                    st=int(i*batchsize)
                    last=int(Ndata)
                
                zclf.partial_fit(Data[st:last,],label[st:last],classes=classnames)
        else:
            # Regression
            #
            for i in range(nloops):
                if i<(nloops-1):
                    st=i*batchsize
                    last=((i+1)*batchsize)
                else:
                    st=i*batchsize
                    last=Ndata
                
                zclf.partial_fit(Data[st:last,],label[st:last])
        #print(zclf)
    return zclf
#%%
    
def RFminibatchSGD(NTrainData,Ntrainlabel,W,b,option=1, batchsize=64, alpha=0.1, eta0=0.1, RFoption=2):
    import numpy as np
    TrainData = NTrainData.copy()
    trainlabel = Ntrainlabel.copy()    
    
    Ndata=np.shape(TrainData)[0]
    nloops=int(np.round(Ndata/batchsize))
    
    #SGD
    if (option==1):
        from sklearn.linear_model import SGDClassifier
        from sklearn.metrics import confusion_matrix
        zclf = SGDClassifier(loss="log", penalty="l2",alpha=alpha, learning_rate= 'optimal', eta0=eta0, n_jobs=1)
        classnames=np.unique(trainlabel)
    else:
        from sklearn.linear_model import SGDRegressor
        zclf = SGDRegressor(loss="squared_loss", penalty="l2",alpha=alpha, learning_rate= 'optimal',eta0=eta0)  
    
    n_epo = 5
    for n in range(n_epo):
        Rshuffle=np.random.permutation(Ndata)
        Data=TrainData[Rshuffle,]
        label=trainlabel[Rshuffle]
        for i in range(nloops):
            if i<(nloops-1):
                st=i*batchsize
                last=((i+1)*batchsize)
            else:
                st=i*batchsize
                last=Ndata
            D=np.shape(W)[1]
            RFData=Data[st:last,]
            n=np.shape(RFData)[0]
            #
            if (RFoption==2):
                RFData=np.sqrt(2.0/D)*(np.cos(np.dot(RFData,W)+np.tile(b.T,(n,1))))
            else:
                RFData=np.sqrt(1/D)*np.concatenate((np.cos(np.dot(RFData,W)), np.sin(np.dot(RFData,W))), axis=1)
            #
            if (option ==1):
                zclf.partial_fit(RFData,label[st:last],classes=classnames)    
            else:
                #regression
                zclf.partial_fit(RFData,label[st:last])
        #print(zclf)
    return zclf  
    