# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 13:00:31 2016

@author: damodara
This file computes the hyperparameters for the SGD classifier  (batch size used =64)

minibatchSGDCV -- this function computes the hyperparameters of SGD classifier with respect to original features
    Input:
        Data, label  --Validation data and corresponding its label
        ncv    -- number of folds in CV
    Output:
        parameter  - computed best parameters
        batch      - best parameter for each batch
        Meanscore  -- meanaccuracy of each parameter
        
minibatchRFSGDCV --this function computes the hyperparameters of SGD classifier with respect to PRFF (RFF) features
    Input:
        Data, label  --Validation data and corresponding its label
        ncv    -- number of folds in CV
        W   -- PRFF (RFF) coefficient matrix
        b  -- bias term (as it is included in W, so it is of zeros)
        RFoption -- 1 for formulation 1 and 2 for formulation 2
    Output:
        Same as above

"""

def minibatchSGDCV(NData,Nlabel,ncv=3, option=1,loss="log"):
    
    import numpy as np
    from sklearn.cross_validation import StratifiedKFold, KFold
    from minibatchSGD import minibatchSGD
    
    Data = NData.copy()
    label = Nlabel.copy()
    No_data = np.shape(Data)[0]
    if (option==1):
        skf = StratifiedKFold(label, n_folds=ncv, random_state=0)
    else:
        skf = KFold(No_data,n_folds=ncv, random_state=0)

    #batchrange = 2.0**(np.asarray(np.arange(4,10,1)))
    batchrange = np.array([64])
    num_batch = len(batchrange)
#    C = np.concatenate((2.0**(np.arange(-15,2,2)), 2.0**(np.arange(4,11,1))))
    C =2.0**(np.arange(-15,2,2))
#    eta = np.asarray([0.1, 0.01, 0.001])
    eta = np.asarray([0.01]) #optimal learning rate is used
    alpharange, raterange = np.meshgrid(C,eta)
    #%%
    Meanscore =[0 for i in range(len(batchrange))]
    BestBatchParameters = np.zeros((num_batch,4))
    #%%
    for nbatch in range(len(batchrange)):  
        batchsize = batchrange[nbatch]
        Meanscore[nbatch]=np.zeros((np.size(alpharange), 5))
        for i in range(np.size(alpharange)):
            #
            score=[]
            it=0
            for train_index, test_index in skf:
                TrainData, TestData = Data[train_index], Data[test_index]
                trainlabel, testlabel = label[train_index], label[test_index]
                clf = minibatchSGD(TrainData, trainlabel, option, loss=loss,batchsize=batchsize, 
                                   alpha=alpharange.flat[i],eta0=raterange.flat[i])
                score.append(clf.score(TestData,testlabel))
                #print(score[it])
                it=it+1
            # cv loop end
            Meanscore[nbatch][i,] = batchsize, np.mean(score), np.std(score), alpharange.flat[i], raterange.flat[i]  
            #print(np.mean(score))
        # end loop of the parameters
        maxind, acc = np.argmax(Meanscore[nbatch][0:,1]), np.amax(Meanscore[nbatch][0:,1])
        BestBatchParameters[nbatch,] = batchsize, acc, alpharange.flat[maxind], raterange.flat[maxind]
    # end loop for the batchsize
    ind = np.argmax(BestBatchParameters[0:,1]) 
    parameter ={'batchsize': BestBatchParameters[ind,0], 'alpha': BestBatchParameters[ind,2],
                'eta0': BestBatchParameters[ind,3], 'cvaccuracy': BestBatchParameters[ind,1]}
    # paremeter estimation completed
    return parameter, BestBatchParameters, Meanscore


def minibatchRFSGDCV(NData,Nlabel,ncv, W,b,option=1, RFoption=2):
    
    import numpy as np
    from sklearn.cross_validation import StratifiedKFold, KFold
    from minibatchSGD import RFminibatchSGD
    Data = NData.copy()
    label = Nlabel.copy()
    No_data = np.shape(Data)[0]
    if (option==1):
        skf = StratifiedKFold(label, n_folds=ncv, random_state=0)
    else:       
        skf = KFold(No_data, n_folds=ncv, random_state=0)

    #batchrange = 2.0**(np.asarray(np.arange(4,10,1)))
    batchrange = np.array([64])
    num_batch = len(batchrange)
    C = 2.0**(np.arange(-15,2,2))
    eta = np.asarray([0.1, 0.01, 0.001])
    alpharange, raterange = np.meshgrid(C,eta)
    #%%
    Meanscore =[0 for i in range(len(batchrange))]
    BestBatchParameters = np.zeros((num_batch,4))
    #%%
    D = np.shape(W)[1]
    for nbatch in range(len(batchrange)):  
        batchsize = batchrange[nbatch]
        Meanscore[nbatch]=np.zeros((np.size(alpharange), 5))
        for i in range(np.size(alpharange)):
            #
            score=[]
            it=0
            for train_index, test_index in skf:
                TrainData, TestData = Data[train_index], Data[test_index]
                trainlabel, testlabel = label[train_index], label[test_index]
                clf = RFminibatchSGD(TrainData,trainlabel,W,b,option, batchsize=batchsize, 
                                   alpha=alpharange.flat[i], eta0=raterange.flat[i], RFoption=RFoption)
                Ntest = np.shape(TestData)[0]
                #
                if (RFoption==2):
                    TestData = np.sqrt(2.0/D)*(np.cos(np.dot(TestData,W)+np.tile(b.T,(Ntest,1))))
                else:
                    TestData = np.sqrt(1/D)*np.concatenate((np.cos(np.dot(TestData,W)), np.sin(np.dot(TestData,W))), axis=1)
                 
                score.append(clf.score(TestData,testlabel))
                #print(score[it])
                it=it+1
            # cv loop end
            Meanscore[nbatch][i,] = batchsize, np.mean(score), np.std(score), alpharange.flat[i], raterange.flat[i]  
            #print(np.mean(score), i)
        # end loop of the parameters
        maxind, acc = np.argmax(Meanscore[nbatch][0:,1]), np.amax(Meanscore[nbatch][0:,1])
        BestBatchParameters[nbatch,] = batchsize, acc, alpharange.flat[maxind], raterange.flat[maxind]
    # end loop for the batchsize
    ind = np.argmax(BestBatchParameters[0:,1]) 
    parameter ={'batchsize': BestBatchParameters[ind,0], 'alpha': BestBatchParameters[ind,2],
                'eta0': BestBatchParameters[ind,3], 'cvaccuracy': BestBatchParameters[ind,1]}
    # paremeter estimation completed
    return parameter, BestBatchParameters, Meanscore       