# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 15:17:23 2016

@author: damodara
"""
#%% 
"""
function loads the data
MNIST, forestcover, digits, iris datasets are loaded from sklearn.datasets
"""
#%%
def adult_dataload():
    from scipy.sparse import csc_matrix
    import scipy.io as sio
    import numpy as np
    filepath='D:/PostDocWork/LSML/RandomFourierFeatures/Datasets/adult/adult/adult123.mat'
    adult=sio.loadmat(filepath)
    Dummy=adult['XTrain']
    TrainData=csc_matrix(Dummy,shape=Dummy.shape).toarray()
    TrainData=TrainData.T
    train_label=np.squeeze(adult['yTrain'])
    Dummy1=adult['XTest']
    TestData = csc_matrix(Dummy1,shape=Dummy1.shape).toarray()
    TestData=TestData.T
    test_label = np.squeeze(adult['yTest'])
    del Dummy, Dummy1
    return TrainData, train_label, TestData, test_label
    

    
def cifar10_dataload():
    import numpy as np
    import scipy.io as sio
    import os
    filename='D:/PostDocWork/LSML/RandomFourierFeatures/Datasets/cifar-10-matlab/cifar-10-batches-mat/CIFAR-10-TrainCombined-Test.mat'
    cifar=sio.loadmat(filename)
    TrainData=cifar['TrainData']
    train_label=cifar['Trainlabel']
    TestData=cifar['TestData']
    test_label=cifar['Testlabel']
    
    return TrainData, train_label, TestData, test_label


def cifar10_deepfeat_dataload():
    import numpy as np
    import scipy.io as sio
    import os
    filename='D:\PostDocWork\LSML\RandomFourierFeatures\Datasets\cifar10-alexnet\cifar10-alexnet-fc7.mat'
    cifar=sio.loadmat(filename)
    TrainData=cifar['TrainData']
    train_label=cifar['Trainlabel']
    TestData=cifar['TestData']
    test_label=cifar['Testlabel']
    
    return TrainData, train_label, TestData, test_label
    
def MNIST_dataload():
    from sklearn.datasets import fetch_mldata
    import numpy as np
    mnist = fetch_mldata('MNIST original')
    Data = mnist.data
    label = mnist.target
    return Data,label
    
def MNIST_official_split_dataload():
    import os
    import numpy as np
    pname ='D:\PostDocWork\LSML\RandomFourierFeatures\Datasets\mnist_official'
    fname ='MNIST_OfficialSplit.npz'
    mnist = np.load(os.path.join(pname,fname))
    TrainData = mnist['TrainData']
    train_label = mnist['Trainlabel']
    TestData = mnist['TestData']
    test_label = mnist['Testlabel']
    return TrainData, train_label, TestData, test_label
    
def forest_dataload():
    from sklearn.datasets import fetch_covtype
    import numpy as np
    forest = fetch_covtype()
    Data= forest['data']
    label = forest['target']
    return Data, label
    
def digits_dataload():
    from sklearn import datasets
    Digits=datasets.load_digits()
    Data=Digits.data/16.
    label=Digits.target
    return Data,label
    
def iris_dataload():
    from sklearn import datasets
    iris=datasets.load_iris()
    Data=iris.data
    label=iris.target
    return Data,label
    
def covtype_dataload():
    from scipy.sparse import csc_matrix
    import scipy.io as sio
    import numpy as np
    filepath='D:\PostDocWork\LSML\RandomFourierFeatures\Datasets\covtype\covtype\covtype.mat'
    adult=sio.loadmat(filepath)
    Dummy=adult['Data']
    Data=csc_matrix(Dummy,shape=Dummy.shape).toarray()
    label=np.squeeze(adult['label'])    
    return Data, label

def ijcnn1_dataload():
    from scipy.sparse import csc_matrix
    import scipy.io as sio
    import numpy as np
    filepath='D:\PostDocWork\LSML\RandomFourierFeatures\Datasets\ijcnn1\ijcnn1_combined.mat'
    adult=sio.loadmat(filepath)
    Dummy=adult['Xtrain']
    TrainData=csc_matrix(Dummy,shape=Dummy.shape).toarray()
    train_label=np.squeeze(adult['ytrain'])
    Dummy1=adult['Xtest']
    TestData = csc_matrix(Dummy1,shape=Dummy1.shape).toarray()
    test_label = np.squeeze(adult['ytest'])
    Dummy2=adult['Xval']
    ValData = csc_matrix(Dummy2,shape=Dummy2.shape).toarray()
    val_label = np.squeeze(adult['yval'])
    return TrainData, train_label, TestData, test_label, ValData, val_label
    
#def rcv1_dataload():

#%% Regression datasets
    
def census_dataload():
    from scipy.sparse import csc_matrix
    import scipy.io as sio
    import numpy as np
    filepath='D:/PostDocWork/LSML/RandomFourierFeatures/Datasets/census/census/census.mat'
    adult=sio.loadmat(filepath)
    Dummy=adult['Xtrain']
    TrainData=csc_matrix(Dummy,shape=Dummy.shape).toarray()
    TrainData=TrainData.T
    train_label=np.squeeze(adult['ytrain'])
    Dummy1=adult['Xtest']
    TestData = csc_matrix(Dummy1,shape=Dummy1.shape).toarray()
    TestData=TestData.T
    test_label = np.squeeze(adult['ytest'])
    del Dummy, Dummy1
    return TrainData, train_label, TestData, test_label

def cpu_dataload():
    from scipy.sparse import csc_matrix
    import scipy.io as sio
    import numpy as np
    filepath='D:/PostDocWork/LSML/RandomFourierFeatures/Datasets/cpu/cpu/cpu.mat'
    adult=sio.loadmat(filepath)
    Dummy=adult['Xtrain']
    TrainData=csc_matrix(Dummy,shape=Dummy.shape).toarray()
    TrainData=TrainData.T
    train_label=np.squeeze(adult['ytrain'])
    Dummy1=adult['Xtest']
    TestData = csc_matrix(Dummy1,shape=Dummy1.shape).toarray()
    TestData=TestData.T
    test_label = np.squeeze(adult['ytest'])
    del Dummy, Dummy1
    return TrainData, train_label, TestData, test_label
    
def YearPredictionMSD_dataload():
    from scipy.sparse import csc_matrix
    import scipy.io as sio
    import numpy as np
    filepath='D:\PostDocWork\LSML\RandomFourierFeatures\Datasets\YearMSD\YearPredictionMSD.mat'
    adult=sio.loadmat(filepath)
    Dummy=adult['Xtrain']
    TrainData=csc_matrix(Dummy,shape=Dummy.shape).toarray()
    train_label=np.squeeze(adult['ytrain'])
    Dummy1=adult['Xtest']
    TestData = csc_matrix(Dummy1,shape=Dummy1.shape).toarray()
    test_label = np.squeeze(adult['ytest'])
    return TrainData, train_label, TestData, test_label
    
def cpusmall_dataload():
    from scipy.sparse import csc_matrix
    import scipy.io as sio
    import numpy as np
    filepath='D:\PostDocWork\LSML\RandomFourierFeatures\Datasets\cpusmall\cpusmall.mat'
    adult=sio.loadmat(filepath)
    Dummy=adult['Data']
    Data=csc_matrix(Dummy,shape=Dummy.shape).toarray()
    label=np.squeeze(adult['label'])    
    return Data, label
    
def cadata_dataload():
    from scipy.sparse import csc_matrix
    import scipy.io as sio
    import numpy as np
    filepath='D:\PostDocWork\LSML\RandomFourierFeatures\Datasets\cadata\cadata.mat'
    adult=sio.loadmat(filepath)
    Dummy=adult['Data']
    Data=csc_matrix(Dummy,shape=Dummy.shape).toarray()
    label=np.squeeze(adult['label'])    
    return Data, label    