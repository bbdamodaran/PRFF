# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 16:19:05 2016

@author: damodara
"""

def grad(Data,W,w,sigma):
    import numpy as np
    from sklearn.metrics import pairwise
    Nexp=np.shape(W)[1]
    Ndata,Nfeat=np.shape(Data)
    K1=pairwise.rbf_kernel(Data,gamma=sigma**2/2)
    from kernelCentralization import kernelCentralization
    #KC=kernelCentralization(K)
    #A=np.sum(K)
    nn = np.shape(W)[1]

    C=np.zeros((Nfeat,Ndata))
    B=np.zeros((Nfeat,Ndata))
    for i in range(Ndata):
        X=Data-np.tile(Data[i,],(Ndata,1))
       # K=pairwise.rbf_kernel(Data[i,],Data,gamma=sigma)
        K=K1[i,:]
        K=K[np.newaxis,]
 #       K=KC[i,:]
#        K=K[np.newaxis,]
        tmp=np.sin(np.dot(X,w))
        B[:,i]=np.dot((X.T), np.diag(np.dot(tmp,K)))
        tmp2=np.dot(tmp,np.ones((1,Nexp)))
        tmp3=(np.cos(np.dot(X,W)))*tmp2
        C[:,i]=np.squeeze(np.dot((X.T),np.sum(tmp3,axis=1)))
        #B[i]=np.sum(np.cos(np.dot(X,W)))+np.sum(np.cos(np.dot(X,w)))
        #tmp=np.sin(np.dot(X,w))
        #C[:,i]=np.squeeze(np.dot(X.T,tmp))
    L=(1/Ndata**2)*((np.sum(B,axis=1)) - (1/Nexp)*np.sum(C,axis=1))
    
    
   #L = (1/Ndata**2)*(A-(1/(Nexp+1))*np.sum(B))*np.sum(C,axis=1)
    L = L[:,np.newaxis]
    return(L)
#%%  ######################################################################
def Lossfunction2(NData,W,gamma,K=None, w=None):
    import numpy as np
    from sklearn.metrics import pairwise
    
    Data = NData.copy()
    if w is None:
        W=W
    else:
        W = np.concatenate((W,w), axis =1)
    
    
    Np, Nexp=np.shape(W)
    Ndata,Nfeat=np.shape(Data)
    
    if (Np != Nfeat):
        Data = np.concatenate((Data, np.ones((Ndata,1), dtype= np.float64)), axis =1)
    # if kernel is not provided
    if K is None:
        K=pairwise.rbf_kernel(Data,gamma=gamma)
    #
    #K=pairwise.rbf_kernel(Data,gamma=sigma)
    #A=np.sum(K)
    #B=np.zeros((Ndata,1))
    
    AK= (2.0/Nexp)*np.dot(np.cos(np.dot(Data,W)), np.cos(np.dot(Data,W)).T)
    #AK= (1.0/Nexp)*np.dot(np.cos(np.dot(Data,W)), np.cos(np.dot(Data,W)).T)
#    L = (np.linalg.norm(K-AK)/np.linalg.norm(K))
    L = np.sum((K-AK)**2)/(Ndata**2)
#    for i in range(Ndata):
#        X=Data-np.tile(Data[i,],(Ndata,1))
#        B[i]=np.sum(np.cos(np.dot(X,W)))
        
    #L=(0.5/Ndata**2)*(A-(1/Nexp)*np.sum(B))**2
    return L
 #%%  ######################################################################   
def Lossfunction1(Data,W,sigma,K=None):
    import numpy as np
    from sklearn.metrics import pairwise
    Nexp=np.shape(W)[1]
    Ndata,Nfeat=np.shape(Data)
    # if kernel is not provided
    if K is None:
        K=pairwise.rbf_kernel(Data,gamma=sigma**2/2)
    #
    #A=np.sum(K)
   # B=np.zeros((Ndata,1))
    
    Dummy = np.concatenate((np.cos(np.dot(Data,W)), np.sin(np.dot(Data,W))), axis=1)
    AK = (2.0/Nexp)*np.dot(Dummy, Dummy.T)
    L = np.linalg.norm(K-AK)/np.linalg.norm(K)
    # normal implementation as loss function
#    AK=np.zeros((Ndata,Ndata))
#    for i in range(Ndata):
#        X=Data-np.tile(Data[i,],(Ndata,1))
#        #B[i]=np.sum(np.cos(np.dot(X,W)))
#        AK[i,:] = np.sum(np.cos(np.dot(X,W)), axis=1)
#        
#    #L=(0.5/Ndata**2)*(A-(1/Nexp)*np.sum(B))**2
#    L = np.linalg.norm(K-(1/Nexp)*AK)/np.linalg.norm(K)
    return L
    
  ##############################################################################
    #%%  ######################################################################
def gradMat(Data,W,w):
    import numpy as np
    from sklearn.metrics import pairwise
    Nexp=np.shape(W)[1]
    Ndata,Nfeat=np.shape(Data)
    K=pairwise.rbf_kernel(Data,gamma=0.02)
    A=np.sum(K)
    X=np.sqrt(2/Nexp)*(np.cos(np.dot(Data,W)))
    B=np.dot(X,X.T)
    C=np.zeros((Nfeat,Ndata))
    for i in range(Ndata):
        tmp1=np.sin(np.dot(Data,w))
        F1=np.cos(np.dot(Data[i,],w))*(np.dot(Data.T,tmp1))
        tmp2=np.cos(np.dot(Data,w))
        F2=np.sin(np.dot(Data[i,],w))*(np.dot(Data.T,tmp2))
        C[:,i]=np.squeeze((F1+F2))
    
    L = (1/Ndata)*(A-(1/Nexp)*np.sum(B))*np.sum(C,axis=1)
    L = L[:,np.newaxis]
    return(L)
    
#%%  ######################################################################    
def gradcoswx1coswx2(NData,W,w,sigma, K1=None):
    # gradient of L= 1/2N (k(x1,x2)-cos(Wx1)cos(Wx2))
    import numpy as np
    from sklearn.metrics import pairwise
     
    Data = NData.copy()
    Nexp = np.shape(W)[1]
    Nexp = Nexp+1
    Ndata,Nfeat = np.shape(Data)
    if K1 is None:
        K1 = pairwise.rbf_kernel(Data,gamma=sigma**2/2)
    
    C=np.zeros((Nfeat,Ndata))
    B=np.zeros((Nfeat,Ndata))
    for i in range(Ndata):
        tmpdata = Data[i,]
        X = np.tile(tmpdata, (Ndata,1))
        K = K1[i,:]
        K = K[:,np.newaxis]
        #
        c1 = np.outer(np.cos(np.dot(Data, w))*K, np.sin(np.dot(Data[i,], w)))
        c2 = np.outer(np.sin(np.dot(Data, w))*K, np.cos(np.dot(Data[i,], w)))
        C[:,i] = np.squeeze((np.dot((X.T), c1)+np.dot((Data.T), c2)))
        #
        
        AK = np.dot(np.cos(np.dot(Data[i,], W)), np.cos(np.dot(Data,W)).T)
        AK = AK[:,np.newaxis]
        #
        b1 = np.outer(np.cos(np.dot(Data, w))*AK, np.sin(np.dot(Data[i,], w))) 
        b2 = np.outer(np.sin(np.dot(Data, w))*AK, np.cos(np.dot(Data[i,], w)))
        B[:,i] = np.squeeze((np.dot((X.T), b1)+np.dot((Data.T), b2)))
        #
        
#    L =(2.0/Nexp)*(1.0/Ndata**2)*(np.sum(C,axis=1) - (2.0/Nexp)*np.sum(B, axis=1))
    L =(1.0/Ndata**2)*(np.sum(C,axis=1) - (2.0/Nexp)*np.sum(B, axis=1))
    #L = (1/Ndata**2)*(np.sum(C,axis=1) - (1/Nexp)*np.sum(B, axis=1))
    L = L[:,np.newaxis]
    return L
#%%  ######################################################################    
def grad1coswx1coswx2(Data,W,w,sigma):
    import numpy as np
    from sklearn.metrics import pairwise
    Nexp = np.shape(W)[1]
    Ndata,Nfeat = np.shape(Data)
    K1 = pairwise.rbf_kernel(Data,gamma=sigma**2/2)
     
    B=np.zeros((Nfeat,Ndata))
    L=np.zeros((Nfeat,1))
    c1 = np.zeros((Nfeat,1))
    c2 = np.zeros((Nfeat,1))
    for i in range(Ndata):
        for j in range(Ndata):
            k = K1[i,j]
            b = (2/Nexp)*np.dot(np.cos(np.dot(Data[i,], W)), np.cos(np.dot(Data[j,], W)).T)
            c1=  np.cos(np.dot(Data[i,], w))*np.sin(np.dot(Data[j,], w))*np.cos(Data[j,].T)
            c1 = c1[:,np.newaxis]
            c2 = np.sin(np.dot(Data[i,], w))*np.cos(np.dot(Data[j,], w))*np.cos(Data[i,].T)
            c2 = c2[:,np.newaxis]
            
            L = L + (1/Ndata)*((k-b)*((c1+c2)))
        
    
    return L
#%%  ######################################################################    
def ApproxKernelLossfunction1(Data,W,sigma,K=None):
    import numpy as np
    from sklearn.metrics import pairwise
    Nexp=np.shape(W)[1]
    Ndata,Nfeat=np.shape(Data)
    # if kernel is not provided
    if K is None:
        K=pairwise.rbf_kernel(Data,gamma=sigma**2/2.0)
    #
        
    Dummy = np.concatenate((np.cos(np.dot(Data,W)), np.sin(np.dot(Data,W))), axis=1)
    AK = (2.0/Nexp)*np.dot(Dummy, Dummy.T)
    L = np.linalg.norm(K-AK)/np.linalg.norm(K)
    # normal implementation as loss function
#    AK=np.zeros((Ndata,Ndata))
#    for i in range(Ndata):
#        X=Data-np.tile(Data[i,],(Ndata,1))
#        #B[i]=np.sum(np.cos(np.dot(X,W)))
#        AK[i,:] = np.sum(np.cos(np.dot(X,W)), axis=1)
#        
#    #L=(0.5/Ndata**2)*(A-(1/Nexp)*np.sum(B))**2
#    AK = (1/Nexp)*AK
#    L = np.linalg.norm(K-AK)/np.linalg.norm(K)
    return L, AK
    
def ApproxKernelLossfunction2(Data,W,sigma,K=None):
    import numpy as np
    from sklearn.metrics import pairwise
    Nexp=np.shape(W)[1]
    Ndata,Nfeat=np.shape(Data)
    # if kernel is not provided
    if K is None:
        K=pairwise.rbf_kernel(Data,gamma=sigma**2/2.0)
    Phi =np.cos(np.dot(Data,W))  
    AK= (2.0/Nexp)*np.dot(Phi, Phi.T)
    #AK= (1.0/Nexp)*np.dot(np.cos(np.dot(Data,W)), np.cos(np.dot(Data,W)).T)
    L = np.linalg.norm(K-AK)/np.linalg.norm(K)
    return L, AK
     
  
#%% stochastic gradient desecent

def StochGrad(Data,W,w,sigma, K1=None):
    # gradient of L= 1/2N (k(x1,x2)-cos(Wx1)cos(Wx2))
    import numpy as np
    from sklearn.metrics import pairwise
    
    nepo = 5
    batchsize = 16
    Ndata, Nfeat = np.shape(Data)
    nbatches=int(np.round(Ndata/batchsize))
    DummyData = copy.copy(Data)
    
    it = 1
    for j in range(nepo):
        np.random.shuffle(DummyData)
        
        for i in range(nbatches):
        
            if i<(nbatches-1):
                st=i*batchsize
                last=((i+1)*batchsize)
            else:
                st=i*batchsize
                last=Ntrain
            SubsetData=DummyData[st:last,]
            from sklearn.metrics import pairwise
            K=pairwise.rbf_kernel(SubsetData,gamma=sigma**2/2)
            
            G = gradcoswx1coswx2(SubsetData, W, w, sigma, K)
            alpha = alphaN /(1+(alphaN*0.1*it))
            w = w - alpha*(0.0*w +G)
            it = it+1
            

#%% Gradient Compuation WithoutLoop

def gradcoswx1coswx2WithOutLoop(Data,W,w,gamma, K1=None):
    '''
    Data -- is the samples from the mini-batch
    W  -- learned coefficient vector upto M-1
    w  -- initial vector (or updated vector by gradient descent) of w_M to be learned
    sigma = sqrt(2*gamma)
    '''
    
    import numpy
    from sklearn.metrics import pairwise
    
    n_components = numpy.shape(W)[1]
    minibatch_size, Nfeat = numpy.shape(Data)
    
    if K1 is None:
        K1 = pairwise.rbf_kernel(Data, gamma=gamma)
        
    Phi =   numpy.cos(numpy.dot(Data,W)) 
    AK  = (2.0/n_components)*numpy.dot(Phi, Phi.T)
    diff_mat = K1-AK
    
    wx_b = numpy.dot(Data, w)  
    
    sin_wx = numpy.sin(wx_b).reshape((-1, 1))
    cos_wx = numpy.cos(wx_b).reshape((-1, 1))

    sin_cos = numpy.dot(sin_wx, cos_wx.T) * 2 / (n_components * minibatch_size ** 2)
    
    diff_sin_cos = numpy.diag(numpy.dot(diff_mat, 2. * sin_cos.T)).reshape((-1, 1))
    dl_dw = numpy.sum(diff_sin_cos * Data, axis=0).reshape((-1,1))
    
    return dl_dw
        
    
    