# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 14:06:51 2016

@author: damodara

In this file,
#%% 
gradForm1   -- Computes the gradient of the loss function based on formulation 1
     Input:
         Data -- data to be used in the loss function
         W   -- precomputed pseudo random fourier expansion (upto L-1 expansion)
         w   -- pseudo random fourier expansion to be computed (L expansion)
         sigma -- sigma parameter to compute the RBF kernel
         K1  -- precomputed RBF kernel with sigma
    Output:
         L -- is the computed gradient vector
#         
gradForm2  -- Computes the gradient of the loss function based on formulation 2
    The description is same as the gradForm1 (see above)
#%%
LossfunctionForm1  -- Computes the loss function based on formulation 1
LossfunctionForm2  -- Computes the loss function based on formulation 2
    Input:
        W  -- Pseudo Random Fourier Coefficient (or Random Fourier Coefficient) matrix
    Output:
        L  -- the computed value of the loss function
ApproxKernelLossfunction1 -- computes the approximated kernel based on formulation 1
ApproxKernelLossfunction2 -- computes the approximated kernel based on formulation 1
"""

#%% This function computes gradient for the loss function based formulation 1
    # gradient of L= 1/2N (k(x1,x2)-cos(W(x1-x2)))
def gradForm1(Data,W,w,sigma, K1=None):
    import numpy as np
    from sklearn.metrics import pairwise
    
    Nexp=np.shape(W)[1]
    Ndata,Nfeat=np.shape(Data)
    if K1 is None:
        K1=pairwise.rbf_kernel(Data,gamma=sigma**2/2)
        
    from kernelCentralization import kernelCentralization
    #KC=kernelCentralization(K1)
    
    nn = np.shape(W)[1]
    C=np.zeros((Nfeat,Ndata))
    B=np.zeros((Nfeat,Ndata))
    for i in range(Ndata):
        X=Data-np.tile(Data[i,],(Ndata,1))
        K=K1[i,:]
        K=K[np.newaxis,]
 #       K=KC[i,:]
#        K=K[np.newaxis,]
        tmp=np.sin(np.dot(X,w))
        B[:,i]=np.dot((X.T), np.diag(np.dot(tmp,K)))
        tmp2=np.dot(tmp,np.ones((1,Nexp)))
        tmp3=(np.cos(np.dot(X,W)))*tmp2
        C[:,i]=np.squeeze(np.dot((X.T),np.sum(tmp3,axis=1)))

    L=(1/Ndata**2)*((np.sum(B,axis=1)) - (1/Nexp)*np.sum(C,axis=1))  
   
    L = L[:,np.newaxis]
    return L

#%% This function computes the value of the Loss function of formulation 1 
def LossfunctionForm1(Data,W,gamma,K=None):
    import numpy as np
    from sklearn.metrics import pairwise
    Nexp=np.shape(W)[1]
    Ndata,Nfeat=np.shape(Data)
    Np, Nexp=np.shape(W)
#    if (Np != Nfeat):
#        W = np.delete(W, Np-1,0)
#        Data = np.concatenate((Data, np.ones((Ndata,1), dtype= np.float64)), axis =1)
    W = np.delete(W, Np-1,0)    
    # if kernel is not provided
    if K is None:
        K=pairwise.rbf_kernel(Data,gamma=gamma)
    # explicit feature computation
    Dummy = np.concatenate((np.cos(np.dot(Data,W)), np.sin(np.dot(Data,W))), axis=1)
    AK = (1.0/Nexp)*np.dot(Dummy, Dummy.T)
    L = np.linalg.norm(K-AK)/np.linalg.norm(K)
    L = np.sum((K-AK)**2)/(Ndata**2)
    # normal implementation as loss function
#    AK=np.zeros((Ndata,Ndata))
#    for i in range(Ndata):
#        X=Data-np.tile(Data[i,],(Ndata,1))
#        B[i]=np.sum(np.cos(np.dot(X,W)))
#        AK[i,:] = np.sum(np.cos(np.dot(X,W)), axis=1)
#        
#    #L=(0.5/Ndata**2)*(A-(1/Nexp)*np.sum(B))**2
#    L = np.linalg.norm(K-(1/Nexp)*AK)/np.linalg.norm(K)
    return L

#%%  This function computes for the gradient of the formulation 2 
def gradForm2(Data,W,w,sigma, K1=None):
    # gradient of L= 1/2N (k(x1,x2)-cos(Wx1)cos(Wx2))
    import numpy as np
    from sklearn.metrics import pairwise
    
    Nexp = np.shape(W)[1]
    Ndata,Nfeat = np.shape(Data)
    
    if K1 is None:
        K1 = pairwise.rbf_kernel(Data,gamma=sigma**2/2)
        
    C=np.zeros((Nfeat,Ndata))
    B=np.zeros((Nfeat,Ndata))
    for i in range(Ndata):
        X = np.tile(Data[i,], (Ndata,1))
        K = K1[i,:].copy()
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
        
    L = (1.0/Ndata**2)*(np.sum(C,axis=1) - (2.0/Nexp)*np.sum(B, axis=1))
    L = L[:,np.newaxis]
    return L 



#%%  This function computes loss function value of formulation 2
def LossfunctionForm2(NData,W,gamma,K=None):
    import numpy as np
    from sklearn.metrics import pairwise
    
    Data = NData.copy()
    Np, Nexp=np.shape(W)
    Ndata,Nfeat=np.shape(Data)
    
    if (Np != Nfeat):
        Data = np.concatenate((Data, np.ones((Ndata,1), dtype= np.float64)), axis =1)
    # if kernel is not provided
    if K is None:
        K=pairwise.rbf_kernel(Data,gamma=gamma)
    Phi = np.cos(np.dot(Data,W))
    AK= (2.0/Nexp)*np.dot(Phi, Phi.T)
    #L = np.linalg.norm(K-AK)/np.linalg.norm(K)
    L = np.sum((K-AK)**2)/(Ndata**2)
    
    ## normal implementation as the loss function
    #A=np.sum(K)
    #B=np.zeros((Ndata,1))
#    for i in range(Ndata):
#        X=Data-np.tile(Data[i,],(Ndata,1))
#        B[i]=np.sum(np.cos(np.dot(X,W)))
        
    #L=(0.5/Ndata**2)*(A-(1/Nexp)*np.sum(B))**2
    
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
#%%    
def ApproxKernelLossfunction2(NData,W,sigma,K=None):
    import numpy as np
    from sklearn.metrics import pairwise
    Data = NData.copy()
    Nexp=np.shape(W)[1]
    Ndata,Nfeat=np.shape(Data)
    # if kernel is not provided
    if K is None:
        K=pairwise.rbf_kernel(Data,gamma=sigma**2/2.0)
    Phi =  np.cos(np.dot(Data,W))
    AK= (2.0/Nexp)*np.dot(Phi, Phi.T)
    L = np.linalg.norm(K-AK)/np.linalg.norm(K)
    return L, AK
     
            
    
    