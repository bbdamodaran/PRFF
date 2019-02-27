# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 14:22:25 2016

@author: damodara
minimization by 
Gradient descent method
"""

def Gradientdescent(SubsetData, W2,wnn,alphaN, RegW,gamma, K=None, update_b=True, philippe_update=False):
    import numpy as np
    from grad import gradcoswx1coswx2, Lossfunction2,gradcoswx1coswx2WithOutLoop
    import time
    import copy
    err2=2
    tol =1e-10
    WTMP = wnn
    woldnn = wnn
    gradnorm=[]
    loss =[]
    intloss=[]
    display = False
    it=0
    max_iter = 150
    Ncheck =50
    beta = np.zeros((max_iter,1))
    sigma_inv2 = 2*gamma
    intloss = np.zeros((max_iter,1))
    loss = np.zeros((max_iter,1))
    while(it<max_iter):
#        randomNoise = np.random.normal(0, 1,1)
        randomNoise = 0.0
        r,s = np.shape(W2)
        if (s==1):
#            intloss.append(Lossfunction2(SubsetData,wnn,gamma,K))
            intloss[it] =Lossfunction2(SubsetData,wnn,gamma,K)
        else:
#            intloss.append(Lossfunction2(SubsetData,W2,gamma,K, wnn))
            intloss[it] =Lossfunction2(SubsetData,W2,gamma,K, wnn)
        if update_b:
           lw = len(wnn)
           intloss[it] = intloss[it]+RegW*np.sum(np.square(wnn[0:lw-1])) 
        else:
           intloss[it] = intloss[it]+RegW*np.sum(np.square(wnn))
        if display:
           print('Initial Loss =', intloss[it])
        
        if (it==0):
            old_loss = intloss[0]
        if (it == Ncheck):
            old_loss = intloss[0]
            min_index = np.argmin(loss[0:Ncheck])
            alphaN = beta[np.abs(min_index-1)]
            if display:
               print('min_index', min_index)
               print('alp', alphaN)
               print('min_loss', loss[min_index])
        if (it<Ncheck):
            old_loss =0.0
        #np.random.shuffle(SubsetData)
        GNew = gradcoswx1coswx2WithOutLoop(SubsetData,W2,wnn,gamma, K)
        GNew1 = GNew+randomNoise
        GNew = (RegW * wnn)+GNew1
        if update_b:
           GNew[-1] = GNew1[-1].copy()                
             
        #gradnorm.append(np.linalg.norm(GNew))
#            alpha = alphaN*np.sqrt(100/(100+ita))
        if it>=Ncheck:     
           alpha = alphaN/(1+(alphaN*RegW*it))

        else:
           alpha = alphaN
                  
        wnew = wnn - (alpha)*(GNew)
        if philippe_update:
            if update_b:
               wnew[0:len(wnew)-1] = wnew[0:len(wnew)-1]*(sigma_inv2/np.std(wnew[0:len(wnew)-1]))
               wnew[len(wnew)-1] = wnew[len(wnew)-1]/(2*np.pi)
            else:
               wnew = wnew*(sigma_inv2/np.std(wnew))
           
        WTMP = np.concatenate((WTMP, wnn), axis =1)
        iternorm2 = np.linalg.norm(wnn)
        if display:
           print('Learning rate is=', alpha)         
        
        r,s = np.shape(W2)
        if (s==1):
#            loss.append(Lossfunction2(SubsetData,wnew,gamma,K))
            loss[it]=Lossfunction2(SubsetData,wnew,gamma,K)
        else:
#            loss.append(Lossfunction2(SubsetData,W2,gamma,K, wnew))
            loss[it]= Lossfunction2(SubsetData,W2,gamma,K, wnew)
        if update_b:
           lw = len(wnew)
           loss[it] = loss[it]+RegW*np.sum(np.square(wnew[0:lw-1])) 
        else:
           loss[it] = loss[it]+RegW*np.sum(np.square(wnew))
           
        new_loss =loss[it]
        if display:
           print('Loss is' , new_loss)
           time.sleep(0.7)
           print('Old Loss is', old_loss)
        if new_loss<old_loss:
           wnn = np.copy(wnew)
           old_loss = np.copy(new_loss)
           if display:
              print('update')
        else:
           alphaN = alphaN*0.7  
           beta[it]= np.copy(alphaN) 
        
        err2=np.linalg.norm(woldnn-wnn)
#            print('The error2 is', err2)
        if (err2<=tol) and (it>=75):
            if display:
               print('done')
            break
        if (it>250):
#                print('break')
            break
        woldnn=np.copy(wnn)
        it=it+1
    if display:
       print('over')
       print('==========================')
    return wnn, loss
    
def NaturalGradientdescent(SubsetData, W2,wnn, alphaN, RegW,sigma, K=None):
    import numpy as np
    from grad import gradcoswx1coswx2, Lossfunction2
    err2=2
    tol =1e-5
    WTMP = wnn
    woldnn = wnn
    gradnorm=[]
    loss =[]
    it=0
    while(err2>tol):
    
        #np.random.shuffle(SubsetData)
        r,s = np.shape(W2)
        if (s==1):
            loss.append(Lossfunction2(SubsetData,wnn,sigma,K))
        else:
            loss.append(Lossfunction2(SubsetData,W2,sigma,K, wnn)) 
        print('Loss is', loss[it])
    
        GNew = gradcoswx1coswx2(SubsetData,W2,wnn,sigma, K)        
        FM = np.dot(GNew, GNew.T)+0.001
        alpha = alphaN /(1+(alphaN*RegW*it)) 
        wnn = wnn- (alpha)* (np.dot(np.linalg.inv(FM), GNew))
        WTMP = np.concatenate((WTMP, wnn), axis =1)
        iternorm2 = np.linalg.norm(wnn)
        
        
        
        err2=np.linalg.norm(woldnn-wnn)
#            print('The error2 is', err2)
        if (err2<=tol):
#                print('done')
            break
        elif (it>250):
#                print('break')
            break
        woldnn=wnn
        it=it+1
    
    return wnn, loss
    
def BoldDrive(SubsetData, W2,wnn, alphaN, RegW,sigma, K=None):
    import numpy as np
    from grad import gradcoswx1coswx2, Lossfunction2
    err2=2
    tol =1e-5
    WTMP = wnn
    woldnn = wnn
    gradnorm=[]
    loss =[]
    it=0
    while(err2>tol):
    
        #np.random.shuffle(SubsetData)
        r,s = np.shape(W2)
        if (s==1):
            loss.append(Lossfunction2(SubsetData,wnn,sigma,K))
        else:
            loss.append(Lossfunction2(SubsetData,W2,sigma,K, wnn)) 
        print('Loss is', loss[it])
        
        GNew = gradcoswx1coswx2(SubsetData,W2,wnn,sigma, K)
        gradnorm.append(np.linalg.norm(GNew))
#            alpha = alphaN*np.sqrt(100/(100+ita))
        alpha = alphaN /(1+(alphaN*RegW*it))            
        wnn = wnn - (alpha)*(RegW*wnn+GNew)
        WTMP = np.concatenate((WTMP, wnn), axis =1)
        iternorm2 = np.linalg.norm(wnn)
       
        err2=np.linalg.norm(woldnn-wnn)
#            print('The error2 is', err2)
        if (err2<=tol):
#                print('done')
            break
        elif (it>250):
#                print('break')
            break
        woldnn=wnn
        it=it+1
    
    return wnn, loss
    
def Adam(SubsetData,W2, wnn, alpha, RegW, sigma, K=None):
    import numpy as np
    from grad import gradcoswx1coswx2, Lossfunction2
    import copy
    
    if K is None:
        from sklearn.metrics import pairwise
        K=pairwise.rbf_kernel(SubsetData,gamma=sigma**2/2)
        
    Nfeat = np.shape(W2)[0]
    err2=2
    tol =1e-5
    WTMP = copy.copy(wnn)
    woldnn = copy.copy(wnn)
    gradnorm=[]
    loss =[]
    it=0
    m = np.zeros((Nfeat,1))
    v = np.zeros((Nfeat,1))
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    alpha=0.1
    while(err2>tol):
       
        #np.random.shuffle(SubsetData)
        it = it+1
#        print('iteration', it)
        GNew = gradcoswx1coswx2(SubsetData,W2,wnn,sigma, K)

        m = b1*m + (1-b1)*GNew
        v = b2*v + (1-b2)*(GNew**2)
        alphaN = alpha*(np.sqrt((1-b2**it)/(1-b1**it)))
 
#        alpha = alphaN /(1+(alphaN*RegW*it))  
        wnn = wnn - (alphaN)* (RegW*wnn+ (m/(np.sqrt(v)+eps)))          
#        wnn = wnn - (alpha)*(0.1*wnn+GNew)
#        wnn = wnn - (alpha)*(0.0*wnn+NG)
        WTMP = np.concatenate((WTMP, wnn), axis =1)
        iternorm2 = np.linalg.norm(wnn)
        r,s = np.shape(W2)
        if (s==1):
            loss.append(Lossfunction2(SubsetData,wnn,sigma,K))
        else:
            loss.append(Lossfunction2(SubsetData,W2,sigma,K, wnn)) 
        print('Loss is', loss[it-1])
        err2=np.linalg.norm(woldnn-wnn)
#            print('The error2 is', err2)
        if (err2<=tol):
#                print('done')
            break
        elif (it>250):
#                print('break')
            break
        woldnn=wnn
        
    
    return wnn, loss  
    
def Adagrad(SubsetData,W2, wnn, alpha, RegW, sigma, K=None):
    import numpy as np
    from grad import gradcoswx1coswx2, Lossfunction2
    import copy
    
    if K is None:
        from sklearn.metrics import pairwise
        K=pairwise.rbf_kernel(SubsetData,gamma=sigma**2/2)
        
    Nfeat = np.shape(W2)[0]
    err2=2
    tol =1e-5
    WTMP = copy.copy(wnn)
    woldnn = copy.copy(wnn)
    gradnorm=[]
    loss =[]
    it=0
    CumGrad = np.zeros((Nfeat,1))
    eps = 1e-5
    alpha=0.1
    
    while(err2>tol):
       
        #np.random.shuffle(SubsetData)
        it = it+1
#        print('iteration', it)
        GNew = gradcoswx1coswx2(SubsetData,W2,wnn,sigma, K)
        gradnorm.append(np.linalg.norm(GNew))
        CumGrad = CumGrad + GNew**2
        alphaN = alpha/ (eps+np.sqrt(CumGrad))

        wnn = wnn - (alphaN)* (RegW*wnn+GNew )         

        WTMP = np.concatenate((WTMP, wnn), axis =1)
        iternorm2 = np.linalg.norm(wnn)
        r,s = np.shape(W2)
        if (s==1):
            loss.append(Lossfunction2(SubsetData,wnn,sigma,K))
        else:
            loss.append(Lossfunction2(SubsetData,W2,sigma,K, wnn))            
        print('Loss is', loss[it-1])   
        
        err2=np.linalg.norm(woldnn-wnn)
#            print('The error2 is', err2)
        if (err2<=tol):
#                print('done')
            break
        elif (it>250):
#                print('break')
            break
        woldnn=wnn
        
#    minloss = np.argmin()
    return wnn, gradnorm, loss  
    
def Rprop(SubsetData, W2,wnn, alphaN, RegW, sigma, K=None):
    import numpy as np
    from grad import gradcoswx1coswx2, Lossfunction2
    from math import copysign
    import copy
    
    if K is None:
        from sklearn.metrics import pairwise
        K=pairwise.rbf_kernel(SubsetData,gamma=sigma**2/2)    
    
    err2=2
    tol =1e-5
    WTMP = wnn
    woldnn = wnn
    gradnorm=[]
    loss =[]
    it=0
    deltaold = 0.01
    delta = deltaold
    deltaMin = np.tile(0.0, (np.size(wnn),1))
    deltaMax = np.tile(0.9, (np.size(wnn),1))
    nplus = 1.2
    nminus = 0.5
    GOld = 0.0
    deltaW = 0.0
    alphaN = 0.1
    
    while(err2>tol):
    
        #np.random.shuffle(SubsetData)
        if (it==51):
            index = np.argmin(loss)
            wnn = WTMP[:, index]
            wnn = wnn[:, np.newaxis]
            deltaold = alpha
            print('Index', index)
    
#       
        r, s = np.shape(W2)
        if (s==1):
            loss.append(Lossfunction2(SubsetData,wnn,sigma,K))
        else:
            loss.append(Lossfunction2(SubsetData,W2,sigma,K, wnn))
        
        print('Loss is ', loss[it])
        
        GNew = gradcoswx1coswx2(SubsetData,W2,wnn,sigma, K)
        
            
        if (it<51):        
            gradnorm.append(np.linalg.norm(GNew))
            
            alpha = alphaN /(1+(alphaN*RegW*it))            
            wnn = wnn - (alpha)*(RegW*wnn+GNew)
#            
            WTMP = np.concatenate((WTMP, wnn), axis =1)
###            print('The error2 is', err2)
        else:
            sigGrad = GNew*GOld
            deltaPlus = np.minimum(deltaold*nplus, deltaMax)*(sigGrad>0)
            deltaminus = np.maximum(deltaold*nminus, deltaMin)*(sigGrad<0)
            deltaequal = deltaold*(sigGrad==0)
        
            delta = deltaPlus+deltaminus+deltaequal
            signG = np.sign(sigGrad)
            deltaW = -(signG)*delta*(sigGrad>=0) - deltaW*(sigGrad<0)
            #wnn = wnn + (0.1*wnn+deltaW)
            wnn = wnn +deltaW
                
        iternorm2 = np.linalg.norm(wnn)
        
        
        if (it>0):
            err2=np.linalg.norm(woldnn-wnn)
                
        if (err2<=tol):
#                print('done')
            break
        elif (it>250):
#                print('break')
            break
        woldnn= copy.copy(wnn)
        deltaold = copy.copy(delta)
        GOld = copy.copy(GNew)
        it=it+1
        print('Iteration', it)
        
    return wnn, loss
    

def IPprop(SubsetData, W2,wnn,alphaN, RegW, sigma, K=None):
    import numpy as np
    from grad import gradcoswx1coswx2, Lossfunction2
    from math import copysign
    import copy
    
    if K is None:
        from sklearn.metrics import pairwise
        K=pairwise.rbf_kernel(SubsetData,gamma=sigma**2/2)    
    
    err2=2
    tol =1e-5
    WTMP = wnn
    woldnn = 0
    gradnorm=[]
    loss =[]
    it=0
    deltaold = 0.01
    delta = deltaold
    deltaMin = np.tile(0.0, (np.size(wnn),1))
    deltaMax = np.tile(1.0, (np.size(wnn),1))
    nplus = 1.1
    nminus = 0.25
    GOld = 0.0
    deltaW = 0.0
    old_E = 500
    while(err2>tol):
    
        #np.random.shuffle(SubsetData)
        if (it==51):
            index = np.argmin(loss)
            wnn = WTMP[:, index]
            wnn = wnn[:, np.newaxis]
            deltaold = 0.01
            old_E = 10000
            print('Index', index)
            
        r,s= np.shape(W2)
        if (s==1):
            loss.append(Lossfunction2(SubsetData,wnn,sigma,K))
        else:
            loss.append(Lossfunction2(SubsetData,W2,sigma,K, wnn))
        
        E = loss[it]
        GNew = gradcoswx1coswx2(SubsetData,W2,wnn,sigma, K)
        GNew = (RegW * wnn)+GNew
            
        if (it<51):        
            gradnorm.append(np.linalg.norm(GNew))
            
            alpha = alphaN /(1+(alphaN*RegW*it))            
            wnn = wnn - (alpha)*(GNew)
#            
            WTMP = np.concatenate((WTMP, wnn), axis =1)
####            print('The error2 is', err2)
        else:
            sigGrad = GNew*GOld
            deltaPlus = np.minimum(deltaold*nplus, deltaMax)*(sigGrad>0)
            deltaminus = np.maximum(deltaold*nminus, deltaMin)*(sigGrad<0)
            deltaequal = deltaold*(sigGrad==0)
        
            delta = deltaPlus+deltaminus+deltaequal
            signG = np.sign(sigGrad)+ (it==0)
            deltaW = -(signG)*delta*(sigGrad>=0) - deltaW*((sigGrad<0)*(E>old_E))
            #GNew = GNew*(sigGrad>0)
            #wnn = wnn + (0.1*wnn+deltaW)
            wnn = wnn +deltaW
                
        iternorm2 = np.linalg.norm(wnn)
        
        print('Loss is ', loss[it])
        
        if (it>0):
            err2=np.linalg.norm(woldnn-wnn)
                
        if (err2<=tol):
#                print('done')
            break
        elif (it>250):
#                print('break')
            break
        woldnn= copy.copy(wnn)
        deltaold = copy.copy(delta)
        GOld = copy.copy(GNew)
        
        old_E = copy.copy(E)
        it=it+1
        print('Iteration', it)
        
    return wnn, loss
    
def AdaptiveRateBatchCheck(SubsetData, W2,wnn,alphaN, RegW, sigma, K=None):
    import numpy as np
    from grad import gradcoswx1coswx2, Lossfunction2
    from math import copysign
    import copy
    
    if K is None:
        from sklearn.metrics import pairwise
        K=pairwise.rbf_kernel(SubsetData,gamma=sigma**2/2) 
    
    err2=2
    tol =1e-5
    niter = 251
    WTMP = wnn
    woldnn = 0
    gradnorm=[]
    loss =np.zeros((niter,1))
    it=0
    deltaold = 0.01
    delta = deltaold
    deltaMin = np.tile(0.0, (np.size(wnn),1))
    deltaMax = np.tile(1.0, (np.size(wnn),1))
    nplus = 1.1
    nminus = 0.25
    GOld = 0.0
    deltaW = 0.0
    E_old = 500
    E = 0
    alpha=np.zeros((niter,1))
    Ncheck = 20
    end_ncheck = 0
    n_check = 0
    while(err2>tol):
    
        #np.random.shuffle(SubsetData)
        r,s= np.shape(W2)
        if (s==1):
            loss[it] = (Lossfunction2(SubsetData,wnn,sigma,K))
        else:
            loss[it] = (Lossfunction2(SubsetData,W2,sigma,K, wnn))
        print('Loss is ', loss[it])
        
        if ((np.mod(it, Ncheck)==0) & (it>0)):
            print ('Error check', it)
            E = loss[it]
            st_ncheck = end_ncheck
            n_check = n_check+1
            end_ncheck = Ncheck*n_check
            if (E>E_old):
                print('Error greater', it)
                index = np.argmin(loss[st_ncheck:end_ncheck-1])
                index = index+st_ncheck
                print('Index', index)
                E = loss[index]
                wnn = WTMP[:, index]
                wnn = wnn[:, np.newaxis]
                alphaN = alpha[index]
                print ('alphaN ' , alphaN)
            else:
                alphaN = alpha[it-1]
             
            
        GNew = gradcoswx1coswx2(SubsetData,W2,wnn,sigma, K)
        GNew = (RegW * wnn)+GNew
        alpha[it]= (alphaN /(1+(alphaN*RegW*it)))  
          
        wnn = wnn - (alpha[it])*(GNew)
#   
#        if (it==0):
#            WTMP = wnn
#        else:
        WTMP = np.concatenate((WTMP, wnn), axis =1)
            
        iternorm2 = np.linalg.norm(wnn)
        
        
        
        if (it>0):
            err2=np.linalg.norm(woldnn-wnn)
                
        if (err2<=tol):
#                print('done')
            break
        elif (it>=niter-1):
#           print('break')
            break
        
        woldnn= copy.copy(wnn)
        E_old = copy.copy(E)
        it=it+1
        print('Iteration', it)
        
    minindex = np.argmin(loss)
    wnn = WTMP[:, minindex]
    wnn = wnn[:, np.newaxis]
    return wnn, loss
 
def AdaptiveLearningRateErrorCheck(SubsetData, W2,wnn,alphaN, RegW, gamma,
                                   K=None, philippe_update = False):
    import numpy as np
    from grad import Lossfunction2,gradcoswx1coswx2WithOutLoop
    from math import copysign
    import copy
    import time
    
    if K is None:
        from sklearn.metrics import pairwise
        K=pairwise.rbf_kernel(SubsetData,gamma=gamma) 
    
    err2=2
    tol =1e-5
    niter = 1001
    WTMP = wnn
    woldnn = 0
    gradnorm=[]
    loss =np.zeros((niter,1))
    it=0
    
    SubsetDataN = SubsetData.copy()
    sigma_inv2 = 2*gamma
   
    
    E_old = 500
    E = 0
    alpha=np.zeros((niter,1))
    Ncheck = 100
    end_ncheck = 0
    n_check = 0
    while(err2>tol):
    
        #np.random.shuffle(SubsetData)
        if (it==Ncheck):
#            print ('Error check', it)
            index = np.argmin(loss[0:Ncheck])
#            print('Index', index)
            wnn = WTMP[:, index]
            wnn = wnn[:, np.newaxis]
            alphaN = alpha[index]
            E = loss[index]
#            print('Error', E)
            newit = index
    
        r,s= np.shape(W2)
        
        if (s==1):
            loss[it] = (Lossfunction2(SubsetData,wnn,gamma,K))
        else:
            loss[it] = (Lossfunction2(SubsetData,W2,gamma,K, wnn))
        loss[it]= loss[it]+RegW*(np.sum(np.square(wnn[0:len(wnn)-1])))
#        print('Loss is ', loss[it])
#        time.sleep(0.5)
        
#         IRegW =1
#       #nv = 0.0
#        # Noise addition to the Gradient
#        RegW = IRegW/(1+it)*0.55
#        randomNoise = np.random.normal(0, nv,1)
        randomNoise = 0.0    
        if (it<=Ncheck):   
            #GNew = gradcoswx1coswx2(SubsetData,W2,wnn,sigma, K)
            GNew = gradcoswx1coswx2WithOutLoop(SubsetData,W2,wnn,gamma, K)
            GNew1 = GNew+randomNoise
            GNew = (RegW * wnn)+GNew1
            GNew[-1] = GNew1[-1].copy()
            alpha[it]= (alphaN /(1+(alphaN*RegW*it)))  
          
            wnew = wnn - (alpha[it]*GNew)
            if philippe_update:
               wnew = wnew*(sigma_inv2/np.std(wnew))
        elif (it>(Ncheck)):
            E = loss[it]
            newit =newit+1
            if (E>E_old):
                alphaN = alphaN/5.0
                wnn = woldnn.copy()
                #GNew = gradcoswx1coswx2(SubsetData,W2,wnn,sigma, K)
                GNew = gradcoswx1coswx2WithOutLoop(SubsetData,W2,wnn,gamma, K)
                GNew1 = GNew+randomNoise
                GNew = (RegW * wnn)+GNew1
                GNew[-1] = GNew1[-1].copy()
                beta = (alphaN)# /(1+(alphaN*RegW*newit)))
                wnew = wnn - (beta*GNew)
                if philippe_update:
                   wnew = wnew*(sigma_inv2/np.std(wnew))
            else:
                #GNew = gradcoswx1coswx2(SubsetData,W2,wnn,sigma, K)
                GNew = gradcoswx1coswx2WithOutLoop(SubsetData,W2,wnn,gamma, K)
                GNew1 = GNew+randomNoise
                GNew = (RegW * wnn)+GNew1
                GNew[-1] = GNew1[-1].copy()
                alphaN = alphaN/2.0
                #beta = (alphaN /(1+(alphaN*RegW*newit)))  
                beta = alphaN
                wnew = wnn - (beta*GNew)
                if philippe_update:
                   wnew = wnew*(sigma_inv2/np.std(wnew))
#        SubsetData = SubsetDataN + np.random.normal(0,0.01,(1,1))
        
            
#        if (it==0):
#            WTMP = wnew
#        else:
        WTMP = np.concatenate((WTMP, wnew), axis =1)
            
        iternorm2 = np.linalg.norm(wnew)
        
        
        
        if (it>0):
            err2=np.linalg.norm(woldnn-wnew)
                
        if (err2<=tol):
#                print('done')
            break
        elif (it>=niter-1):
#           print('break')
            break
        woldnn= copy.copy(wnn)
        wnn = copy.copy(wnew)
        E_old = copy.copy(E)
        it=it+1
#        print('Iteration', it)
        
#    print('Iterations', it)
    if (it==1):
        minindex = np.argmin(loss[0])
    else:
        minindex = np.argmin(loss[0:it-1])
    wnn = WTMP[:, minindex] 
    wnn = wnn[:, np.newaxis]
    
    return wnn, loss   
    
    
def philippe_AdaptiveLearningRateErrorCheck(SubsetData, W2,wnn,alphaN, RegW, sigma, K=None):
    '''
    weight update as mentioned by philippe
    new_w = sigma/std(new_w)
    '''
    import numpy as np
    from grad import gradcoswx1coswx2, Lossfunction2,gradcoswx1coswx2WithOutLoop
    from math import copysign
    import copy
    import time
    
    if K is None:
        from sklearn.metrics import pairwise
        K=pairwise.rbf_kernel(SubsetData,gamma=sigma**2/2) 
    
    err2=2
    tol =1e-10
    niter = 201
    WTMP = wnn
    woldnn = 0
    gradnorm=[]
    loss =np.zeros((niter,1))
    it=0
    
    SubsetDataN = SubsetData.copy()
    gamma=sigma**2/2
    b_update = True
    
    E_old = 500
    E = 0
    alpha=np.zeros((niter,1))
    Ncheck = 100
    end_ncheck = 0
    n_check = 0
    while(it<niter):
    
        #np.random.shuffle(SubsetData)
        if (it==Ncheck):
            print ('Error check', it)
            index = np.argmin(loss[0:Ncheck])
            print('Index', index)
            wnn = WTMP[:, index]
            wnn = wnn[:, np.newaxis]
            alphaN = alpha[index]
#            alphaN = 1.0
            E = loss[index]
            old_loss =E
            print('Error', E)
            newit = index
    
        r,s= np.shape(W2)
        
        if (s==1):
            loss[it] = (Lossfunction2(SubsetData,wnn,sigma,K))
        else:
            loss[it] = (Lossfunction2(SubsetData,W2,sigma,K, wnn))
        loss[it] = loss[it]+RegW*np.sum(np.square(wnn))
        #print('Loss is ', loss[it])
        
#         IRegW =1
#       #nv = 0.0
#        # Noise addition to the Gradient
#        RegW = IRegW/(1+it)*0.55
#        randomNoise = np.random.normal(0, nv,1)
        randomNoise = 0.0    
        if (it<=Ncheck):   
            #GNew = gradcoswx1coswx2(SubsetData,W2,wnn,sigma, K)
            GNew = gradcoswx1coswx2WithOutLoop(SubsetData,W2,wnn,sigma, K)
            GNew1 = GNew+randomNoise
            GNew = (RegW * wnn)+GNew1
            if b_update:
               GNew[-1] = GNew1[-1].copy()
            alpha[it]= (alphaN /(1+(alphaN*RegW*it)))  
          
            wnew = wnn - (alpha[it]*GNew)
            # philippe update
            wnew = wnew*(np.sqrt(2*gamma)/np.std(wnew))
            wnn = copy.copy(wnew)
        elif (it>(Ncheck)):
            
            GNew = gradcoswx1coswx2WithOutLoop(SubsetData,W2,wnn,sigma, K)
            GNew1 = GNew+randomNoise
            GNew = (RegW * wnn)+GNew1
            if b_update:
               GNew[-1] = GNew1[-1].copy()                
            beta = (alphaN) /(1+(alphaN*RegW*newit))
            wnew = wnn - (beta*GNew)
                # philippe update
            wnew = wnew*(np.sqrt(2*gamma)/np.std(wnew))
            print('Learning rate =', beta)
            if (s==1):
                nloss = (Lossfunction2(SubsetData,wnew,sigma,K))
            else:
                nloss = (Lossfunction2(SubsetData,W2,sigma,K, wnew))
                
            new_loss = nloss+RegW*np.sum(np.square(wnew))
            print('New Loss =', new_loss)
            print('Old Loss =', old_loss)
            time.sleep(0.7)
            newit =newit+1
            if (new_loss<old_loss):                 
                wnn = np.copy(wnew)
                old_loss = np.copy(new_loss) 
                print('update')
            else:
                alphaN = alphaN*0.7

#        SubsetData = SubsetDataN + np.random.normal(0,0.01,(1,1))
            
            
#        if (it==0):
#            WTMP = wnew
#        else:
        WTMP = np.concatenate((WTMP, wnew), axis =1)
            
        iternorm2 = np.linalg.norm(wnew)
        
        
        
        if (it>0):
            err2=np.linalg.norm(woldnn-wnew)
                
#        if (err2<=tol):
##                print('done')
#            break
        if (it>=niter-1):
#           print('break')
            break
        woldnn= copy.copy(wnn)
        it=it+1
#        print('Iteration', it)
        
#    print('Iterations', it)
    if (it==1):
        minindex = np.argmin(loss[0])
    else:
        minindex = np.argmin(loss[0:it-1])
    wnn = WTMP[:, minindex] 
    wnn = wnn[:, np.newaxis]
    
    return wnn, loss   