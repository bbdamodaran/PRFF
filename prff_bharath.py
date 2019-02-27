import numpy
import copy
import time
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.base import clone
import pylab
from GradientDescentMethods import AdaptiveLearningRateErrorCheck,philippe_AdaptiveLearningRateErrorCheck
from GradientDescentMethods import Gradientdescent
__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


class PRFF_Bharath(RBFSampler):
    def __init__(self, gamma=1.0, n_components=100, random_state=None, minibatch_size=128, max_iter=1000, n_pass=1,
                 tol=1e-8, alpha=10., lbda=0., verbose=False, update_b=True,philippe_update =False):
        RBFSampler.__init__(self, gamma=gamma, n_components=n_components, random_state=random_state)
        self.minibatch_size = minibatch_size
        self.max_iter = max_iter
        self.n_pass = n_pass
        self.tol = tol
        self.alphaN = alpha
        self.lbda = lbda
        self.verbose = verbose
        self.update_b = update_b
        self.philippe_update = philippe_update
        
    def fit(self, X, y=None):
        RBFSampler.fit(self, X=X, y=y)
        X_copy = X.copy()
        Ndata = numpy.shape(X_copy)[0]
        Nfeat = numpy.shape(X_copy)[1]
        sigma_inv = numpy.sqrt(2 * self.gamma)
        

#        self.random_weights_ = sigma_inv*self.random_weights_
        non_rff_initalize = False
        if non_rff_initalize:
           self.random_weights_ = (1e-10)*numpy.random.normal(0,1,(numpy.shape(self.random_weights_)[0],numpy.shape(self.random_weights_)[1]))
           self.random_offset_ = (1e-10)*self.random_offset_
        if self.update_b is True:
            WRD = numpy.concatenate((self.random_weights_, self.random_offset_[numpy.newaxis,:]), axis = 0)
            X_copy = numpy.concatenate((X_copy, numpy.ones((Ndata,1), dtype= numpy.float64)), axis=1)
        else:
            WRD = self.random_weights_
            
        
        nbatches = int(numpy.round(Ndata / self.minibatch_size))
        
        j = 0
        numit = 0
        minloss = numpy.zeros((self.n_components, 1), dtype=int)
        IntLoss = numpy.zeros((self.n_components, 1), dtype=numpy.float64)
        EnLoss = numpy.zeros((self.n_components, 1), dtype=numpy.float64)
        loss_write = False
        if loss_write:
           fntxtname = 'losstrack.txt'
           loss_track = open(fntxtname, 'w')
           loss_track.write('dimension\t StartLoss\t EndLoss')
        RegW = copy.copy(self.lbda)
        while (j < self.n_components):
            # shuffle the data
            numpy.random.shuffle(X_copy)
            for i in range(nbatches):
                wn = WRD[:, j]
                wn = wn[:, numpy.newaxis]
                wnn = wn.copy()
                woldnn = wnn.copy()
                err2 = 2
                if (j == 0):
                    W2 = WRD[:, 0]
                    W2 = W2[:, numpy.newaxis]
#                   W2 = WRD.copy()
                    #
                    ##
                if i < (nbatches - 1):
                    st = i * self.minibatch_size
                    last = ((i + 1) * self.minibatch_size)
                else:
                    st = i * self.minibatch_size
                    last = Ndata
                #
                SubsetData = X_copy[st:last, ]   
                #
#                indices_minibatch = numpy.random.choice(X_copy.shape[0], self.minibatch_size)
#                SubsetData = X_copy[indices_minibatch]
                
                # RBF Kernel on the batch
                K = rbf_kernel(SubsetData, gamma=self.gamma)
#                K = rbf_kernel(SubsetData, gamma=self.gamma)
                numit = numit + 1
                ita = 1
                it = 0
                loss = []
                WTMP = wnn
                #%%
                
                alphaN = copy.copy(self.alphaN)
                if (j==0):
                     IntLoss[j]=Lossfunction(SubsetData,W2,self.gamma,K)
                     IntLoss[j]=IntLoss[j]+RegW*numpy.sum(numpy.square(W2[0:len(W2)-1]))
                else:
                    IntLoss[j]=Lossfunction(SubsetData,W2,self.gamma,K,wnn)
                    IntLoss[j]=IntLoss[j]+RegW*numpy.sum(numpy.square(wnn[0:len(wnn)-1]))
                if self.verbose:
                    print('Intial Loss', IntLoss[j])
        
                
                #wnn, loss = Rprop(SubsetData, W2, wnn, sigma, K)
                
                
                wnn, loss = AdaptiveLearningRateErrorCheck(SubsetData, W2,wnn,alphaN, RegW, 
                                                           self.gamma, K, philippe_update = self.philippe_update )
                # philippe update
#                wnn, loss = philippe_AdaptiveLearningRateErrorCheck(SubsetData, W2,wnn,alphaN, RegW, gamma, K)
#                print('reg w=', RegW)
#                wnn,loss = Gradientdescent(SubsetData, W2,wnn,alphaN, RegW, 
#                                           self.gamma, K,update_b= self.update_b, 
#                                           philippe_update = self.philippe_update)
#               
                '''    
                # %%
                while (err2 > self.tol):
                    
                    if (j==0):
                        loss.append(Lossfunction(SubsetData,W2,sigma,K))
                    else:
                        loss.append(Lossfunction(SubsetData,W2,sigma,K,wnn)) 
                    # Gradient
                    #GNew = gradForm2(SubsetData, W2, wnn, sigma, K)
                    GNew = gradForm2WithOutLoop(SubsetData, W2, wnn, sigma, K)
                    
                    alpha = self.alphaN / (1 + (self.alphaN * self.lbda * ita))
                    # update rule
                    wnn = wnn - (alpha) * (self.lbda * wnn + GNew)
                    WTMP = numpy.concatenate((WTMP, wnn), axis=1)
                    ita = ita + 1
                    err2 = numpy.linalg.norm(woldnn - wnn)
                    if err2 <= self.tol or it > self.max_iter:
                        break
                    woldnn = wnn
                    it = it + 1
                    
                minloss[j] = numpy.argmin(loss)
                wnn = WTMP[:, minloss[j]]
                '''
                if (j==0):
                    W2 = wnn
                    EnLoss[j]=Lossfunction(SubsetData,W2,self.gamma,K)
                    EnLoss[j]=EnLoss[j]+RegW*numpy.sum(numpy.square(W2[0:len(W2)-1]))
                else:
                    EnLoss[j]=Lossfunction(SubsetData,W2,self.gamma,K,wnn)
                    EnLoss[j]=EnLoss[j]+RegW*numpy.sum(numpy.square(wnn[0:len(wnn)-1]))
                if loss_write:
                   loss_track.write('%s\t' %j)
                   loss_track.write('%s\t' %IntLoss[j])
                   loss_track.write('%s\n' %EnLoss[j])
                if (j==0):
                    W2 = wnn
                else:
                    W2 = numpy.concatenate((W2, wnn), axis=1)

               # W2[:,j]=numpy.squeeze(wnn)
                #EnLoss[j]=Lossfunction(SubsetData,W2,sigma,K)
                
                if self.verbose:
                    print('End Loss', EnLoss[j])
                    time.sleep(2)
                    print('Feature Expansion', j + 1, ' is over')
                    
                j = j + 1
                if (j >= self.n_components):
                    break
        if loss_write:        
           loss_track.close()        
        NW = numpy.shape(W2)[0]
        if self.update_b is True:
            self.random_offset_ = W2[NW-1:,]
            self.random_weights_ = numpy.delete(W2,NW-1, axis=0)
        else:
            self.random_weights_ = W2
        self.intial_loss = IntLoss
        self.end_loss = EnLoss
        return self

    def clone(self):
        sampler = clone(self)
        sampler.random_weights_ = self.random_weights_.copy()
        sampler.random_offset_ = self.random_offset_.copy()
        return sampler
        
    def bh_transform(self, X, k):
        """Apply the approximate feature map to X, only upto k features.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.
        k : only use upto k feature subset
        Returns
        -------
        X_new : array-like, shape (n_samples, k)
        """


        import numpy as np

        
        from .utils import check_array, check_random_state, as_float_array
        from .utils.extmath import safe_sparse_dot
        from .utils.validation import check_is_fitted
        
        
        check_is_fitted(self, 'random_weights_')

        X = check_array(X, accept_sparse='csr')
        projection = safe_sparse_dot(X, self.random_weights_[:,0:k])
        projection += self.random_offset_[0:k]
        np.cos(projection, projection)
        projection *= np.sqrt(2.) / np.sqrt(k)
        return projection


def gradForm2(Data, W, w, gamma, K1=None):
    # gradient of L= 1/2N (k(x1,x2)-cos(Wx1)cos(Wx2))
    import numpy as np
    from sklearn.metrics import pairwise

    Nexp = np.shape(W)[1]
    Ndata, Nfeat = np.shape(Data)

    if K1 is None:
        K1 = pairwise.rbf_kernel(Data, gamma=gamma)

    C = np.zeros((Nfeat, Ndata))
    B = np.zeros((Nfeat, Ndata))
    for i in range(Ndata):
        X = np.tile(Data[i,], (Ndata, 1))
        K = K1[i, :]
        K = K[:, np.newaxis]
        #
        c1 = np.outer(np.cos(np.dot(Data, w)) * K, np.sin(np.dot(Data[i,], w)))
        c2 = np.outer(np.sin(np.dot(Data, w)) * K, np.cos(np.dot(Data[i,], w)))
        C[:, i] = np.squeeze((np.dot((X.T), c1) + np.dot((Data.T), c2)))
        #

        AK = np.dot(np.cos(np.dot(Data[i,], W)), np.cos(np.dot(Data, W)).T)
        AK = AK[:, np.newaxis]
        #
        b1 = np.outer(np.cos(np.dot(Data, w)) * AK, np.sin(np.dot(Data[i,], w)))
        b2 = np.outer(np.sin(np.dot(Data, w)) * AK, np.cos(np.dot(Data[i,], w)))
        B[:, i] = np.squeeze((np.dot((X.T), b1) + np.dot((Data.T), b2)))
        #

    L = (1.0 / Ndata ** 2) * (np.sum(C, axis=1) - (2.0 / Nexp) * np.sum(B, axis=1))
    L = L[:, np.newaxis]
    return L
    
def gradForm2WithOutLoop(Data,W,w, gamma, K1 = None):
    
    import numpy
    from sklearn.metrics import pairwise
    
    n_components = numpy.shape(W)[1]
    minibatch_size, Nfeat = numpy.shape(Data)
    
    if K1 is None:
        K1 = pairwise.rbf_kernel(Data, gamma=gamma)
    
    Phi = numpy.cos(numpy.dot(Data,W))
    AK= (2.0/n_components)*numpy.dot(Phi, Phi.T)
    diff_mat = K1-AK
    
    wx_b = numpy.dot(Data, w)
    
    sin_wx = numpy.sin(wx_b).reshape((-1, 1))
    cos_wx = numpy.cos(wx_b).reshape((-1, 1))

    sin_cos = numpy.dot(sin_wx, cos_wx.T) * 2 / (n_components * minibatch_size ** 2)
    
    diff_sin_cos = numpy.diag(numpy.dot(diff_mat, 2. * sin_cos.T)).reshape((-1, 1))
    dl_dw = numpy.sum(diff_sin_cos * Data, axis=0).reshape((-1,1))
    
    return dl_dw
    
def Lossfunction(Data,W,gamma, K= None, w = None):
    import numpy as np
    from sklearn.metrics import pairwise
    
    if w is None:
        W=W
    else:
        W = np.concatenate((W,w), axis =1)
    
    
    Nexp=np.shape(W)[1]
    Ndata,Nfeat=np.shape(Data)
    # if kernel is not provided
    if K is None:
        K=pairwise.rbf_kernel(Data,gamma=gamma)
  
    Phi = np.cos(np.dot(Data,W))
    AK= (2.0/Nexp)*np.dot(Phi, Phi.T)
    
    L = np.sum((K- AK) ** 2) / Ndata ** 2
#    L= np.linalg.norm(K-AK)/np.linalg.norm(K)

    return L