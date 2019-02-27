# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 11:40:51 2017

@author: damodara
"""

import numpy
import copy
import time
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.base import clone
import pylab
from GradientDescentMethods import AdaptiveLearningRateErrorCheck
from revrand.basis_functions import OrthogonalRBF
__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


class ORFF(RBFSampler):
    def __init__(self, gamma=1.0, n_components=100, random_state=None, minibatch_size=128, max_iter=1000, n_pass=1,
                 tol=1e-8, alpha=10., lbda=0., verbose=False, update_b=True):
        RBFSampler.__init__(self, gamma=gamma, n_components=n_components, random_state=random_state)
        self.minibatch_size = minibatch_size
        self.max_iter = max_iter
        self.n_pass = n_pass
        self.tol = tol
        self.alphaN = alpha
        self.lbda = lbda
        self.verbose = verbose
        self.update_b = update_b
        
    def fit(self, X, y=None):
        RBFSampler.fit(self, X=X, y=y)
        X_copy = X.copy()
        Ndata = numpy.shape(X_copy)[0]
        Xdim = numpy.shape(X_copy)[1]
#        if self.update_b is True:
#            WRD = numpy.concatenate((self.random_weights_, self.random_offset_[numpy.newaxis,:]), axis = 0)
#            X_copy = numpy.concatenate((X_copy, numpy.ones((Ndata,1), dtype= numpy.float64)), axis=1)
#        else:
#            WRD = self.random_weights_
        
#        WRD = self.random_weights_
#        WRD = numpy.random.randn(Xdim,self.n_components)
#        Q = numpy.linalg.qr(WRD, mode='raw')[0]
#        S = numpy.sqrt(numpy.random.chisquare(Xdim, Xdim))
#        weights = numpy.diag(S).dot(Q.T)
        sigma = numpy.sqrt(1/(2 * self.gamma))
#        self.random_weights_ = weights 
#        

#        reps = int(numpy.ceil(self.n_components / Xdim))
#        Q = numpy.empty((Xdim, Xdim*reps))
#
#        for r in range(reps):
#            #W = self.random_weights_
#            #W = self._random.randn(Xdim, Xdim)
#            W = numpy.random.randn(Xdim, Xdim)
#            Q[:, (r * Xdim):((r + 1) * Xdim)] = numpy.linalg.qr(W)[0]
#
#        S = numpy.sqrt(numpy.random.chisquare(df=Xdim, size=Xdim))
#        weights = numpy.diag(S).dot(Q[:, :self.n_components])
#        sigma = numpy.sqrt(2 * self.gamma)
#        #self.random_weights_ = (1/sigma)*weights
#        #self.random_weights_ = numpy.sqrt(2*sigma)*weights
        
        or_rbf = OrthogonalRBF(Xdim= Xdim, nbases=self.n_components, lenscale = sigma,
                              random_state= self.random_state)
        self. weights = or_rbf.W
        self. offset = numpy.random.rand(self.n_components)*(2*numpy.pi)
        self.random_weights_ = or_rbf.W/sigma
        #self. random_offset_ = 3.0
        
        
        
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
        
    def orf_transform(self, X):
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

        

        
        
        
        k= numpy.shape(self.weights)[1]

        projection = np.dot(X, self.weights)
        projection += self.offset
        np.cos(projection, projection)
        projection *= np.sqrt(2.) / np.sqrt(k)
        return projection        



    

