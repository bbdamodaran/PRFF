# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 10:00:32 2017

@author: damodara
"""

import numpy
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.base import clone
#from revrand.basis_functions import OrthogonalRBF
import pylab
import time
__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


class ORFPRFF(RBFSampler):
    def __init__(self, gamma=1.0, n_components=100, random_state=None, minibatch_size=128, max_iter=1000, n_pass=1,
                 tol=1e-8, alpha=10., lbda=0., verbose=False, update_b=True):
        RBFSampler.__init__(self, gamma=gamma, n_components=n_components, random_state=random_state)
        self.verbose = verbose
        self.update_b = update_b
        self.intial_loss = 0.0
        self.end_loss = 0.0
        self.n_components = n_components
        self.gamma = gamma
        
    def loss_function(self,data, gram_data):
        Ndata = numpy.shape(data)[0]
        Phi = self.transform(data)
        
        gram_approx = numpy.dot(Phi, Phi.T)
        
#        L= numpy.linalg.norm(gram_data-gram_approx)/numpy.linalg.norm(gram_data)
        L = numpy.sum((gram_data-gram_approx)**2)/Ndata**2
        return L
        
    def fit(self, X, y=None):
        RBFSampler.fit(self, X=X, y=y)
        Xdim = numpy.shape(X)[0]
        #or_rbf = OrthogonalRBF(Xdim= Xdim, nbases=self.n_components, lenscale = self.gamma,
                              #random_state= self.random_state)
        #self. random_weights_ = or_rbf.W
        #self. random_offset_ = 3.0
        
        WRD = self.random_weights_
        Q = numpy.linalg.qr(WRD)[0]
        S = numpy.sqrt(numpy.random.chisquare(Xdim, Xdim))
        weights = numpy.diag(S).dot(Q)
        sigma = numpy.sqrt(2 * self.gamma)
        self.random_weights_ = numpy.sqrt(2*sigma)*weights
        
#        #if self.update_b is True:
#            WRD = numpy.concatenate((self.random_weights_, self.random_offset_[numpy.newaxis,:]), axis = 0)
#            X_copy = numpy.concatenate((X_copy, numpy.ones((Ndata,1), dtype= numpy.float64)), axis=1)
#        else:
#            WRD = self.random_weights    
       
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