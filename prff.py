import numpy
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.base import clone
import pylab
import time
__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


class PRFF(RBFSampler):
    def __init__(self, gamma=1.0, n_components=100, random_state=None, minibatch_size=128, max_iter=1000, n_pass=1,
                 tol=1e-5, alpha=10., lbda=0., verbose=False, update_b=True):
        RBFSampler.__init__(self, gamma=gamma, n_components=n_components, random_state=random_state)
        self.minibatch_size = minibatch_size
        self.max_iter = max_iter
        self.n_pass = n_pass
        self.tol = tol
        self.alpha = alpha
        self.lbda = lbda
        self.verbose = verbose
        self.update_b = update_b
        self.intial_loss = 0.0
        self.end_loss = 0.0
        
    def loss_function(self,data, gram_data):
        Ndata = numpy.shape(data)[0]
        Phi = self.transform(data)
        
        gram_approx = numpy.dot(Phi, Phi.T)
        
#        L= numpy.linalg.norm(gram_data-gram_approx)/numpy.linalg.norm(gram_data)
        L = numpy.sum((gram_data-gram_approx)**2)/Ndata**2
        return L
        
    def fit(self, X, y=None):
        RBFSampler.fit(self, X=X, y=y)
        for i_pass in range(self.n_pass):
            IntLoss = numpy.zeros((self.n_components,1))
            EnLoss = numpy.zeros((self.n_components,1))
            for comp in range(self.n_components):
                if self.verbose:
                    print("COMPONENT %d, " % comp, end="")
                indices_minibatch = numpy.random.choice(X.shape[0], self.minibatch_size)
                minibatch = X[indices_minibatch]
                gram_minibatch = rbf_kernel(minibatch, gamma=self.gamma)
                phi = self.transform(minibatch)
                diff_mat = gram_minibatch - numpy.dot(phi, phi.T)
                n_iter = 0
                err = numpy.inf
                IntLoss[comp] = self.loss_function(minibatch, gram_minibatch)
                if self.verbose:
                    print('Intial Loss', IntLoss[comp])
                
                while err > self.tol and n_iter < self.max_iter:
                    w_old = self.random_weights_[:, comp].copy()
#                    phi = self.transform(minibatch)
#                    diff_mat = gram_minibatch - numpy.dot(phi, phi.T)

                    wx_b = numpy.dot(minibatch, self.random_weights_[:, comp]) + self.random_offset_[comp]
                    sin_wx = numpy.sin(wx_b).reshape((-1, 1))
                    cos_wx = numpy.cos(wx_b).reshape((-1, 1))

                    sin_cos = numpy.dot(sin_wx, cos_wx.T) * 2 / (self.n_components * self.minibatch_size ** 2)
                    diff_sin_cos = numpy.diag(numpy.dot(diff_mat, 2. * sin_cos.T)).reshape((-1, 1))
                    dl_dw = numpy.sum(diff_sin_cos * minibatch, axis=0)

                    self.random_weights_[:, comp] -= (self.alpha) * (self.lbda* self.random_weights_[:, comp] + dl_dw)
                    
                    if self.update_b:
                        dl_db = numpy.sum(diff_sin_cos)
                        self.random_offset_[comp] -= self.alpha * dl_db
                    err = numpy.linalg.norm(w_old - self.random_weights_[:, comp])
                    n_iter += 1
                EnLoss[comp] = self.loss_function(minibatch, gram_minibatch)
                
                if self.verbose:
                    print("%d iterations" % n_iter)
                    print('End Loss', EnLoss[comp])
                    time.sleep(2)
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
        
        
     

    
        
        
        
        
