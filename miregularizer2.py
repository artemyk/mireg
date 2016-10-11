import numpy as np

from keras.layers.core import K
from keras.regularizers import ActivityRegularizer
from keras.layers import Dense
from keras.layers import regularizers
from keras.engine import InputSpec # , Layer, Merge

if K._BACKEND == 'tensorflow':
    import tensorflow as tf 
    def get_diag(t):
        return tf.diag(t)
    #floatname = 'float'
else:
    import theano.tensor.nlinalg
    def get_diag(t):
        return theano.tensor.nlinalg.diag(t)
    #floatname = 'floatX'
        
# Mutual information regularizer
"""
class MIRegularizer(ActivityRegularizer):
    def __init__(self, alpha=0., var=1.0):
        super(MIRegularizer, self).__init__()
        f = K.cast_to_floatx
        self.alpha      = f(alpha)     # weight of MI regularization
        self.var        = f(var)       # KDE Gaussian variance
        self.gauss_norm = f(1.0/np.sqrt(2*np.pi*var))  # normalizing constants
        # kde entropy for output from single input (delta function)
        self.hcond      = f(-np.log(self.gauss_norm * np.exp(0.0 / (2*var))))  
        self.uses_learning_phase = True
        
    def get_mi(self, output):
        return self.kde_entropy(self.layer.output) - self.hcond
    
    def __call__(self, loss):
        if not hasattr(self, 'layer'):
            raise Exception('Need to call `set_layer` on ActivityRegularizer instance before calling the instance.')
            
        mi = self.get_mi(self.layer.output)
        regularized_loss = loss + self.alpha * mi
        return K.in_train_phase(regularized_loss, loss)

    def kde_entropy(self, output):
        # Kernel density estimation of entropy
        y1 = K.expand_dims(output, 0)
        y2 = K.expand_dims(output, 1)

        dists = K.sum((y1-y2)**2, axis=2)
        probs = self.gauss_norm * K.exp(-dists / (2*self.var))

        probs = K.mean(probs, axis=1)
        lprobs = K.log(probs)
        h = K.sum(-K.mean(lprobs))
        return h

    def get_config(self):
        return {'name': self.__class__.__name__, 'alpha': self.alpha, 'var': self.var}
    
"""

def logsumexp(mx, axis):
    cmax = K.max(mx, axis=axis)
    cmax2 = K.expand_dims(cmax, 1)
    mx2 = mx - cmax2
    return cmax + K.log(K.sum(K.exp(mx2), axis=1))

def kde_entropy(output, var):
    dims = int(output.get_shape()[1])
    N    = K.cast(K.shape(output)[0], K.floatx() )
    
    normconst = (dims/2.0)*K.log(2*np.pi*var)
            
    # Kernel density estimation of entropy
    y1 = K.expand_dims(output, 0)
    y2 = K.expand_dims(output, 1)

    dists = K.sum((y1-y2)**2, axis=2) / (2*var)
    normCount = N

    ## Removes effect of diagonals, i.e. leave-one-out entropy
    #normCount = N-1
    #diagvals = get_diag(10e20*K.ones_like(dists[0,:]))
    #dists = dists + diagvals
    
    lprobs = logsumexp(-dists, axis=1) - np.log(normCount) - normconst
    
    h = -K.mean(lprobs)
    
    return h # , normconst + (dims/2.0)

def kde_condentropy(output, var):
    dims = int(output.get_shape()[1])
    #normconst = (dims/2.0)*K.log(2*np.pi*var)
    #return normconst + (dims/2.0)
    normconst = (dims/2.0)*K.log(2*np.pi*var)
    return normconst

#def kde_mi(data, var, entropy_only=0):
#    h, hcond = kde_entropy(data, var)
#    return h - (1.0 - entropy_only)

"""
import scipy.misc
def kde_entropy_from_dists_loo2(dists, N, dims, var):
    dists2 = dists / (2*var)

    normconst = (dims/2.0)*K.log(2*np.pi*var)

    # Removes effect of diagonals, i.e. leave-one-out entropy
    normCount = float(N-1)
    diagvals = get_diag(10e20*K.ones_like(dists[0,:]))
    dists2 = dists2 + diagvals
    
    lprobs = logsumexp(-dists2, axis=1) - np.log(normCount) - normconst

    h = -K.mean(lprobs)
    return h
"""


"""

class mireg(ActivityRegularizer):
    # Mutual information regularizer
    def __init__(self): #, alpha, entropy_only):
        super(mireg, self).__init__()
        #self.alpha      = alpha          # weight of MI regularization
        # kde entropy for output from single input (delta function)
        self.uses_learning_phase = True
        #self.entropy_only = entropy_only # whether to compute entropy or MI
        #self.randoutput = K.variable(np.random.random((250,50)))
        
    #def get_val(self):
    #    data = self.layer.input
    #    var = K.exp(self.layer.logvar)
    #    c_loss, hcond = kde_entropy(data, var)
    #    c_loss = c_loss - (1 - self.layer.entropy_only) * hcond
    #    return c_loss, hcond
    
    def __call__(self, loss):
        #c_loss, hcond = self.get_val()
        #output = self.randoutput
        c_loss = kde_mi(self.layer.input, K.exp(self.layer.logvar), self.entropy_only)
        regularized_loss = loss + self.layer.alpha * c_loss
        return K.in_train_phase(regularized_loss, loss)
        
    def get_config(self):
        return {'name': self.__class__.__name__}
"""
    
class MILayer(Dense):
    def __init__(self, output_dim, alpha=1.0, initlogvar=-2.0, entropy_only=False, add_noise=False, **kwargs):
        #output_dim, input_dim=None, 
        #self.output_dim   = output_dim
        #self.input_dim    = input_dim
        super(MILayer, self).__init__(output_dim, 
                                      bias=False, 
                                      weights=[np.eye(output_dim)], 
                                      #activity_regularizer=mireg(), 
                                      **kwargs)
        
        self.param_alpha        = alpha
        self.param_initlogvar   = initlogvar
        self.param_entropy_only = entropy_only
        self.param_add_noise    = add_noise
        
        self.uses_learning_phase = True
        
        #self.input_spec = [InputSpec(ndim=2)]
        #if self.input_dim: kwargs['input_shape'] = (self.input_dim,)

    def set_myparams(self, alpha, logvar, entropy_only, add_noise):
        #K.set_value(self.alpha, alpha)
        K.set_value(self.logvar, float(logvar))
        #K.set_value(self.entropy_only, 1 if entropy_only else 0)
        #K.set_value(self.add_noise, 1 if add_noise else 0)
        
    def init_trainable_weights(self):
        self.trainable_weights = []
        
    def build(self, input_shape):
        super(MILayer, self).build(input_shape)
        
        #self.alpha        = K.variable(self.param_alpha)
        #self.entropy_only = K.variable(1 if self.param_entropy_only else 0)
        #self.add_noise    = K.variable(1 if self.param_add_noise else 0)
        self.alpha        = self.param_alpha
        self.entropy_only = int(self.param_entropy_only)
        self.add_noise    = int(self.param_add_noise)
        self.logvar       = K.variable(float(self.param_initlogvar))
        
        self.init_trainable_weights()

    def call(self, x, mask=None):
        noise_x = x + self.add_noise * K.random_normal(shape=K.shape(x), mean=0., std=K.sqrt(K.exp(self.logvar)))
        return K.in_train_phase(noise_x, x)
        
    def get_config(self):
        return {'name': self.__class__.__name__, 'alpha': self.alpha, 
                'initlogvar'  : self.param_initlogvar,
                'entropy_only': self.param_entropy_only,
                'add_noise'   : self.param_add_noise,
               }
    
class MILayerTrainable(MILayer):
    def init_trainable_weights(self):
        self.trainable_weights = [self.logvar]
        
