import numpy as np

from keras.layers.core import K
from keras.regularizers import ActivityRegularizer
from keras.layers import Dense
from keras.layers import regularizers
from keras.engine import InputSpec # , Layer, Merge

from keras import backend

if backend._BACKEND == 'tensorflow':
    import tensorflow as tf 
    def get_diag(t):
        return tf.diag(t)
    floatname = 'float'
else:
    import theano.tensor.nlinalg
    def get_diag(t):
        return theano.tensor.nlinalg.diag(t)
    floatname = 'floatX'
        
# Mutual information regularizer
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
        
    def __call__(self, loss):
        if not hasattr(self, 'layer'):
            raise Exception('Need to call `set_layer` on ActivityRegularizer instance before calling the instance.')
            
        mi = self.kde_entropy(self.layer.output) - self.hcond
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
    
def logsumexp(mx):
    cmax = K.max(mx, axis=1)
    cmax2 = K.expand_dims(cmax, 1)
    mx2 = mx - cmax2
    return cmax + K.log(K.sum(K.exp(mx2), axis=1))

def kde_entropy(output, var):
    Nint = K.shape(output)[0]
    N    = K.cast(Nint, floatname )
    dims = K.cast(K.shape(output)[1], floatname )
    
    normconst = (dims/2)*K.log(2*np.pi*var)
            
    # Kernel density estimation of entropy
    y1 = K.expand_dims(output, 0)
    y2 = K.expand_dims(output, 1)

    dists = K.sum((y1-y2)**2, axis=2) / (2*var)

    # Removes effect of diagonals
    diagvals = get_diag(10e20*K.ones_like(K.sum(dists,axis=1)))
    dists = dists + diagvals
    
    lprobs = logsumexp(-dists) - K.log(N-1) - normconst
    
    h = -K.mean(lprobs)
    
    return h, normconst
    

    
class mireg(ActivityRegularizer):
    # Mutual information regularizer
    def __init__(self): #, alpha, entropy_only):
        #self.alpha      = alpha          # weight of MI regularization
        # kde entropy for output from single input (delta function)
        self.uses_learning_phase = True
        #self.entropy_only = entropy_only # whether to compute entropy or MI
        #self.randoutput = K.variable(np.random.random((250,50)))

    def __call__(self, loss):
        var = K.exp(self.layer.logvar)
        output = self.layer.input
        #output = self.randoutput
        
        c_loss, normconst = kde_entropy(output, var)
        c_loss = c_loss - (1 - self.layer.entropy_only) * normconst
            
        regularized_loss = loss + self.layer.alpha * c_loss
        
        return K.in_train_phase(regularized_loss, loss)
        
    def get_config(self):
        return {'name': self.__class__.__name__}
    
class MILayer(Dense):
    def __init__(self, output_dim, input_dim=None, alpha=1.0, initlogvar=-2.0, entropy_only=False, add_noise=False, **kwargs):
        self.output_dim   = output_dim
        self.input_dim    = input_dim
        self.activity_regularizer = mireg() # alpha,entropy_only)
        
        self.param_alpha        = alpha
        self.param_initlogvar   = initlogvar
        self.param_entropy_only = entropy_only
        self.param_add_noise    = add_noise
        
        self.input_spec = [InputSpec(ndim=2)]
        if self.input_dim: kwargs['input_shape'] = (self.input_dim,)
        super(Dense, self).__init__(**kwargs)

    def set_myparams(self, alpha, logvar, entropy_only, add_noise):
        K.set_value(self.alpha, alpha)
        K.set_value(self.logvar, float(logvar))
        K.set_value(self.entropy_only, 1 if entropy_only else 0)
        K.set_value(self.add_noise, 1 if add_noise else 0)
        
    def init_trainable_weights(self):
        self.trainable_weights = []
        
    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]

        self.alpha        = K.variable(self.param_alpha)
        self.entropy_only = K.variable(1 if self.param_entropy_only else 0)
        self.add_noise    = K.variable(1 if self.param_add_noise else 0)
        self.logvar       = K.variable(float(self.param_initlogvar))
        self.init_trainable_weights()
        
        self.input_spec = [InputSpec(dtype=K.floatx(), shape=(None, input_dim))]
                    
        self.regularizers = []
        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

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
        
