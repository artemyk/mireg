import numpy as np
import scipy

from keras.layers.core import K
from keras.regularizers import ActivityRegularizer
from keras.layers import Dense
from keras.layers import regularizers
from keras.engine import InputSpec # , Layer, Merge

nats2bits = 1.0/np.log(2)

if K._BACKEND == 'tensorflow':
    import tensorflow as tf 
    def get_diag(t):
        return tf.diag(t)
else:
    import theano.tensor.nlinalg
    def get_diag(t):
        return theano.tensor.nlinalg.diag(t)

def logsumexp(mx, axis):
    cmax = K.max(mx, axis=axis)
    cmax2 = K.expand_dims(cmax, 1)
    mx2 = mx - cmax2
    return cmax + K.log(K.sum(K.exp(mx2), axis=1))

def kde_entropy(output, var):
    dims = K.cast(K.shape(output)[1], K.floatx() ) #int(K.shape(output)[1])
    N    = K.cast(K.shape(output)[0], K.floatx() )
    
    normconst = (dims/2.0)*K.log(2*np.pi*var)
            
    # Kernel density estimation of entropy
    
    # get dists matrix
    x2 = K.expand_dims(K.sum(K.square(output), axis=1), 1)
    #x2 = x2 + K.transpose(x2)
    #return K.shape(x2)
    dists = x2 + K.transpose(x2) - 2*K.dot(output, K.transpose(output))
    dists = dists / (2*var)
    
    #y1 = K.expand_dims(output, 0)
    #y2 = K.expand_dims(output, 1)
    #dists = K.sum(K.square(y1-y2), axis=2) / (2*var)
    
    normCount = N

    ## Removes effect of diagonals, i.e. leave-one-out entropy
    #normCount = N-1
    #diagvals = get_diag(10e20*K.ones_like(dists[0,:]))
    #dists = dists + diagvals
    
    lprobs = logsumexp(-dists, axis=1) - K.log(normCount) - normconst
    
    h = -K.mean(lprobs)
    
    return nats2bits * h # , normconst + (dims/2.0)

def kde_condentropy(output, var):
    dims = K.cast(K.shape(output)[1], K.floatx() ) # int(output.get_shape()[1])
    # #normconst = (dims/2.0)*K.log(2*np.pi*var)
    # #return normconst + (dims/2.0)
    normconst = (dims/2.0)*K.log(2*np.pi*var)
    return nats2bits * normconst
    #c = 0.5 * dims * K.log( 2 * np.pi * np.e * var )
    #return nats2bits * c

#def kde_mi(data, var, entropy_only=0):
#    h, hcond = kde_entropy(data, var)
#    return h - (1.0 - entropy_only)

def kde_entropy_from_dists_loo(dists, N, dims, var):
    # should have large values on diagonal
    dists2 = dists / (2*var)
    normconst = (dims/2.0)*K.log(2*np.pi*var)
    lprobs = logsumexp(-dists2, axis=1) - np.log(N-1) - normconst
    h = -K.mean(lprobs)
    return nats2bits * h

from keras.layers import Layer
from keras.regularizers import ActivityRegularizer

class MIComputer(object):
    def __init__(self, inputvar, kdelayer, noiselayer):
        self.input = inputvar
        self.kdelayer = kdelayer
        self.noiselayer = noiselayer
        
    def get_h(self):
        totalvar = K.exp(self.noiselayer.logvar)+K.exp(self.kdelayer.logvar)
        return kde_entropy(self.input, totalvar)
    
    def get_hcond(self):
        return kde_condentropy(self.input, K.exp(self.noiselayer.logvar))
        #return kde_entropy(self.noiselayer.get_noise(self.input), K.exp(self.noiselayer.logvar))
    
    def get_mi(self):
        return self.get_h() - self.get_hcond()
    
        
class MIRegularizer(ActivityRegularizer):
    def __init__(self, micomputer, alpha):
        super(MIRegularizer, self).__init__()
        self.micomputer = micomputer
        self.alpha = K.variable(alpha)
        
    def __call__(self, loss):
        if not hasattr(self, 'layer'):
            raise Exception('Need to call `set_layer` on ActivityRegularizer instance before calling the instance.')
        mi = self.micomputer.get_mi()
        regularized_loss = loss + self.alpha * mi
        return K.in_train_phase(regularized_loss, loss)
    
    """
    def __init__(self, layer, data):
        super(MIRegularizer, self).__init__()
        self.layer = layer
        self.traindata = K.variable(d.train.X)
        self.testdata  = K.variable(d.test.X)

    def get_input(self):
        #return self.layer.input
        return self.traininput
    
    def get_h(self):
        totalvar = K.exp(self.layer.logvar)+K.exp(self.layer.kdelayer.logvar)
        return kde_entropy(self.get_input(), totalvar)
    
    def get_hcond(self):
        return kde_condentropy(self.get_input(), K.exp(self.layer.logvar))
    
    def get_mi(self):
        return self.get_h(mode) - self.get_hcond()
    
    def __call__(self, loss):
        if not hasattr(self, 'layer'):
            raise Exception('Need to call `set_layer` on ActivityRegularizer instance before calling the instance.')
            
        mi = self.get_mi()
        regularized_loss = loss + self.layer.alpha * mi
        return K.in_train_phase(regularized_loss, loss)
    """
    
class GaussianNoise2(Layer):
    # with variable noise
    def __init__(self, init_logvar, kdelayer, 
                 regularize_mi_input=None, 
                 init_alpha=1.0, 
                 get_noise_input_func=None, 
                 trainable=True,
                 *kargs, **kwargs):
        self.supports_masking = True
        self.init_logvar = init_logvar
        #self.uses_learning_phase = True
        self.kdelayer = kdelayer
        #self.regularizemi = regularizemi
        self.get_noise_input_func = get_noise_input_func
        if regularize_mi_input is not None:
            self.mi_regularizer = MIRegularizer(MIComputer(get_noise_input_func(regularize_mi_input), kdelayer=kdelayer, noiselayer=self),
                                               alpha=init_alpha)
        else:
            self.mi_regularizer = None
        self.logvar = K.variable(0.0)
        #self.init_alpha = init_alpha
        #self.alpha = K.variable(0.0)
        
        self.is_trainable = trainable
        
        super(GaussianNoise2, self).__init__(*kargs, **kwargs)
        
    def build(self, input_shape):
        super(GaussianNoise2, self).build(input_shape)
        K.set_value(self.logvar, self.init_logvar)
        #K.set_value(self.alpha, self.init_alpha)
        
        if self.is_trainable:
            self.trainable_weights = [self.logvar,]
        else:
            self.trainable_weights = []
            
        if self.mi_regularizer:
            self.regularizers.append(self.mi_regularizer)
        
    def get_noise(self, x):
        #if not hasattr(self, 'saved_noise'):
        #    self.saved_noise = K.random_normal(shape=K.shape(x), mean=0., std=1)
        return K.exp(0.5*self.logvar) * K.random_normal(shape=K.shape(x), mean=0., std=1)
    
    def call(self, x, mask=None):
        print self.input_spec
        return x+self.get_noise(x) # return K.in_train_phase(x+noise, x)

class KDEParamLayer(Layer):
    # with variable noise
    def __init__(self, init_logvar):
        self.init_logvar = init_logvar
        self.logvar = K.variable(0.0)
        super(KDEParamLayer, self).__init__()
        
    def build(self, input_shape):
        super(KDEParamLayer, self).build(input_shape)
        K.set_value(self.logvar, self.init_logvar)
        self.trainable_weights = []

    def call(self, x, mask=None):
        return x
        

#def kde_entropy_from_dists_loo(dists, N, dims, var):
#    # dists should have large values on diagonal
#    dists2 = dists / (2*var)
#    normconst = (dims/2.0)*K.log(2*np.pi*var)
#    lprobs = logsumexp(-dists2, axis=1) - np.log(N-1) - normconst
#    h = -K.mean(lprobs)
#    return h

        
      

#from keras.layers import Input
#from miregularizer2 import kde_entropy_from_dists_loo
from keras.callbacks import Callback
#import scipy
#import time

        
class NoiseTrain(Callback):
    def __init__(self, traindata, noiselayer):
        super(NoiseTrain, self).__init__()
        self.traindata = traindata
        self.noiselayer = noiselayer
        
    def on_train_begin(self, logs={}):
        modelobj = self.model.model
        inputs = modelobj.inputs + modelobj.targets + modelobj.sample_weights + [ K.learning_phase(),]
        lossfunc = K.function(inputs, [modelobj.total_loss])
        jacfunc  = K.function(inputs, K.gradients(modelobj.total_loss, self.noiselayer.logvar))
        sampleweights = np.ones(len(self.traindata.X))
        def obj(logvar):
            v = K.get_value(self.noiselayer.logvar)
            K.set_value(self.noiselayer.logvar, logvar.flat[0])
            r = lossfunc([self.traindata.X, self.traindata.Y, sampleweights, 1])[0]
            K.set_value(self.noiselayer.logvar, v)
            return r
        def jac(logvar):
            v = K.get_value(self.noiselayer.logvar)
            K.set_value(self.noiselayer.logvar, logvar.flat[0])
            r = np.atleast_2d(np.array(jacfunc([self.traindata.X, self.traindata.Y, sampleweights, 1])))[0]
            K.set_value(self.noiselayer.logvar, v)
            return r
            
        self.obj = obj # lambda logvar: lossfunc([self.traindata.X_train, self.traindata.Y_train, self.sampleweights, logvar[0], 1])[0]
        self.jac = jac # lambda logvar: np.array(jacfunc([self.traindata.X_train, self.traindata.Y_train, self.sampleweights, logvar[0], 1]))
    
    def on_epoch_begin(self, epoch, logs={}):
        r = scipy.optimize.minimize(self.obj, K.get_value(self.noiselayer.logvar), jac=self.jac)
        best_val = r.x[0]
        cval =  K.get_value(self.noiselayer.logvar)
        max_var = 1.0 + cval
        if best_val > max_var:
            # don't raise it too fast, so that gradient information is preserved 
            best_val = max_var
            
        K.set_value(self.noiselayer.logvar, best_val)
        #print 'noiseLV=%.5f' % K.get_value(self.noiselayer.logvar)
        
class KDETrain(Callback):
    def __init__(self, entropy_train_data, kdelayer, *kargs, **kwargs):
        super(KDETrain, self).__init__(*kargs, **kwargs)
        self.kdelayer = kdelayer
        self.entropy_train_data = entropy_train_data
        
    def on_train_begin(self, logs={}):
        self.nlayerinput = lambda x: K.function([self.model.layers[0].input], [self.kdelayer.input])([x])[0]
        N, dims = self.entropy_train_data.shape
        Kdists = K.placeholder(ndim=2)
        Klogvar = K.placeholder(ndim=0)
        def obj(logvar, dists):
            #print 'here', logvar # lossfunc([dists, logvar[0]])[0]
            return lossfunc([dists, logvar.flat[0]])[0]
        def jac(logvar, dists):
            #print logvar, lossfunc([dists, logvar[0]]), jacfunc([dists, logvar[0]])
            return np.atleast_2d(np.array(jacfunc([dists, logvar.flat[0]])))[0] 
            
        lossfunc = K.function([Kdists, Klogvar,], [kde_entropy_from_dists_loo(Kdists, N, dims, K.exp(Klogvar))])
        jacfunc  = K.function([Kdists, Klogvar,], K.gradients(kde_entropy_from_dists_loo(Kdists, N, dims, K.exp(Klogvar)), Klogvar))
        self.obj =obj #  lambda logvar, dists: np.array([lossfunc([dists, logvar[0]]),]) # [0]
        self.jac =jac # lambda logvar, dists: jacfunc([dists, np.array([logvar]).flat[0]])[0]

    @staticmethod
    def get_dists(output):
        N, dims = output.shape

        # Kernel density estimation of entropy
        y1 = output[None,:,:]
        y2 = output[:,None,:]

        dists = np.sum((y1-y2)**2, axis=2) 
        return dists
    
    def on_epoch_begin(self, epoch, logs={}):
        vals = self.nlayerinput(self.entropy_train_data)
        dists = self.get_dists(vals)
        dists += 10e20 * np.eye(dists.shape[0])
        r = scipy.optimize.minimize(self.obj, K.get_value(self.kdelayer.logvar).flat[0], 
                                    jac=self.jac, 
                                    args=(dists,),
                                    )
        best_val = r.x.flat[0]
        K.set_value(self.kdelayer.logvar, best_val)
        #print 'kdeLV=%.5f' % K.get_value(self.kdelayer.logvar)


class ReportVars(Callback):
    def __init__(self, kdelayer, noiselayer, *kargs, **kwargs):
        super(ReportVars, self).__init__(*kargs, **kwargs)
        self.noiselayer = noiselayer
        self.kdelayer = kdelayer
        
    def on_epoch_end(self, epoch, logs={}):
        lv1 = K.get_value(self.kdelayer.logvar)
        lv2 = K.get_value(self.noiselayer.logvar)
        logs['kdeLV']   = lv1
        logs['noiseLV'] = lv2
        print 'kdeLV=%.5f, noiseLV=%.5f' % (lv1, lv2)  

from utils import randsample
import tensorflow as tf
def get_logs(model, data, kdelayer, noiselayer, max_entropy_calc_N=None):
    logs = {}

    modelobj = model.model
    inputs = modelobj.inputs + modelobj.targets + modelobj.sample_weights + [ K.learning_phase(),]
    lossfunc = K.function(inputs, [modelobj.total_loss])
    sampleweightstrn = np.ones(len(data.train.X))
    sampleweightstst = np.ones(len(data.test.X))
    noreglosstrn = lambda: lossfunc([data.train.X, data.train.Y, sampleweightstrn, 0])[0]
    noreglosstst = lambda: lossfunc([data.test.X , data.test.Y , sampleweightstst, 0])[0]

    lv1 = K.get_value(kdelayer.logvar)
    lv2 = K.get_value(noiselayer.logvar)
    logs['kdeLV']   = lv1
    logs['noiseLV'] = lv2
    print 'kdeLV=%.5f, noiseLV=%.5f' % (lv1, lv2),
    
    
    if max_entropy_calc_N is None:
        mitrn = data.train.X
        mitst = data.test.X
    else:
        mitrn = randsample(data.train.X, max_entropy_calc_N)
        mitst = randsample(data.test.X, max_entropy_calc_N)
    
    mi_obj_trn = MIComputer(noiselayer.get_noise_input_func(mitrn), kdelayer=kdelayer, noiselayer=noiselayer)
    mi_obj_tst = MIComputer(noiselayer.get_noise_input_func(mitst), kdelayer=kdelayer, noiselayer=noiselayer)
    
    if True:
        mivals_trn = map(lambda x: float(K.eval(x)), [mi_obj_trn.get_mi(), mi_obj_trn.get_h(), mi_obj_trn.get_hcond()]) # [data.train.X,]))
        logs['mi_trn'] = mivals_trn[0]
        mivals_tst = map(lambda x: float(K.eval(x)), [mi_obj_tst.get_mi(), mi_obj_tst.get_h(), mi_obj_tst.get_hcond()]) # [data.train.X,]))
        logs['mi_tst'] = mivals_tst[0]
        logs['kl_trn'] = noreglosstrn()
        logs['kl_tst'] = noreglosstst()
        print ', mitrn=%s, mitst=%s, kltrn=%.3f, kltst=%.3f' % (mivals_trn, mivals_tst, logs['kl_trn'], logs['kl_tst'])
    else:
        print
        
    return logs
    #logs['tstloss'] = self.totalloss([self.xX_test,0])
        