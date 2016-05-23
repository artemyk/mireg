import os
os.environ['KERAS_BACKEND']='theano'

import numpy as np
import keras.backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Activation


import logging    
logging.getLogger('keras').setLevel(logging.INFO)

from utils import *
from miregularizer import *

d=load_mnist()
import json
import keras.callbacks
class LogvarHistory(keras.callbacks.Callback):
    def on_epoch_end(self, batch, logs={}):
        if self.model.milayer is not None:
            v = float(K.get_value(self.model.layers[-2].logvar))
        else:
            v = 0.
        logs['logvar'] = v

def dorun(d, model, milayer, Wsave, prms, numepochs=70):
    s = "_".join(['%s=%s'%(k,v) for k,v in prms.iteritems()])
        
    fname = 'results2/%s-%s.json'%(model.modelname,s)
    if os.path.isfile(fname):
        print "%s already exists, continuing" % fname
        
    print "Doing", model.modelname, s
    
    milayer.set_myparams(**prms)
    model.set_weights(Wsave)

    r=model.fit(d.X_train, d.Y_train, nb_epoch=numepochs, 
                batch_size=500, 
                validation_split=0.1, callbacks=[LogvarHistory(),]) # , verbose=1)
    cdata = {'params':prms, 'results':r.history, 'backend':K._BACKEND}
    with open(fname, 'w') as outfile:
        json.dump(cdata, outfile)
        
        
def getmodel(milayer, HIDDEN_DIM, INPUT_DIM, modelname='undefined',hidden_layers=1):
    model = Sequential()
    model.modelname = modelname
    #model.add(Dense(2*HIDDEN_DIM, input_dim=INPUT_DIM, activation='tanh'))
    for _ in range(hidden_layers):
        model.add(Dense(HIDDEN_DIM,input_dim=INPUT_DIM, activation='tanh'))
    if milayer is not None:
        model.add(milayer)
    model.milayer = milayer
    model.add(Dense(d.nb_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

alphavals = [0,]+list(10**np.linspace(-2,2,8))
HIDDEN_DIM = 20
ndx=0


#****************
HIDDEN_DIM = 100
milayer = MILayer(HIDDEN_DIM, alpha=.1, initlogvar=-2., add_noise=False, entropy_only=False)
#milayer = None
model = getmodel(milayer, HIDDEN_DIM, d.X_train.shape[1], hidden_layers=2)
r=model.fit(d.X_train, d.Y_train, nb_epoch=150, 
            batch_size=1000, 
            validation_split=0.1, callbacks=[LogvarHistory(),]) # , verbose=1)
#****************

asdfsadf
# Trainable
initprms = dict(alpha=0.0, initlogvar=-2., add_noise=False, entropy_only=False)
milayer = MILayerTrainable(HIDDEN_DIM, **initprms)
model = getmodel(milayer, HIDDEN_DIM, d.X_train.shape[1], modelname='trainable')
Wsave = model.get_weights()

initlogvar = 0.0
for alpha in alphavals:
    for add_noise in [True, False]:
        for entropy_only in [True, False]:
            prms = dict(alpha=alpha, initlogvar=initlogvar, add_noise=add_noise, entropy_only=entropy_only)
            dorun(d, model, milayer, Wsave, prms)
del model, milayer, Wsave




# not trainable
initprms = dict(alpha=0.0, initlogvar=-2., add_noise=False, entropy_only=False)
milayer = MILayer(HIDDEN_DIM, **initprms)
model = getmodel(milayer, HIDDEN_DIM, d.X_train.shape[1], modelname='trainable')
Wsave = model.get_weights()

for alpha in alphavals:
    for initlogvar in np.linspace(-4, 4, 5):
        for add_noise in [True, False]:
            for entropy_only in [True, False]:
                prms = dict(alpha=alpha, initlogvar=initlogvar, add_noise=add_noise, entropy_only=entropy_only)
                dorun(d, model, milayer, Wsave, prms)
                