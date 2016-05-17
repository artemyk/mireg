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
        logs['logvar'] = float(K.get_value(self.model.layers[-2].logvar))


alphavals = [0,]+list(10**np.linspace(-2,1,8))
HIDDEN_DIM = 20
ndx=0
numepochs = 75

for alpha in alphavals:
    for trainablevar in [False, True]:
        for initlogvar in np.linspace(-5, 5, 8):
            for add_noise in [True, False]:
                for entropy_only in [True, False]:
                    
                    prms = dict(alpha=alpha, trainablevar=trainablevar,initlogvar=initlogvar, add_noise=add_noise, entropy_only=entropy_only)
                    s = "_".join(['%s=%s'%(k,v) for k,v in prms.iteritems()])
                    print "Doing", s
                    
                    model = Sequential()
                    model.add(Dense(2*HIDDEN_DIM, input_dim=d.X_train.shape[1], activation='tanh'))
                    model.add(Dense(HIDDEN_DIM, input_dim=d.X_train.shape[1], activation='tanh'))
                    model.add(MILayer(HIDDEN_DIM, **prms))
                    model.add(Dense(d.nb_classes, activation='softmax'))

                    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
                    
                    r=model.fit(d.X_train, d.Y_train, nb_epoch=numepochs, 
                                batch_size=500, 
                                validation_split=0.1, callbacks=[LogvarHistory(),]) # , verbose=1)
                    cdata = {'params':prms, 'results':r.history}
                    with open('results/model-%s.json'%s, 'w') as outfile:
                        json.dump(cdata, outfile)
                    

