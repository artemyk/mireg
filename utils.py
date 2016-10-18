import numpy as np
from keras.utils import np_utils
from keras.layers.core import K
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA

def get_activations(model, layer, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output,])
    activations = get_activations([X_batch,0])
    return activations

def plot_activity(activity, colors=None, doPCA=True, size=.2, dims=2, opts={}):
    if dims not in [2,3]:
        raise Exception('dims must be in [2,3]')
    cl = PCA if doPCA else FastICA
    X_reduced = cl(n_components=dims, **opts).fit_transform(activity)
    #plt.figure(figsize=(10,10))
    kargs = [X_reduced[:,0], X_reduced[:,1]]
    if dims == 3:
        kargs.append(X_reduced[:,2])
    kwargs = dict(s=size,  edgecolor='none', c=colors)
    plt.gca().scatter(*kargs, **kwargs)
    
from keras.datasets import mnist

class Data(object):
    pass

def load_mnist(max_train_items=None, max_test_items=None):
    #(X_train, y_train), (X_test, y_test) = cifar10.load_data()
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    if max_train_items is not None:
        skip_every_trn = int(X_train.shape[0] / max_train_items)
        X_train = X_train[::skip_every_trn,:,:]
        y_train = y_train[::skip_every_trn]
    if max_test_items is not None:
        skip_every_tst = int(X_test.shape[0] / max_test_items)
        X_test  = X_test[::skip_every_tst,:,:]
        y_test  = y_test[::skip_every_tst]

    nb_classes = 10

    X_train = np.reshape(X_train, [X_train.shape[0], -1])
    X_train = X_train.astype('float32') / 255.0
    X_train -= X_train.mean(axis=0)[None,:]
    X_test  = np.reshape(X_test, [X_test.shape[0], -1])
    X_test  = X_test.astype('float32') / 255.0
    X_test -= X_test.mean(axis=0)[None,:]

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test  = np_utils.to_categorical(y_test, nb_classes)

    d=Data()
    d.X_train, d.X_test, d.Y_train, d.Y_test, d.y_train, d.y_test, d.nb_classes = \
      X_train, X_test, Y_train, Y_test, y_train, y_test, nb_classes
    
    return d


def load_mnist_rnn(max_train_items=None, max_test_items=None, normalize=True):
    #(X_train, y_train), (X_test, y_test) = cifar10.load_data()
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    if max_train_items is not None:
        skip_every_trn = int(X_train.shape[0] / max_train_items)
        X_train = X_train[::skip_every_trn,:,:]
        y_train = y_train[::skip_every_trn]
    if max_test_items is not None:
        skip_every_tst = int(X_test.shape[0] / max_test_items)
        X_test  = X_test[::skip_every_tst,:,:]
        y_test  = y_test[::skip_every_tst]


    X_train = X_train.astype('float32') / 255.0
    if normalize:
        X_train -= X_train.mean(axis=0)[None,:]
    X_test  = X_test.astype('float32') / 255.0
    if normalize:
        X_test -= X_test.mean(axis=0)[None,:]
    
    d=Data()
    d.X_train = X_train[:,8:,:].reshape([len(X_train), -1, 28*4])
    d.X_test  = X_test[:,8:,:].reshape([len(X_test), -1, 28*4])
    d.id_train = y_train
    d.id_test = y_test
    
    d.Y_train = d.X_train.copy()
    d.X_train = np.squeeze(d.X_train[:,0,:])

    d.Y_test = d.X_test.copy()
    d.X_test = np.squeeze(d.X_test[:,0,:])
    
    return d


