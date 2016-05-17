from sklearn.decomposition import PCA, FastICA
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils

def plot_activity(activity, doPCA=True):
    cl = PCA if doPCA else FastICA
    X_reduced = cl(n_components=2).fit_transform(activity)
    plt.figure(figsize=(10,10))
    plt.scatter(X_reduced[:,0], X_reduced[:,1], s=.2,  edgecolor='none', c=y_train)
    
from keras.datasets import mnist

class Data(object):
    pass

def load_mnist():
    #(X_train, y_train), (X_test, y_test) = cifar10.load_data()
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
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