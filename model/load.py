import numpy as np 
from scipy.misc import imread, imresize, imshow

from keras.models import load_model
from keras import backend
from keras.optimizers import RMSprop

import tensorflow as tf
import os

def init():
	model = load_model('model/Digit_Recognition_Kaggle_CNN_099371_Model.h5')
	model.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', \
				  metrics=['accuracy'])
	graph = tf.get_default_graph()
	return model, graph
