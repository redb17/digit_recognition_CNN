from flask import Flask, render_template, request
import numpy as numpy
from scipy.misc import imsave, imread, imresize
import keras.models
import re
import sys
import os
import base64
import tensorflow as tf
from keras.optimizers import RMSprop

sys.path.append(os.path.abspath('./model'))
from load import *

app = Flask(__name__)

global model
model, graph = init()

def convertImg(imgData1):
	imgstr = re.search(r'base64,(.*)',imgData1.decode("utf-8")).group(1)
	print(type(imgstr))
	with open('out.png','wb') as output:
		output.write(base64.decodebytes(imgstr.encode('utf-8')))

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/predict/', methods=['GET', 'POST'])
def predict():
	imgData = request.get_data()
	convertImg(imgData)

	x = imread('out.png', mode='L')
	x = np.invert(x)
	x = imresize(x, (28, 28))
	x = x.reshape(-1, 28, 28, 1)
	
	with graph.as_default():
		out = model.predict(x)
		response = np.argmax(out)
		return str(response)

if __name__ == '__main__':
	port = int(os.environ.get('PORT', 8000))
	app.run(port=port)