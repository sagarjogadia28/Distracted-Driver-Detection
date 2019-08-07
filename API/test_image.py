import numpy as np
import gc
import os
import glob
import cv2
import math
import pickle
import datetime
import pandas as pd
import time

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, \
									   ZeroPadding2D

# from keras.layers.normalization import BatchNormalization
# from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import model_from_json
# from sklearn.metrics import log_loss
from numpy.random import permutation
from keras import backend as K
K.set_image_dim_ordering('th')

np.random.seed(2016)
use_cache = 1
# color type: 1 - grey, 3 - rgb
color_type_global = 3
initialized = False


def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()


def merge_several_folds_geom(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a *= np.array(data[i])
    a = np.power(a, 1/nfolds)
    return a.tolist()


def normalize_test_data(img, img_rows=224, img_cols=224, color_type=1, filename="driver.jpg"):
	#print("checkpoint 3")
	test_data = []
	test_id = []
	#print(img)
	test_data.append(img)
	test_id.append(filename)
	# test_data, test_id = load_test(img, img_rows, img_cols, color_type)
	# cache_data((test_data, test_id), cache_path)
	# else:
	# print('Restore test from cache!')
	# (test_data, test_id) = restore_data(cache_path)

	test_data = np.array(test_data, dtype=np.uint8)
	#print (test_data[0])
	#print('checkpoint 2')
	if color_type == 1:
		test_data = test_data.reshape(test_data.shape[0], color_type,
									  img_rows, img_cols)
	else:
		test_data = test_data.transpose((0, 3, 1, 2))
		

	test_data = test_data.astype('float32')
	mean_pixel = [103.939, 116.779, 123.68]
	for c in range(3):
		test_data[:, c, :, :] = test_data[:, c, :, :] - mean_pixel[c]
	# test_data /= 255
	print('Test shape:', test_data.shape)
	print(test_data.shape[0], 'test samples')
	return test_data, test_id


def read_model(index, cross=''):
    json_name = 'architecture' + str(index) + cross + '.json'
    weight_name = 'model_weights' + str(index) + cross + '.h5'
    model = model_from_json(open(os.path.join('cache', json_name)).read())
    model.load_weights(os.path.join('cache', weight_name))
    return model


def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3',
                                                 'c4', 'c5', 'c6', 'c7',
                                                 'c8', 'c9'])
    result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)
    print (predictions)
    if not os.path.isdir('subm'):
        os.mkdir('subm')
    suffix = info
    sub_file = os.path.join('subm', 'submission_' + suffix + '.csv')
    result1.to_csv(sub_file, index=False)
    probability = predictions[0]
    l = len(probability)
    max_prob = 0.0
    index = -1
    for i in range(0, l):
    	# print (max_prob, i)
    	if max_prob <= probability[i]:
    		index = i
    		max_prob = probability[i]
    # c = ['safe driving', 'texting - right', 'talking on the phone - right', 'texting - left', 'talking on the phone - left', 'operating the radio', 'drinking', 'hair and makeup', 'talking to passenger', 'looing around']
    # print (c[index])
    return index



def test_model_and_submit(img, start=1, end=1, img_rows=224, img_cols=224, filename="driver.jpg", modelStr = '_vgg_16_2x20'):
	# img_rows, img_cols = 224, 224
	# batch_size = 64
	# random_state = 51
	nb_epoch = 15
	global initialized
	if initialized == False:
		return "Wait for some time for a system to initialize"
	print('Classifying image............')
	# print("Debugging")
	test_data, test_id = normalize_test_data(img, img_rows, img_cols,
													  color_type_global, filename)
	# print("passed")
	yfull_test = []
	global models
	print (len(models))
	for index in range(start - 1, end):
		# Store test predictions
		# model = read_model(index, modelStr)
		test_prediction = models[index].predict(test_data, batch_size=32, verbose=1)
		yfull_test.append(test_prediction)

	info_string = 'loss_' + modelStr \
				  + '_r_' + str(img_rows) \
				  + '_c_' + str(img_cols) \
				  + '_folds_' + str(end - start + 1) \
				  + '_ep_' + str(nb_epoch) \
				  + '_file_' + filename

	test_res = merge_several_folds_mean(yfull_test, end - start + 1)
	type_of_dist = create_submission(test_res, test_id, info_string)
	return type_of_dist


def load_models(start=1, end=1, modelStr=''):
	print ("loading models....")
	global models
	for index in range(start, end + 1):
		models.append(read_model(index, modelStr))
	print ("\n\n\n\n**********\n")
	print (len(models))
	print ("\n\n\n\n**********\n")
	# return models

models = []

# img = cv2.resize(img, (img_cols, img_rows))
# test_model_and_submit(cv2.resize(img, (img_cols, img_rows)), start, end, img_rows, img_cols, filename)
# gc.collect()
def init(start=1, end=1, modelStr=''):
	# print ("\n\n\n\n**********\n")
	# print ("cooleeed")
	# print ("\n\n\n\n**********\n")
	load_models(start, end, modelStr)
	global initialized
	initialized = True

print ("\n\n\n\n**********\n")
print ("Initializing required models...")
print ("\n\n\n\n**********\n")
start = 1
end = 3
modelStr = '_vgg_16_2x20'
init(start, end, modelStr)

