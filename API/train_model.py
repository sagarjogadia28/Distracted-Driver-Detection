# -*- coding: utf-8 -*-

import numpy as np

import os
import glob
import cv2
import math
import pickle
import datetime
import pandas as pd

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

# color_type = 1 - gray
# color_type = 3 - RGB


def get_im(path, img_rows, img_cols, color_type=1):
    # Load as grayscale
    if color_type == 1:
        img = cv2.imread(path, 0)
    elif color_type == 3:
        img = cv2.imread(path)
    # Reduce size
    resized = cv2.resize(img, (img_cols, img_rows))
    # mean_pixel = [103.939, 116.799, 123.68]
    # resized = resized.astype(np.float32, copy=False)

    # for c in range(3):
    #    resized[:, :, c] = resized[:, :, c] - mean_pixel[c]
    # resized = resized.transpose((2, 0, 1))
    # resized = np.expand_dims(img, axis=0)
    return resized


def get_driver_data():
    dr = dict()
    path = os.path.join('driver_imgs_list.csv')
    print('Read drivers data')
    f = open(path, 'r')
    headers = f.readline().strip()
    for line in f.readlines():
        driver_id, class_id, img_id = line.strip().split(',')
        dr[img_id] = driver_id
    f.close()
    return dr


def load_train(img_rows, img_cols, color_type=1):
    X_train = []
    y_train = []
    driver_id = []

    driver_data = get_driver_data()

    print('Read train images')
    for j in range(10):
        print('Load folder c{}'.format(j))
        path = os.path.join('train',
                            'c' + str(j), '*.jpg')
        files = glob.glob(path)
        for fl in files:
            flbase = os.path.basename(fl)
            img = get_im(fl, img_rows, img_cols, color_type)
            X_train.append(img)
            y_train.append(j)
            driver_id.append(driver_data[flbase])

    unique_drivers = sorted(list(set(driver_id)))
    print('Unique drivers: {}'.format(len(unique_drivers)))
    print(unique_drivers)
    return X_train, y_train, driver_id, unique_drivers


def load_test(img_rows, img_cols, color_type=1):
    print('Read test images')
    path = os.path.join('test', '*.jpg')
    files = glob.glob(path)
    X_test = []
    X_test_id = []
    total = 0
    thr = math.floor(len(files)/10)
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im(fl, img_rows, img_cols, color_type)
        X_test.append(img)
        X_test_id.append(flbase)
        total += 1
        if total % thr == 0:
            print('Read {} images from {}'.format(total, len(files)))

    return X_test, X_test_id


def cache_data(data, path):
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    if os.path.isdir(os.path.dirname(path)):
        file = open(path, 'wb')
        pickle.dump(data, file)
        file.close()
    else:
        print('Directory doesnt exists')


def restore_data(path):
    data = dict()
    if os.path.isfile(path):
        print('Restore data from pickle........')
        file = open(path, 'rb')
        data = pickle.load(file)
    return data


def save_model(model, index, cross=''):
    json_string = model.to_json()
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    json_name = 'architecture' + str(index) + cross + '.json'
    weight_name = 'model_weights' + str(index) + cross + '.h5'
    open(os.path.join('cache', json_name), 'w').write(json_string)
    model.save_weights(os.path.join('cache', weight_name), overwrite=True)


def read_model(index, cross=''):
    json_name = 'architecture' + str(index) + cross + '.json'
    weight_name = 'model_weights' + str(index) + cross + '.h5'
    model = model_from_json(open(os.path.join('cache', json_name)).read())
    model.load_weights(os.path.join('cache', weight_name))
    return model


def split_validation_set(train, target, test_size):
    random_state = 51
    X_train, X_test, y_train, y_test = \
        train_test_split(train, target,
                         test_size=test_size,
                         random_state=random_state)
    return X_train, X_test, y_train, y_test


def get_accuracy(predictions, test_id):
    filename = 'driver_imgs_list'
    f = open(filename, 'r')
    headers = f.strip().split(',')
    true_results = []
    for line in f.readlines():
        driver_id, class_id, img_id = line.strip().split(',')
        class_id = int(class_id[1])
        true_results[img_id] = class_id

    prediction_results = []
    true_target = []
    j = 0
    for prediction in predictions:
        probability = prediction[0]
        l = len(probability)
        max_prob = 0.0
        index = -1
        for i in range(0, l):
            # print (max_prob, i)
            if max_prob <= probability[i]:
                index = i
                max_prob = probability[i]
        prediction_results.append(index)
        true_target.append(true_results[test_id[j]])
        j += 1

    prediction_results = np.array(prediction_results)
    true_target = np.array(true_target)
    accuracy = (prediction_results == true_target)
    return accuracy.mean()



def read_and_normalize_and_shuffle_train_data(img_rows, img_cols,
                                              color_type=1):

    cache_path = os.path.join('cache', 'train_r_' + str(img_rows) +
                              '_c_' + str(img_cols) + '_t_' +
                              str(color_type) + '.dat')

    if not os.path.isfile(cache_path) or use_cache == 0:
        train_data, train_target, driver_id, unique_drivers = \
            load_train(img_rows, img_cols, color_type)
        cache_data((train_data, train_target, driver_id, unique_drivers),
                   cache_path)
    else:
        print('Restore train from cache!')
        (train_data, train_target, driver_id, unique_drivers) = \
            restore_data(cache_path)

    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)

    if color_type == 1:
        train_data = train_data.reshape(train_data.shape[0], color_type,
                                        img_rows, img_cols)
    else:
        train_data = train_data.transpose((0, 3, 1, 2))

    train_target = np_utils.to_categorical(train_target, 10)
    train_data = train_data.astype('float32')
    mean_pixel = [103.939, 116.779, 123.68]
    for c in range(3):
        train_data[:, c, :, :] = train_data[:, c, :, :] - mean_pixel[c]
    # train_data /= 255
    perm = permutation(len(train_target))
    train_data = train_data[perm]
    train_target = train_target[perm]
    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target, driver_id, unique_drivers


def read_and_normalize_test_data(img_rows=224, img_cols=224, color_type=1):
    cache_path = os.path.join('cache', 'test_r_' + str(img_rows) +
                              '_c_' + str(img_cols) + '_t_' +
                              str(color_type) + '.dat')
    if not os.path.isfile(cache_path) or use_cache == 0:
        test_data, test_id = load_test(img_rows, img_cols, color_type)
        cache_data((test_data, test_id), cache_path)
    else:
        print('Restore test from cache!')
        (test_data, test_id) = restore_data(cache_path)

    test_data = np.array(test_data, dtype=np.uint8)

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


def dict_to_list(d):
    ret = []
    for i in d.items():
        ret.append(i[1])
    return ret


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


def copy_selected_drivers(train_data, train_target, driver_id, driver_list):
    data = []
    target = []
    index = []
    for i in range(len(driver_id)):
        if driver_id[i] in driver_list:
            data.append(train_data[i])
            target.append(train_target[i])
            index.append(i)
    data = np.array(data, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    index = np.array(index, dtype=np.uint32)
    return data, target, index


def vgg_std16_model(img_rows, img_cols, color_type=1):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(color_type,
                                                 img_rows, img_cols)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    model.load_weights('weights/vgg16_weights.h5')
    # model.load_model('weights/vgg16_weights.h5')

    # Code above loads pre-trained data and
    model.layers.pop()
    model.add(Dense(10, activation='softmax'))
    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    return model


def train_model(nfolds=10, nb_epoch=10, split=0.2, modelStr=''):

    # Now it loads color image
    # input image dimensions
    img_rows, img_cols = 224, 224
    batch_size = 32
    random_state = 20

    train_data, train_target, driver_id, unique_drivers = \
        read_and_normalize_and_shuffle_train_data(img_rows, img_cols,
                                                  color_type_global)

    # ishuf_train_data = []
    # shuf_train_target = []
    # index_shuf = range(len(train_target))
    # shuffle(index_shuf)
    # for i in index_shuf:
    #     shuf_train_data.append(train_data[i])
    #     shuf_train_target.append(train_target[i])

    # yfull_train = dict()
    # yfull_test = []
    num_fold = 0
    kf = KFold(len(unique_drivers), n_folds=nfolds,
               shuffle=True, random_state=random_state)
    for train_drivers, test_drivers in kf:
        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        # print('Split train: ', len(X_train), len(Y_train))
        # print('Split valid: ', len(X_valid), len(Y_valid))
        # print('Train drivers: ', unique_list_train)
        # print('Test drivers: ', unique_list_valid)
        # model = create_model_v1(img_rows, img_cols, color_type_global)
        # model = vgg_bn_model(img_rows, img_cols, color_type_global)
        model = vgg_std16_model(img_rows, img_cols, color_type_global)

        model.fit(train_data, train_target, batch_size=batch_size,
                  nb_epoch=nb_epoch,
                  show_accuracy=True, verbose=1,
                  validation_split=split, shuffle=True)

        # print('losses: ' + hist.history.losses[-1])

        # print('Score log_loss: ', score[0])

        save_model(model, num_fold, modelStr)
        del model
        # predictions_valid = model.predict(X_valid, batch_size=128, verbose=1)
        # score = log_loss(Y_valid, predictions_valid)
        # print('Score log_loss: ', score)
        # Store valid predictions
        # for i in range(len(test_index)):
        #    yfull_train[test_index[i]] = predictions_valid[i]

    print('Start testing model............')
    test_data, test_id = read_and_normalize_test_data(img_rows, img_cols,
                                                      color_type_global)
    yfull_test = []

    for index in range(1, num_fold + 1):
        # 1,2,3,4,5
        # Store test predictions
        model = read_model(index, modelStr)
        test_prediction = model.predict(test_data, batch_size=128, verbose=1)
        yfull_test.append(test_prediction)
        del model

    info_string = 'loss_' + modelStr \
                  + '_r_' + str(img_rows) \
                  + '_c_' + str(img_cols) \
                  + '_folds_' + str(nfolds) \
                  + '_ep_' + str(nb_epoch)

    test_res = merge_several_folds_mean(yfull_test, nfolds)
    create_submission(test_res, test_id, info_string)
    accuracy = get_accuracy(test_res, test_id)
    return accuracy


# nfolds, nb_epoch, split
# train_model(8, 6, 0.15, '_vgg_16_2x20')
