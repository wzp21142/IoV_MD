# %%

from tensorflow.keras import *
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tensorflow as tf


def get_info(bucket_list, data_array):
    info_list = ['type', 'rcvTime', 'pos', 'pos_noise', 'spd', 'spd_noise', 'acl', 'acl_noise', 'hed', 'hed_noise',
                 'sendTime', 'sender', 'senderPseudo']
    pos_list = [0, 1, [2, 3, 4], [5, 6, 7], [8, 9, 10], [11, 12, 13], [14, 15, 16], [17, 18, 19], [20, 21, 22],
                [23, 24, 25], 26, 27, 28, 29]
    bucket_pos = []
    for loc_buc in range(len(bucket_list)):
        for loc_info in range(len(info_list)):
            if bucket_list[loc_buc] == info_list[loc_info]:
                if isinstance(pos_list[loc_info], int):
                    bucket_pos.append(pos_list[loc_info])
                else:
                    for t in pos_list[loc_info]:
                        bucket_pos.append(t)
    result = np.empty([data_array.shape[0], len(bucket_pos)], dtype=np.float64)
    col_cnt = 0
    for i in bucket_pos:
        row_cnt = 0
        for j in data_array:
            result[row_cnt][col_cnt] = j[i]
            row_cnt += 1
        col_cnt += 1
    return result


def learn(model, data_path):
    os.chdir(data_path)

    testX_regularized = np.load('testX_processed.npy')
    testY = np.load('testY_processed.npy')
    trainX_regularized = np.load('trainX_processed.npy')
    trainY = np.load('trainY_processed.npy')
    valX_regularized = np.load('valX_processed.npy')
    valY = np.load('valY_processed.npy')

    trainY = tf.one_hot(trainY, 2)
    valY = tf.one_hot(valY, 2)
    testY_m = tf.one_hot(testY, 2)
    history = model.fit(trainX_regularized, trainY, batch_size=64, epochs=3, validation_data=(valX_regularized, valY))
    effect = model.evaluate(testX_regularized, testY_m)
    # testX_regularized
    y_pred = model.predict(testX_regularized, batch_size=1)
    y_pred_label = []
    for i in y_pred:
        if i[0] > i[1]:
            y_pred_label.append(0.0)
        else:
            y_pred_label.append(13.0)
    right = 0
    for loc in range(223972):
        if testY[loc] == y_pred_label[loc] * 1.0:
            right += 1
    acc = right / 223972
    print(acc)
