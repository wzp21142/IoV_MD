from tensorflow.keras import *
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


def get_info(bucket_list, data_label_df):
    result = []
    for i in data_label_df:
        result.append(i[bucket_list])
    return result


def split_data(dataframes):
    for frame in dataframes[0]:
        for col in frame:
            try:
                for i in range(len(frame[col].values[0])):
                    t_list = []
                    for j in frame[col].values:
                        t_list.append(j[i])
                    frame[col + chr(88 + i)] = t_list
                frame.drop(col, axis=1, inplace=True)
            except TypeError:
                continue


def regularize(raw_data):
    frame_list = []
    for df in raw_data:
        new_dataframe = pd.DataFrame(index=df.index)
        columns = df.columns.tolist()
        for c in columns:
            if c != 'type':
                d = df[c]
                max_ = d.max()
                min_ = d.min()
                new_dataframe[c] = ((d - min_) / (max_ - min_)).tolist()
            else:
                new_dataframe[c] = df[c]
        frame_list.append(new_dataframe)
    return frame_list


def data_convert(data):  # 数据正则与空值填充
    ori_label = np.asarray(data[1]).astype('float64')
    data = regularize(data[0])
    data[0].fillna(0.0, inplace=True)
    data_regularized = data[0].values
    label = []
    for _ in range(len(data[0])):
        label.append(ori_label[0])
    for i in range(len(data)):
        if i != 0:
            data[i].fillna(0.0, inplace=True)
            try:
                data_regularized = np.vstack((data_regularized, data[i].values))
                for _ in range(len(data[i])):
                    label.append(ori_label[i])
            except ValueError:
                continue
    return np.array(data_regularized), np.array(label)


data_path = 'VeReMi-Dataset/000_Select_data'
os.chdir(data_path)
datalist = os.listdir()
test = np.load('test.npy', allow_pickle=True)
val = np.load('val.npy', allow_pickle=True)
train = np.load('train.npy', allow_pickle=True)
train = train.T
val = val.T
test = test.T
split_data(train)
split_data(val)
split_data(test)
trainX_regularized, trainY = data_convert(train)
valX_regularized, valY = data_convert(val)
testX_regularized, testY = data_convert(test)

model = Sequential()
model.add(layers.Dense(2, activation='softmax'))
model.compile(optimizer=optimizers.Adam(),
              # loss=keras.losses.CategoricalCrossentropy(),  # 需要使用to_categorical
              loss=losses.binary_crossentropy,
              metrics=['accuracy'])
history = model.fit(trainX_regularized, trainY, batch_size=64, epochs=10, validation_data=(valX_regularized, valY))
effect = model.evaluate(testX_regularized, testY)
