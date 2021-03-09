import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.python.keras import Input
from tensorflow.python.keras.callbacks import LearningRateScheduler
from tensorflow.python.keras.models import Sequential, Model, load_model
import pandas as pd
import numpy as np
import math


def data_load():
    data_path = 'VeReMi-Dataset/000_Select_data/timesorted_data/'
    train = pd.read_csv(data_path + 'train_set_time_sorted.csv')
    test = pd.read_csv(data_path + 'test_set_time_sorted.csv')
    x_train = train.iloc[:, 1:30]
    x_test = test.iloc[:, 1:30]
    y_train = train.iloc[:, 31]
    y_test = test.iloc[:, 31]
    del train
    del test
    return x_train, x_test, y_train, y_test


def AE_train(encoding_dim, x_train, epochs_num):
    # 编码层
    input_data = Input(shape=[29])
    encoded = Dense(24, activation='relu')(input_data)
    encoded = Dense(16, activation='relu')(encoded)
    encoded = Dense(8, activation='relu')(encoded)
    encoder_output = Dense(encoding_dim)(encoded)
    # 解码层
    decoded = Dense(8, activation='relu')(encoder_output)
    decoded = Dense(16, activation='relu')(decoded)
    decoded = Dense(24, activation='relu')(decoded)
    decoded = Dense(29, activation='tanh')(decoded)

    autoencoder = Model(inputs=input_data, outputs=decoded)
    encoder = Model(inputs=input_data, outputs=encoder_output)

    autoencoder.compile(optimizer='adam', loss='mse')

    def step_decay(epoch):
        initial_lrate = 0.01
        drop = 0.5
        epochs_drop = 10.0
        _lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
        return _lrate

    lrate = LearningRateScheduler(step_decay)

    history = autoencoder.fit(x_train, x_train, epochs=epochs_num, batch_size=256, callbacks=[lrate])

    loss = history.history['loss']
    epochs = range(1, epochs_num + 1)
    plt.title('Loss')
    plt.plot(epochs, loss, 'blue', label='loss')
    plt.legend()
    plt.show()
    encoder.save("encoder_model.h5")


def AE_predict(x_test, y_test):
    encoder = load_model("encoder_model.h5")
    encoded_data = encoder.predict(x_test)
    ndy = np.array(y_test)
    # 扩展一下标签维数然后合并
    ndy = ndy[:, np.newaxis]
    x = np.concatenate([encoded_data, ndy], axis=1)
    del ndy, encoded_data
    return x


def draw_AE_result(_x):
    plt.figure(figsize=(16, 8))  # 设置画布大小
    ax = plt.axes(projection='3d')  # 设置三维轴
    ax.scatter3D(_x[:, 0], _x[:, 1], _x[:, 2], c=_x[:, 3])
    plt.rcParams.update({'font.family': 'Times New Roman'})
    plt.rcParams.update({'font.weight': 'normal'})
    plt.rcParams.update({'font.size': 15})
    label_font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}
    plt.xlabel('X', label_font)
    plt.ylabel('Y', label_font, rotation=38)  # y 轴名称旋转 38 度
    ax.set_zlabel('Z', label_font)
    plt.savefig('3D.jpg', bbox_inches='tight', dpi=2400)  # 保存图片，如果不设置 bbox_inches='tight'，保存的图片有可能显示不全
    plt.show()


'''x_train, x_test, y_train, y_test = data_load()
AE_train(3, x_train, 10)
x = AE_predict(x_test, y_test)
print('x')'''
