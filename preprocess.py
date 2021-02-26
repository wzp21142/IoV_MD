import pandas as pd
import numpy as np
import os
import tarfile


def read_raw_file(data_path):
    os.chdir(data_path)
    datalist = [i for i in os.listdir() if '.tar.gz' in i]
    first = True
    dataset = None
    for i in datalist:
        tar = tarfile.open(i, "r:gz")
        print(i)
        print(tar.getmembers)
        for member in tar.getmembers():
            f = tar.extractfile(member)
            if member.name.find("Truth") == -1:
                if f is not None:
                    label = []
                    OMNet_id = []
                    label_s = int(member.name.split('/')[1].split('-')[3][1:])
                    OMNet_id_s = int(member.name.split('/')[1].split('-')[2])
                    t = pd.read_json(f.read(), lines=True)
                    for _ in range(len(t)):
                        label.append(label_s)
                        OMNet_id.append(OMNet_id_s)
                    split_data(t)
                    t['OMNet_id'] = OMNet_id
                    t['label'] = label
                    if not first:
                        dataset = pd.concat([dataset, t], axis=0, join='inner')
                    else:
                        dataset = t
                        first = False
    return dataset


def split_data(dataframe):
    for col in dataframe:
        try:
            for i in range(len(dataframe[col].values[0])):
                t_list = []
                for j in dataframe[col].values:
                    t_list.append(j[i])
                dataframe[col + chr(88 + i)] = t_list
            dataframe.drop(col, axis=1, inplace=True)
        except TypeError:
            continue


def data_convert(data):  # 数据正则与空值填充
    columns = data.columns.tolist()
    for c in columns:
        if c != 'type' and c != 'OMNet_id' and c != 'label':
            d = data[c]
            max_ = d.max()
            min_ = d.min()
            data[c] = ((d - min_) / (max_ - min_)).tolist()
    data.fillna(0.0, inplace=True)
    return np.array(data.iloc[:, :-1]), np.array(data.iloc[:, -1:])


data_path = 'VeReMi-Dataset/000_Select_data'
dataset = read_raw_file(data_path)

ratio_train = 0.8  # 训练集比例
ratio_val = 0.1  # 验证集比例
ratio_test = 1 - ratio_train - ratio_val  # 测试集比例
train_set = dataset.sample(frac=ratio_train, replace=False)
val_set = dataset.sample(frac=ratio_val, replace=False)
test_set = dataset.sample(frac=ratio_test, replace=False)
print("train sample ratio: " + str(ratio_train))
print("val sample ratio: " + str(ratio_val))
print("test sample ratio: " + str(ratio_test))

trainX, trainY = data_convert(train_set)
valX, valY = data_convert(val_set)
testX, testY = data_convert(test_set)

np.save('trainX_processed.npy', trainX)
np.save('trainY_processed.npy', trainY)
np.save('valX_processed.npy', valX)
np.save('valY_processed.npy', valY)
np.save('testX_processed.npy', testX)
np.save('testY_processed.npy', testY)
