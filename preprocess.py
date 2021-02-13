#%%

import pandas as pd
import numpy as np
import os
import tarfile

data_path='J:/VeReMi-Dataset/000_Select_data'
os.chdir(data_path)
datalist=[i for i in os.listdir(data_path) if '.tar.gz' in i]

df_list=[]
label=[]
OMNet_id=[]
for i in datalist:
    tar = tarfile.open(i, "r:gz")
    print(i)
    print(tar.getmembers)
    for member in tar.getmembers():
         f = tar.extractfile(member)
         if member.name.find("Truth")==-1:
             if f is not None:
                 label.append(int(member.name.split('/')[1].split('-')[3][1:]))
                 OMNet_id.append(int(member.name.split('/')[1].split('-')[2]))
                 t=pd.read_json(f.read(),lines=True)
                 df_list.append(t)

normal=[]
dos=[]
dataset=np.vstack((df_list,OMNet_id,label))

ratio_train = 0.8 #训练集比例
ratio_val = 0.1 #验证集比例
ratio_test = 0.1 #测试集比例
assert (ratio_train + ratio_val + ratio_val) == 1.0,'Total ratio Not equal to 1' ##检查总比例是否等于1
cnt_test = round(dataset.shape[1] * ratio_test ,0)
cnt_val = round(dataset.shape[1] * ratio_val ,0)
cnt_train = dataset.shape[1] - cnt_test - cnt_val
print("test Sample:" + str(cnt_test))
print("val Sample:" + str(cnt_val))
print("train Sample:" + str(cnt_train))

np.random.shuffle(dataset)
train_list=[[]]
val_list=[]
test_list=[]
t=dataset[:,1]
for i in range(int(cnt_train)):
    #np.append(train_list,dataset[:,i],axis=1)
    train_list.append(dataset[:,i])
for i in range(int(cnt_train) ,int(cnt_train + cnt_val)):
    val_list.append(dataset[:,i])

for i in range(int(cnt_train + cnt_val) ,int(cnt_train + cnt_val + cnt_test)):
    test_list.append(dataset[:,i])
print(1)
#del dataset
#gc.collect(dataset)
del(train_list[0])
del(val_list[0])
del(test_list[0])

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
    ori_id = np.asarray(data[2]).astype('float64')
    data = regularize(data[0])
    data[0].fillna(0.0, inplace=True)
    data_regularized = data[0].values
    label = []
    id = []
    print(len(data[0]))
    for _ in range(len(data[0])):
        label.append(ori_label[0])
        id.append(ori_id[0])
    for i in range(len(data)):
        if i != 0:
            data[i].fillna(0.0, inplace=True)
            try:
                data_regularized = np.vstack((data_regularized, data[i].values))
                for _ in range(len(data[i])):
                    label.append(ori_label[i])
                    id.append(ori_id[i])
            except ValueError:
                continue
    return np.array(data_regularized), np.array(id),np.array(label)

train = np.array(train_list).T
val = np.array(val_list).T
test = np.array(test_list).T

split_data(train)
split_data(val)
split_data(test)
trainX_regularized, trainid,trainY = data_convert(train)
valX_regularized, valid,valY = data_convert(val)
testX_regularized, testid,testY = data_convert(test)

trainX=np.row_stack((trainid,trainX_regularized))
valX=np.row_stack((valid,valX_regularized))
testX=np.row_stack((testid,testX_regularized))

np.save('testX_processed.npy',testX)
np.save('testY_processed.npy',testY)
np.save('trainX_processed.npy',trainX)
np.save('trainY_processed.npy',trainY)
np.save('valX_processed.npy',valX)
np.save('valY_processed.npy',valY)