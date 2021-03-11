import pandas as pd
import numpy as np
import os
import tarfile
import argparse


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


def read_raw_file(data_path):
    os.chdir(data_path)
    datalist = [i for i in os.listdir() if '.tar.gz' in i]
    first = True
    new_data_set = None
    for i in datalist:
        tar = tarfile.open(i, "r:gz")
        print(i)
        for member in tar.getmembers():
            f = tar.extractfile(member)
            if member.name.find("Truth") == -1:
                if f is not None:
                    label = []
                    omnet_id = []
                    label_s = int(member.name.split('/')[1].split('-')[3][1:])
                    omnet_ids = int(member.name.split('/')[1].split('-')[2])
                    t = pd.read_json(f.read(), lines=True)
                    t.fillna(0.0, inplace=True)
                    for _ in range(len(t)):
                        label.append(label_s)
                        omnet_id.append(omnet_ids)
                    split_data(t)
                    t['omnet_id'] = omnet_id
                    t['label'] = label
                    if not first:
                        new_data_set = pd.concat([new_data_set, t], axis=0, join='outer')
                    else:
                        new_data_set = t
                        first = False
    new_data_set.loc[new_data_set['label'] != 0, 'label'] = 1
    return new_data_set


def data_convert(data):  # 数据正则与空值填充
    columns = data.columns.tolist()
    for c in columns:
        if c != 'type' and c != 'omnet_id' and c != 'label':
            d = data[c]
            max_ = d.max()
            min_ = d.min()
            data[c] = ((d - min_) / (max_ - min_)).tolist()
    data.fillna(0.0, inplace=True)
    return data


if __name__ == "__main__":
    file_path = 'VeReMi-Dataset/000_Select_data'  # 数据集保存路径
    df = read_raw_file(file_path)
    df = df.drop(df.columns[df.max() == 0], axis=1)
    df = df.sort_values(by='rcvTime', ascending=True)
    df = data_convert(df)
    df.to_csv('dataset.csv')
