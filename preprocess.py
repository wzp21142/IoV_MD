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
    parser = argparse.ArgumentParser("Demo of argparse")

    parser.add_argument('-s', type=int, default=1, help="option of sorting the dataset by rcvTime")
    args = parser.parse_args()
    print(args)

    ratio_train = 0.8  # 训练集比例
    ratio_val = 0.1  # 验证集比例
    ratio_test = 1 - ratio_train - ratio_val  # 测试集比例
    file_path = 'VeReMi-Dataset/000_Select_data'  # 数据集保存路径
    dataset = read_raw_file(file_path)

    if args.s:
        data_set = dataset.sort_values(by='rcvTime', ascending=True)
        train_set = dataset[0: int(len(data_set)*ratio_train)].copy()
        val_set = dataset[int(len(data_set)*ratio_train): int(len(data_set)*(ratio_train+ratio_val))].copy()
        test_set = dataset[int(len(data_set)*(ratio_train+ratio_val)): len(data_set)].copy()
    else:
        train_set = dataset.sample(frac=ratio_train, replace=False)
        val_set = dataset.sample(frac=ratio_val, replace=False)
        test_set = dataset.sample(frac=ratio_test, replace=False)

    train_set.pop('rcvTime')
    val_set.pop('rcvTime')
    test_set.pop('rcvTime')
    print("train sample ratio: " + str(ratio_train))
    print("val sample ratio: " + str(ratio_val))
    print("test sample ratio: " + str(ratio_test) + '\n')

    train_set = data_convert(train_set)
    val_set = data_convert(val_set)
    test_set = data_convert(test_set)

    if args.s:
        prefix = "timesorted_data/"
        suffix = "_time_sorted"
    else:
        prefix = "process_data/"
        suffix = ""
    if not os.path.exists(prefix):
        os.mkdir(prefix)
    train_set.to_csv(prefix + 'train_set' + suffix + '.csv')
    val_set.to_csv(prefix + 'val_set' + suffix + '.csv')
    test_set.to_csv(prefix + 'test_set' + suffix + '.csv')
    print("train label ratio: " + str(len(train_set[train_set['label'] == 1])/len(train_set)))
    print("val label ratio: " + str(len(val_set[val_set['label'] == 1])/len(val_set)))
    print("test label ratio: " + str(len(test_set[test_set['label'] == 1])/len(test_set)))
