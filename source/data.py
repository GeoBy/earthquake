import numpy as np
import pandas as pd
import threading
import config
import multiprocessing
from data_processing.load_write import *
from data_processing.preprocessing import *
import random
import shutil


def divide_train():
    para = config.parameters
    # train_data = np.loadtxt(para['data_path'], dtype=np.float, delimiter=',', skiprows=1)
    train_df = pd.read_csv(para['data_path'])
    train_data = train_df.values

    divided_train_data_list = divide_data(train_data)
    file_count = 0
    for divided_train_data in divided_train_data_list:
        output_csv(os.path.join('../data/train/', str(file_count) + '.csv'),
                   divided_train_data)
        file_count += 1
    return 0


def cal_test_shape():
    para = config.parameters
    test_path = para["test_data_path"]
    test_data_list = get_data_name_list(test_path)
    row_list = []
    for test_data_name in test_data_list:
        test_data = load_csv(os.path.join(test_path, test_data_name))
        # row_list.append(test_data.shape)
        print(test_data.shape)
    # with open("./row_info.txt", mode='w+') as row_info_file:
    #     for row in row_list:
    #         row_info_file.write(str(row[0]))
    #         row_info_file.write('\n')
    #     row_info_file.close()


def data_load():
    para = config.parameters
    train_path = para["data_path"]
    train_data_name_list = get_data_name_list(train_path)
    train_data_dict = {}
    for train_data_name in train_data_name_list[:1]:
        train_data_circle = load_csv(os.path.join(train_path, train_data_name))
        circle_length = train_data_circle.shape[0]
        individual_probability = 1 / (circle_length / 150000)
        cut_num = 100
        while np.power((1 - individual_probability), cut_num) > 0.05:
            cut_num += 1
        train_data_segments, train_data_labels = cut_train_data(train_data_circle, cut_num)
        for cut_count in range(cut_num):
            output_csv(os.path.join(train_path,
                                    'train_seg',
                                    train_data_name[:-4] + '_' + str(train_data_labels[cut_count]) + '.csv'),
                       train_data_segments[cut_count])


def devide_seg(data_path):
    path, name = os.path.split(data_path)
    train_data_circle = load_csv(data_path)
    circle_length = train_data_circle.shape[0]
    individual_probability = 1 / (circle_length / 150000)
    cut_num = 100
    while np.power((1 - individual_probability), cut_num) > 0.05:
        cut_num += 1
    train_data_segments, train_data_labels = cut_train_data(train_data_circle, cut_num)
    for cut_count in range(cut_num):
        output_csv(os.path.join(path,
                                'train_seg',
                                name[:-4] + '_' + str(train_data_labels[cut_count]) + '.csv'),
                   train_data_segments[cut_count])
        print(name[:-4] + '_' + str(train_data_labels[cut_count]) + '.csv',
              multiprocessing.current_process().name, 'finished')


def main():
    para = config.parameters
    data_name_list = get_data_name_list(para['data_path'])
    index = [i for i in range(len(data_name_list))]
    random.shuffle(index)
    data_name_list = [data_name_list[i] for i in index]
    end_index = int(len(data_name_list) / 4)
    for data_name in data_name_list[:end_index]:
        source_path = os.path.join(para['data_path'], data_name)
        destination_path = para['validation_path']
        shutil.move(source_path, destination_path)

# def main():
#     train_data_attr, train_data_label = read_train_data('../data/train/train_seg/')
#     test_data_attr, test_data_name = read_test_data('../data/test/')


if __name__ == '__main__':
    attr_mean, attr_var = io.load_csv('./mean_std.csv', header=None)
    label_mean, label_var = io.load_csv('./mean_std_label.csv', header=None)
    print(attr_mean, attr_var)
    print(label_mean, label_var)
