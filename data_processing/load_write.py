import numpy as np
import pandas as pd
import os
import time
import re
import multiprocessing as mp


def get_data_name_list(path, p=r'(.)*' + '.csv'):
    pattern = re.compile(p)
    data_names = []
    for data_name in os.listdir(path):
        if re.match(pattern, data_name):
            data_names.append(data_name)
    data_names = sorted(data_names)
    return data_names


def load_csv(path, header):
    train_df = pd.read_csv(path, header=header, index_col=None)
    train_data = train_df.values
    return train_data


def output_csv(path, data):
    # np.savetxt(path, data, delimiter=',', fmt=fmt)
    data_df = pd.DataFrame(data)
    data_df.to_csv(path, header=False, index=False)
    return 0


def join_path_name(path, name_list):
    for i in range(len(name_list)):
        name_list[i] = path + name_list[i]
    return name_list


def load_train_seg(path):
    train_seg = load_csv(path, header=None)
    p, train_seg_label = os.path.split(path)
    train_seg_label = float(train_seg_label.split('_')[-1][:-4])
    return train_seg, train_seg_label


def load_test_seg(path):
    test_seg = load_csv(path, header=0)
    p, seg_name = os.path.split(path)
    return test_seg, seg_name


def one_thread(train_data_path_list):
    train_data_list = []
    for train_data_path in train_data_path_list:
        train_seg, seg_name = load_train_seg(train_data_path)
        train_data_list.append((train_seg, seg_name))
    train_data_attr = [train_data[0] for train_data in train_data_list]
    train_data_label = [train_data[1] for train_data in train_data_list]
    return train_data_attr, train_data_label


def read_train_data_multi_processing(path, threads):
    train_data_name_list = get_data_name_list(path)
    train_data_name_list = join_path_name(path, train_data_name_list)
    thread_data_num = int(len(train_data_name_list) / threads)
    start_t = time.time()
    pool = mp.Pool(processes=threads)
    train_data_threads = pool.map(one_thread, [train_data_name_list[i: i + thread_data_num]
                                            for i in range(0, len(train_data_name_list), thread_data_num)])
    pool.close()
    pool.join()
    train_data_list=[]
    for train_data_thread in train_data_threads:
        train_data_list.extend(train_data_thread)

    train_data_attr = [train_data[0] for train_data in train_data_list]
    train_data_label = [train_data[1] for train_data in train_data_list]
    return train_data_attr, train_data_label

def read_train_data(path):
    train_data_name_list = get_data_name_list(path)
    train_data_name_list = join_path_name(path, train_data_name_list)
    train_data_list = []
    for train_data_path in train_data_name_list:
        train_attr, label= load_train_seg(train_data_path)
        train_data_list.append((train_attr, label))
        print(train_data_path)
    train_data_attr = [train_data[0] for train_data in train_data_list]
    train_data_label = [train_data[1] for train_data in train_data_list]
    return train_data_attr, train_data_label


def read_test_data(path):
    test_data_name_list = get_data_name_list(path)
    test_data_name_list = join_path_name(path, test_data_name_list)
    test_data_list = []
    for test_data_path in test_data_name_list:
        test_seg, seg_name = load_test_seg(test_data_path)
        test_data_list.append((test_seg, seg_name))
        print(test_data_path)
    test_data_attr = [test_data[0] for test_data in test_data_list]
    test_data_label = [test_data[1] for test_data in test_data_list]
    return test_data_attr, test_data_label


def test():
    train_data_attr, train_data_label = read_train_data('../data/train/train_seg/')
    print(len(train_data_attr[0]), train_data_label[0], train_data_attr[0])



if __name__ == '__main__':
    test()
