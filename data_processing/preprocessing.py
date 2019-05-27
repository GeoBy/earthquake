import numpy as np
import multiprocessing as mp
from sklearn import preprocessing as skpp
from sklearn.model_selection import train_test_split
import data_processing.load_write as io
from scipy.fftpack import rfft
import matplotlib.pyplot as plt

def divide_data(data):
    divided_data_list = []
    data_start = 0
    data_end = 0
    for row in range(data.shape[0] - 1):
        current_label = data[row, 1]
        if row % 1000 == 0:
            print(row)
        if current_label < data[row + 1, 1]:
            print('file')
            data_end = row + 1
            divided_data_list.append(data[data_start:data_end, :])
            data_start = data_end
    return divided_data_list


def cut_train_data(data, cut_num):
    length = data.shape[0]
    cut_train_data_list = []
    cut_train_data_label_list = []
    for i in range(cut_num):
        rand_start = int(np.random.random_sample() * (length - 150000))
        rand_end = rand_start + 150000
        cut_train_data_list.append(data[rand_start:rand_end, 0])
        cut_train_data_label_list.append(data[rand_end - 1, 1])
    return cut_train_data_list, cut_train_data_label_list


def multi_process(func, cpu_kernel_num, args):
    pool = mp.Pool(cpu_kernel_num)
    pool.map(func, args)


def concate_list(array_list):
    arr = np.array(array_list)
    arr = arr.reshape([-1, 1])
    return arr


# def get_mean_stdv(data):
#     mean = np.mean(data, axis=0)
#     stdv = np.sqrt(np.var(data, axis=0))
#     return mean, stdv


def get_mean_var(data):
    scaler = skpp.StandardScaler(copy=False).fit(data)
    return scaler.mean_, np.sqrt(scaler.var_)


def norm(data, mean, std):
    return (data - mean) / std


def norm_restore(data, mean, std):
    return data * std + mean

def fourier_trans(data):
    z = rfft(data, axis=0)
    return z

if __name__ == '__main__':
    data = io.load_csv('../data/train/train_seg/4_4.6012981086.csv', header=None)
    f = fourier_trans(data)
    print(type(f))
    # x = np.arange(0, 150000)
    # plt.scatter(x, f)
    # plt.savefig('./f.png')
    # plt.show()
