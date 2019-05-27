import numpy as np
import pandas as pd
import re
import os
from scipy.fftpack import rfft
import matplotlib.pyplot as plt


class DataSegList:
    def __init__(self, path):
        self.path = path
        self.data_name_list = get_data_name_list(self.path)
        self.data_array_list = []
        self.norm_array_lisy = []

    def output(self, output_path):
        if self.data_array_list:
            for data_array in self.data_array_list:
                output_csv(output_path, data_array)
        else:
            print('no array yet')

    def load_as_arrays(self):
        for i in range(len(self.data_name_list)):
            p = os.path.join(self.path, self.data_name_list[i])
            self.data_array_list.append(load_csv(p, header=None))

class DataSeg:
    def __init__(self, path):
        self.data = load_csv(path)
        self.frequent_zone = None
        self.data_size = self.data.shape[0]

    def fft(self):
        self.frequent_zone =  rfft(self.data, axis=0)

    def plt_time_zone(self, path):
        x = np.arange(0, self.data_size)
        plt.plot(x, self.data)
        plt.savefig(path)
        plt.show()

    def plt_frequency_zone(self, path):
        x = np.arange(0, self.frequent_zone.shape[0])
        plt.plot(x, self.frequent_zone)
        plt.savefig(path)
        plt.show()



def output_csv(output_path, data):
    data_df = pd.DataFrame(data)
    data_df.to_csv(output_path, header=False, index=False)
    return 0

def get_data_name_list(path, p=r'(.)*' + '.csv'):
    pattern = re.compile(p)
    data_names = []
    for data_name in os.listdir(path):
        if re.match(pattern, data_name):
            data_names.append(data_name)
        data_names = sorted(data_names)
    return data_names

def load_csv(path, header=None):
    train_df = pd.read_csv(path, header=header, index_col=None)
    train_data = train_df.values
    return train_data