import tensorflow as tf
import os
import numpy as np
import pandas as pd
import time

import model.one_dimension_conv as conv1d
import data_processing.load_write as io
import data_processing.preprocessing as pp
import config
import train

os.environ['LD_LIBRARY_PATH'] = '/home/stu5/local/cuda10/lib64'
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth=True

def main():
    para = config.parameters
    with tf.Graph().as_default() as g:
        x = tf.placeholder(dtype=tf.float32, shape=(None, 150000, 1))
        is_train = tf.placeholder_with_default(False, (), 'is_train')

        model = conv1d.Conv1dModel(para=para)
        y_ = model.forward(x, is_train)

        saver = tf.train.Saver()
        with tf.Session(config=gpu_config) as sess:
            test_attrs, test_names = io.read_test_data(para['test_data_path'])
            label_mean, label_var = io.load_csv('./mean_std_label.csv', header=None)
            attr_mean, attr_var = io.load_csv('./mean_std.csv', header=None)

            norm_test_data_attr = []
            for test_data_attr in test_attrs:
                norm_test_data_attr.append(pp.norm(test_data_attr, attr_mean, attr_var))

            ckpt = tf.train.get_checkpoint_state(para['model_path'])

            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

                norm_test_data_attr = np.array(norm_test_data_attr).reshape((-1, 150000, 1))

                test_dict = {
                    x: norm_test_data_attr,
                    is_train: False
                }

                prediction = sess.run(y_, feed_dict=test_dict)
                prediction = pp.norm_restore(prediction, label_mean, label_var)
                result = np.concatenate((np.array(test_names, dtype=np.str).reshape(-1, 1),
                                         prediction.astype(dtype=np.str).reshape(-1, 1)), axis=1)
                io.output_csv('../result/test_result_5.csv', result)
                print('over')
            else:
                print('no model\n')


if __name__ == '__main__':
    main()
