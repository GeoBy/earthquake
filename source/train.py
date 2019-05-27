import tensorflow as tf
import numpy as np
import os
import time
import shutil
import random
import matplotlib.pyplot as plt

import model.one_dimension_conv as conv_1d
import data_processing.load_write as io
import data_processing.preprocessing as pp
import config

gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True

DATA_LENGTH = 150000
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
TF_CPP_MIN_LOG_LEVEL = 2


# gpu_options = tf.GPUOptions(allow_growth=True)

def main(init_train=True):
    para = config.parameters
    x = tf.placeholder(dtype=tf.float32, shape=(None, 150000, 1), name='x')
    y = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='y')
    is_train = tf.placeholder_with_default(False, (), 'is_train')
    model = conv_1d.Conv1dModel(para)
    y_ = model.forward(x, is_train)
    global_step = tf.Variable(0, trainable=False)

    with tf.name_scope('train_losses'):
        l2_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        l2_loss = sum(l2_losses)
        loss = tf.losses.mean_squared_error(y, y_) + l2_loss
        train_loss_summary = tf.summary.scalar("Loss", loss)

    with tf.name_scope('val_losses'):
        val_loss = tf.losses.mean_squared_error(y, y_)
        val_summary = tf.summary.scalar('val_loss', val_loss)

    with tf.name_scope('learning_rate'):
        learning_rate = tf.train.exponential_decay(para['learning_rate'],
                                                   global_step, 10,
                                                   para['learning_rate_decay'], staircase=True)
        lr_summary = tf.summary.scalar('learning_rate', learning_rate)

    with tf.name_scope('train_op'):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        opt = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
        opt = tf.group([opt, update_ops])

    with tf.name_scope('saver'):
        saver = tf.train.Saver()
    train_summary = tf.summary.merge([train_loss_summary, lr_summary])
    init = tf.global_variables_initializer()

    with tf.Session(config=gpu_config) as sess:
        if init_train:
            if os.path.exists(para['model_path']):
                shutil.rmtree(para['model_path'])
                os.mkdir(para['model_path'])
            else:
                os.mkdir(para['model_path'])

            if os.path.exists(para['log_path']):
                shutil.rmtree(para['log_path'])
                os.mkdir(para['log_path'])
            else:
                os.mkdir(para['log_path'])
            sess.run(init)
        else:
            ckpt = tf.train.get_checkpoint_state(para['model_path'])
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

        train_writer = tf.summary.FileWriter(para['log_path'] + 'train/')
        val_writer = tf.summary.FileWriter(para['log_path'] + 'test/')

        train_data_attrs, train_data_labels = io.read_train_data(para['data_path'])
        val_data_attrs, val_data_labels = io.read_train_data(para['validation_path'])

        norm_val_data_attrs = []
        norm_val_data_labels = []
        norm_train_data_attrs = []
        norm_train_data_labels = []
        attr_mean, attr_var = io.load_csv('./mean_std.csv', header=None)
        label_mean, label_var = io.load_csv('./mean_std_label.csv', header=None)
        for train_data_attr in train_data_attrs:
            norm_train_data_attrs.append(pp.norm(train_data_attr, attr_mean, attr_var))
        for train_data_label in train_data_labels:
            norm_train_data_labels.append(pp.norm(train_data_label, label_mean, label_var))
        for val_data_attr in val_data_attrs:
            norm_val_data_attrs.append(pp.norm(val_data_attr, attr_mean, attr_var))
        for val_data_label in val_data_labels:
            norm_val_data_labels.append(pp.norm(val_data_label, label_mean, label_var))

        norm_val_data_attrs = np.array(norm_val_data_attrs).reshape((-1, 150000, 1))
        norm_val_data_labels = np.array(norm_val_data_labels).reshape((-1, 1))

        validation_dict = {
            x: norm_val_data_attrs,
            y: norm_val_data_labels,
            is_train: False
        }

        lowest_val_loss = 10
        for step in range(para['training_steps']):
            start_time = time.time()
            print('train_step: %d start' % step)

            index = [i for i in range(len(norm_train_data_labels))]
            random.shuffle(index)
            norm_train_data_attrs = [norm_train_data_attrs[i] for i in index]
            norm_train_data_labels = [norm_train_data_labels[i] for i in index]

            for batch_attr, batch_label in [(norm_train_data_attrs[i: i + para['batch_size']],
                                             norm_train_data_labels[i: i + para['batch_size']]) \
                                            for i in range(0, len(norm_train_data_attrs), para['batch_size'])]:

                batch_attr = np.array(batch_attr).reshape((-1, 150000, 1))
                batch_label = np.array(batch_label).reshape(-1, 1)
                train_dict = {
                    x: batch_attr,
                    y: batch_label,
                    is_train: True
                }
                update_times = sess.run(global_step)
                sess.run(opt, feed_dict=train_dict)
            train_sum, train_loss, l2 = sess.run([train_summary, loss, l2_loss], feed_dict=train_dict)
            train_writer.add_summary(train_sum, update_times)

            val_sum, current_val_loss, val_result = sess.run([val_summary, val_loss, y_],
                                                             feed_dict=validation_dict)
            val_writer.add_summary(val_sum, update_times)
            print("after %d update times, test_loss is %f, train_loss is %f, l2 loss is %f" % (update_times,
                                                                                current_val_loss,
                                                                                train_loss,
                                                                                l2))
            plt.scatter(norm_val_data_labels, val_result, c=['r'])
            plt.xlabel('real')
            plt.ylabel('prediction')
            plt.savefig(para['model_path'] +  str(step) + '.png')
            plt.show()
            if current_val_loss < lowest_val_loss:
                lowest_val_loss = current_val_loss
                saver.save(sess, para['model_path'] + 'train_' + str(step), global_step=global_step)
                print('model saved')

            end_time = time.time()
            print('after %f seconds, train step %d over' % (end_time - start_time, step))

        print('train over')
    return 0


if __name__ == '__main__':
    main(init_train=True)
