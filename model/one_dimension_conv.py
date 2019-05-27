import tensorflow as tf


class Conv1dModel:
    def __init__(self, para):
        self.para = para
        self.l2r = para['L2_regular_rate']

    def conv_1d(self, x, kernel_length, channel_num, stride, name):
        conv = tf.layers.conv1d(x, channel_num, kernel_length,
                                strides=stride,
                                padding='same',
                                activation=None,
                                use_bias=True,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.contrib.layers.xavier_initializer(),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2r),
                                bias_regularizer=tf.contrib.layers.l2_regularizer(self.l2r),
                                name=name,
                                )
        return conv

    def fc(self, x, output_size, name):
        with tf.variable_scope(name):
            output = tf.layers.dense(
                x,
                output_size,
                activation=None,
                use_bias=True,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                bias_initializer=tf.contrib.layers.xavier_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2r),
                bias_regularizer=tf.contrib.layers.l2_regularizer(self.l2r),
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                trainable=True,
                name=name
            )
        return output

    # def forward(self, x, is_train):
    #     with tf.name_scope("Convolution_layers"):
    #         conv1 = self.conv_1d(x, 100, 16, 10, name="conv1")
    #         relu1 = tf.nn.relu(conv1, name='relu1')
    #         pool1 = tf.layers.max_pooling1d(relu1, 2, 2, name="pool1")
    #         conv2 = self.conv_1d(pool1, 50, 32, 5, name="conv2")
    #         relu2 = tf.nn.relu(conv2, name='relu2')
    #         pool2 = tf.layers.max_pooling1d(relu2, 2, 2, name="pool2")
    #         conv3 = self.conv_1d(pool2, 25, 64, 3, name="conv3")
    #         relu3 = tf.nn.relu(conv3, name='relu3')
    #         pool3 = tf.layers.max_pooling1d(relu3, 2, 2, name="pool3")
    #         conv4 = self.conv_1d(pool3, 5, 64, 1, name="conv4")
    #         relu4 = tf.nn.relu(conv4, name='relu4')
    #         pool4 = tf.layers.max_pooling1d(relu4, 2, 2, name="pool4")
    #         conv5 = self.conv_1d(pool4, 5, 128, 1, name="conv5")
    #         relu5 = tf.nn.relu(conv5, name='relu5')
    #         pool5 = tf.layers.max_pooling1d(relu5, 2, 2, name="pool5")
    #         flatten = tf.layers.flatten(pool5)
    #
    #     with tf.name_scope("Dense_layers"):
    #         dense1 = self.fc(flatten, 1024, 'dense1')
    #         dropout1 = tf.layers.dropout(dense1, training=is_train, name="dropout1", rate=0.5)
    #         relu6 = tf.nn.relu(dropout1, name='relu6')
    #
    #         dense2 = self.fc(relu6, 512, 'dense2')
    #         dropout2 = tf.layers.dropout(dense2, training=is_train, name="dropout2", rate=0.5)
    #         relu7 = tf.nn.relu(dropout2, name='relu7')
    #
    #         dense3 = self.fc(relu7, 256, 'dense3')
    #         dropout3 = tf.layers.dropout(dense3, training=is_train, name="dropout3", rate=0.5)
    #         relu8 = tf.nn.relu(dropout3, name='relu8')
    #
    #         dense4 = self.fc(relu8, 1, 'dense4')
    #         sigmoid1 = tf.nn.sigmoid(dense4, name='sigmoid')
    #         prediction = 2.5 - 4 * sigmoid1
    #     return prediction

    def forward(self, x, is_train):
        with tf.name_scope("Convolution_layers"):
            conv1 = self.conv_1d(x, 1000, 16, 10, name="conv1")
            relu1 = tf.nn.relu(conv1, name='relu1')
            conv2 = self.conv_1d(relu1, 500, 16, 10, name="conv2")
            relu2 = tf.nn.relu(conv2, name='relu2')
            conv3 = self.conv_1d(relu2, 100, 32, 5, name="conv3")
            relu3 = tf.nn.relu(conv3, name='relu3')
            conv4 = self.conv_1d(relu3, 50, 32, 5, name="conv4")
            relu4 = tf.nn.relu(conv4, name='relu4')
            conv5 = self.conv_1d(relu4, 25, 64, 2, name="conv5")
            relu5 = tf.nn.relu(conv5, name='relu5')
            conv6 = self.conv_1d(relu5, 10, 64, 2, name="conv6")
            relu6 = tf.nn.relu(conv6, name='relu6')
            conv7 = self.conv_1d(relu6, 5, 64, 1, name="conv7")
            relu7 = tf.nn.relu(conv7, name='relu7')
            conv8 = self.conv_1d(relu7, 5, 64, 1, name="conv8")
            relu8 = tf.nn.relu(conv8, name='relu8')
            flatten = tf.layers.flatten(relu8)

        with tf.name_scope("Dense_layers"):
            dense1 = self.fc(flatten, 512, 'dense1')
            dropout1 = tf.layers.dropout(dense1, training=is_train, name="dropout1", rate=0.5)
            relu11 = tf.nn.relu(dropout1, name='relu11')

            dense2 = self.fc(relu11, 512, 'dense2')
            dropout2 = tf.layers.dropout(dense2, training=is_train, name="dropout2", rate=0.5)
            relu12 = tf.nn.relu(dropout2, name='relu12')

            dense3 = self.fc(relu12, 256, 'dense3')
            dropout3 = tf.layers.dropout(dense3, training=is_train, name="dropout3", rate=0.5)
            relu13 = tf.nn.relu(dropout3, name='relu13')

            dense4 = self.fc(relu13, 1, 'dense4')
        return dense4

    # def forward_with_bn(self, x, is_train):
    #     with tf.name_scope("Convolution_layers"):
    #         self.conv1 = self.conv_1d(x, 50, 16, 10, name="conv1")
    #         self.bn1 = tf.layers.batch_normalization(self.conv1, training=is_train, name='bn1')
    #         self.relu1 = tf.nn.relu(self.bn1, name='relu1')
    #         self.pool1 = tf.layers.max_pooling1d(self.relu1, 2, 2, name="pool1")
    #         self.conv2 = self.conv_1d(self.pool1, 10, 32, 2, name="conv2")
    #         self.bn2 = tf.layers.batch_normalization(self.conv2, training=is_train, name='bn2')
    #         self.relu2 = tf.nn.relu(self.bn2, name='relu2')
    #         self.pool2 = tf.layers.max_pooling1d(self.relu2, 2, 2, name="pool2")
    #         self.conv3 = self.conv_1d(self.pool2, 5, 64, 1, name="conv3")
    #         self.bn3 = tf.layers.batch_normalization(self.conv3, training=is_train, name='bn3')
    #         self.relu3 = tf.nn.relu(self.bn3, name='relu3')
    #         self.pool3 = tf.layers.max_pooling1d(self.relu3, 2, 2, name="pool3")
    #         self.conv4 = self.conv_1d(self.pool3, 5, 64, 1, name="conv4")
    #         self.bn4 = tf.layers.batch_normalization(self.conv4, training=is_train, name='bn4')
    #         self.relu4 = tf.nn.relu(self.bn4, name='relu4')
    #         self.pool4 = tf.layers.max_pooling1d(self.relu4, 2, 2, name="pool4")
    #
    #         flatten = tf.layers.flatten(self.pool4)
    #
    #     with tf.name_scope("Dense_layers"):
    #         self.dense1 = self.fc(flatten, 512, 'dense1')
    #         self.bn5 = tf.layers.batch_normalization(self.dense1, training=is_train, name='bn5')
    #         # self.dropout1 = tf.layers.dropout(self.dense1, training=is_train, name="dropout1", rate=0.5)
    #         self.relu5 = tf.nn.relu(self.bn5, name='relu5')
    #
    #         self.dense2 = self.fc(self.relu5, 64, 'dense2')
    #         self.bn6 = tf.layers.batch_normalization(self.dense2, training=is_train, name='bn6')
    #         # self.dropout2 = tf.layers.dropout(self.dense2, training=is_train, name="dropout2", rate=0.5)
    #         self.relu6 = tf.nn.relu(self.bn6, name='relu6')
    #
    #         self.dense3 = self.fc(self.relu6, 1, 'dense3')
    #         self.bn7 = tf.layers.batch_normalization(self.dense3, training=is_train, name='bn7')
    #         self.sigmoid1 = tf.nn.sigmoid(self.dense3, name='sigmoid')
    #         prediction = 1.5 - 3 * self.sigmoid1
    #
    #     return prediction
    def forward_with_bn(self, x, is_train):
        with tf.name_scope("Convolution_layers"):
            conv1 = self.conv_1d(x, 100, 16, 10, name="conv1")
            bn1 = tf.layers.batch_normalization(conv1, training=is_train, name='bn1')
            relu1 = tf.nn.relu(bn1, name='relu1')
            pool1 = tf.layers.max_pooling1d(relu1, 2, 2, name="pool1")
            conv2 = self.conv_1d(pool1, 50, 32, 5, name="conv2")
            bn2 = tf.layers.batch_normalization(conv2, training=is_train, name='bn2')
            relu2 = tf.nn.relu(bn2, name='relu2')
            pool2 = tf.layers.max_pooling1d(relu2, 2, 2, name="pool2")
            conv3 = self.conv_1d(pool2, 25, 64, 3, name="conv3")
            bn3 = tf.layers.batch_normalization(conv3, training=is_train, name='bn3')
            relu3 = tf.nn.relu(bn3, name='relu3')
            pool3 = tf.layers.max_pooling1d(relu3, 2, 2, name="pool3")
            conv4 = self.conv_1d(pool3, 5, 64, 1, name="conv4")
            bn4 = tf.layers.batch_normalization(conv4, training=is_train, name='bn4')
            relu4 = tf.nn.relu(bn4, name='relu4')
            pool4 = tf.layers.max_pooling1d(relu4, 2, 2, name="pool4")
            conv5 = self.conv_1d(pool4, 5, 64, 1, name="conv5")
            bn5 = tf.layers.batch_normalization(conv5, training=is_train, name='bn5')
            relu5 = tf.nn.relu(bn5, name='relu5')
            pool5 = tf.layers.max_pooling1d(relu5, 2, 2, name="pool5")
            flatten = tf.layers.flatten(pool5)

        # with tf.name_scope("Dense_layers"):
        #     dense1 = self.fc(flatten, 1024, 'dense1')
        #     # bn6 = tf.layers.batch_normalization(dense1, training=is_train, name='bn6')
        #     dropout1 = tf.layers.dropout(dense1, training=is_train, name="dropout1", rate=0.5)
        #     relu6 = tf.nn.relu(dropout1, name='relu6')
        #
        #     dense2 = self.fc(relu6, 512, 'dense2')
        #     # bn7 = tf.layers.batch_normalization(dense2, training=is_train, name='bn7')
        #     dropout2 = tf.layers.dropout(dense2, training=is_train, name="dropout2", rate=0.5)
        #     relu7 = tf.nn.relu(dropout2, name='relu7')
        #
        #     dense3 = self.fc(relu7, 256, 'dense3')
        #     # bn8 = tf.layers.batch_normalization(dense3, training=is_train, name='bn8')
        #     dropout3 = tf.layers.dropout(dense3, training=is_train, name="dropout3", rate=0.5)
        #     relu8 = tf.nn.relu(dropout3, name='relu8')
        #
        #     dense4 = self.fc(relu8, 1, 'dense4')
        #     # bn9 = tf.layers.batch_normalization(dense4, training=is_train, name='bn9')
        #     sigmoid1 = tf.nn.sigmoid(dense4, name='sigmoid')
        #     prediction = 1.5 - 3 * sigmoid1
        with tf.name_scope("Dense_layers"):
            dense1 = self.fc(flatten, 1024, 'dense1')
            bn6 = tf.layers.batch_normalization(dense1, training=is_train, name='bn6')
            # dropout1 = tf.layers.dropout(dense1, training=is_train, name="dropout1", rate=0.5)
            relu6 = tf.nn.relu(bn6, name='relu6')

            dense2 = self.fc(relu6, 512, 'dense2')
            bn7 = tf.layers.batch_normalization(dense2, training=is_train, name='bn7')
            # dropout2 = tf.layers.dropout(dense2, training=is_train, name="dropout2", rate=0.5)
            relu7 = tf.nn.relu(bn7, name='relu7')

            dense3 = self.fc(relu7, 256, 'dense3')
            bn8 = tf.layers.batch_normalization(dense3, training=is_train, name='bn8')
            # dropout3 = tf.layers.dropout(dense3, training=is_train, name="dropout3", rate=0.5)
            relu8 = tf.nn.relu(bn8, name='relu8')

            dense4 = self.fc(relu8, 1, 'dense4')
            bn9 = tf.layers.batch_normalization(dense4, training=is_train, name='bn9')
            sigmoid1 = tf.nn.sigmoid(bn9, name='sigmoid')
            prediction = 2.5 - 4 * sigmoid1

        return prediction


