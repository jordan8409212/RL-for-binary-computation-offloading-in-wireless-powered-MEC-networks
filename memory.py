#  #################################################################
#  This file contains memory operation including encoding and decoding operations.
#
# version 1.0 -- January 2018. Written by Liang Huang (lianghuang AT zjut.edu.cn)
#  #################################################################

from __future__ import print_function
import tensorflow as tf
import numpy as np
import os


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)

        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        # Use tf.summary.scalar to record stddev, min value and max value
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


# DNN network for memory
class MemoryDNN:
    def __init__(self, net, learning_rate, training_interval, test_interval, batch_size, memory_size, output_graph=True,
                 save_weights=True):
        # net: [n_input, n_hidden_1st, n_hidded_2ed, n_output]
        assert (len(net) is 4)  # only 4-layer DNN
        self.net = net
        self.training_interval = training_interval  # learn every training_interval
        self.test_interval = test_interval
        self.lr = learning_rate
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.output_graph = output_graph
        self.save_weights = save_weights

        # store all binary actions
        self.enumerate_actions = []
        # stored  memory entry
        self.memory_counter = 1
        # store training cost
        self.cost_his = []
        # store testing cost
        self.testcost_his = []

        self.merged = []  # Used to pass the merge string tensor

        # initialize zero memory [h, m]
        self.memory = np.zeros((self.memory_size, self.net[0] + self.net[-1]))
        # reset graph
        tf.reset_default_graph()
        # construct memory network
        self._build_net()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        if save_weights:
            self.saver = tf.train.Saver(max_to_keep=1)

        self.sess.run(tf.global_variables_initializer())  # In TF, all variables need to be initialized

    def _build_net(self):
        def build_layers(h, c_names, net, w_initializer, b_initializer):
            # tf.variable_scope() is used to define ops that creates variables(layers), use with tf.get_variable to
            # implement sharing of variables
            # When using tf.variable and tf.get_variable, their return value will be collected into collections.
            # Defaults to [tf.GraphKeys.GLOBAL_VARIABLES]
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [net[0], net[1]], initializer=w_initializer, collections=c_names)
                # variable_summaries(w1)
                b1 = tf.get_variable('b1', [1, net[1]], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(h, w1) + b1)  # h size: num_channels * N(num_WD)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [net[1], net[2]], initializer=w_initializer, collections=c_names)
                # variable_summaries(w2)
                b2 = tf.get_variable('b2', [1, net[2]], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

            with tf.variable_scope('Mode'):
                w3 = tf.get_variable('w3', [net[2], net[3]], initializer=w_initializer, collections=c_names)
                variable_summaries(w3)
                b3 = tf.get_variable('b3', [1, net[3]], initializer=b_initializer, collections=c_names)
                out = tf.matmul(l2, w3) + b3

            return out

        # ------------------ build memory_net ------------------
        self.h = tf.placeholder(tf.float32, [None, self.net[0]], name='h')  # input
        # off-loading mode or local computing mode
        self.m = tf.placeholder(tf.float32, [None, self.net[-1]], name='mode')  # for calculating loss
        self.is_train = tf.placeholder("bool")  # train or evaluate

        with tf.variable_scope('memory_net'):
            c_names, w_initializer, b_initializer = \
                ['memory_net_params', tf.GraphKeys.GLOBAL_VARIABLES], \
                tf.random_normal_initializer(0., 1 / self.net[0]), tf.constant_initializer(0.1)  # config of layers

            self.m_pred = build_layers(self.h, c_names, self.net, w_initializer, b_initializer)

        with tf.variable_scope('loss'):
            # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x)), z = labels, x = logits range:[-inf, inf]
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.m, logits=self.m_pred))
            # self.loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.m,
            # logits=self.m_pred), axis=1))
        tf.summary.scalar('Cross entropy loss', self.loss)  # Return a string type tensor

        with tf.variable_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def remember(self, h, m):
        # replace the old memory with new memory
        idx = self.memory_counter % self.memory_size
        self.memory[idx, :] = np.hstack((h, m))

        self.memory_counter += 1

    def learn(self):
        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        h_train = batch_memory[:, 0: self.net[0]]  # net[0] = number of wireless devices
        m_train = batch_memory[:, self.net[0]:]
        # print(m_train)

        # ex: t = tf.constant(42.0)
        # u = tf.constant(37.0)
        # tu = tf.mul(t, u)
        # ut = tf.mul(u, t)
        # with sess.as_default():
        #    tu.eval()  # runs one step
        #    ut.eval()  # runs one step
        #    sess.run([tu, ut])  # evaluates both tensors in a single step

        # use tensorboard to visualize your graph
        if self.output_graph:
            # $ tensorboard --logdir=logs
            self.merged = tf.summary.merge_all()

            # train the DNN
            _, cost, summary = self.sess.run([self._train_op, self.loss, self.merged],
                                             feed_dict={self.h: h_train, self.m: m_train})
            self.cost_his.append(cost)
            return summary

        else:
            _, cost = self.sess.run([self._train_op, self.loss],
                                    feed_dict={self.h: h_train, self.m: m_train})
            self.cost_his.append(cost)

    # Used to calculate test loss
    def calc_loss(self, m, m_pred):
        # print(m_pred)
        testloss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.convert_to_tensor(m, dtype=tf.float32),
                                                    logits=tf.convert_to_tensor(m_pred, dtype=tf.float32)))
        if self.output_graph:
            loss = self.sess.run(testloss)
            test_summary = tf.summary.scalar('Test loss', loss)
            test_writer = tf.summary.FileWriter("logs/test", self.sess.graph)  # Create a writer to save logs
            test_writer.add_summary(test_summary, self.memory_counter)
            test_writer.flush()
            self.testcost_his.append(loss)
            return loss
        else:
            loss = self.sess.run(testloss)
            self.testcost_his.append(loss)
            return loss

    def encode(self, h, m):
        # encoding the entry
        self.remember(h, m)  # replace the old memory with new memory
        # train the DNN every 10 step
        # Warm start
        #        if self.memory_counter> self.memory_size / 2 and self.memory_counter % self.training_interval == 0:
        if self.memory_counter % self.training_interval == 0:
            if self.output_graph:
                summary = self.learn()
                self.saver.save(self.sess,
                                os.path.join(os.getcwd(),
                                             'model\\frame{:0>5d}\\DROO-model'.format(self.memory_counter)))
                train_writer = tf.summary.FileWriter("logs/train", self.sess.graph)  # Create a writer to save logs
                train_writer.add_summary(summary, self.memory_counter)
                train_writer.flush()

            else:
                self.learn()
                self.saver.save(self.sess,
                                os.path.join(os.getcwd(), 'model\DROO-model@frame{:0>5d}'.format(self.memory_counter)))

    def decode(self, h, k, mode='OP'):
        # to have batch dimension when feed into tf placeholder
        h = h[np.newaxis, :]

        m_pred = self.sess.run(self.m_pred, feed_dict={self.h: h})

        if mode is 'OP':
            return self.knm(m_pred[0], k)
        elif mode is 'KNN':
            return self.knn(m_pred[0], k)
        else:
            print("The action selection must be 'OP' or 'KNN'")

    def knm(self, m, k):
        # return k-nearest-mode
        m_list = [1 * (m > 0.5)]
        x1 = m_list[0]
        if k > 1:
            m_abs = abs(m - 0.5)
            idx_list = np.argsort(m_abs)[:k - 1]  # sort from small to big
            for i in range(k - 1):
                if m[idx_list[i]] > 0.5:
                    # set a positive user to 0
                    m_list.append(1 * (m[idx_list[i]] - x1 < 0))
                else:
                    # set a negtive user to 1
                    m_list.append(1 * (m[idx_list[i]] - x1 <= 0))

        return m_list

    def knn(self, m, k):
        # list all 2^N binary offloading actions
        if len(self.enumerate_actions) is 0:
            import itertools
            self.enumerate_actions = np.array(list(map(list, itertools.product([0, 1], repeat=self.net[0]))))

        # the 2-norm
        sqd = ((self.enumerate_actions - m) ** 2).sum(1)
        idx = np.argsort(sqd)
        return self.enumerate_actions[idx[:k]]

    def plot_cost(self):
        import matplotlib.pyplot as plt
        fig1, ax1 = plt.subplots(2, 1, dpi=150)  # nrows, ncols, and index in order
        ax1[0].plot(np.arange(len(self.cost_his)) * self.training_interval, self.cost_his, 'xkcd:bright green')
        ax1[0].set(title='Training Loss', xlabel='Time Frames', ylabel='Cross entropy loss')
        ax1[1].plot(np.arange(len(self.testcost_his)) * self.test_interval, self.testcost_his, 'xkcd:red')
        ax1[1].set(title='Testing Loss', xlabel='Time Frames', ylabel='Cross entropy loss')
        fig1.tight_layout()
        plt.show()
        plt.savefig('Training and test loss separately.jpeg')

        plt.figure(dpi=150)
        plt.plot(np.arange(len(self.cost_his)) * self.training_interval, self.cost_his, 'xkcd:bright green',
                 label='Train loss')
        plt.plot(np.arange(len(self.testcost_his)) * self.test_interval, self.testcost_his, 'xkcd:red',
                 label='Test loss')
        plt.ylabel('Cross entropy loss')
        plt.xlabel('Time Frames')
        plt.legend(loc='best')
        plt.show()
        plt.savefig('Training and test loss together.jpeg')
