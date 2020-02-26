#  #################################################################
#  Deep Reinforcement Learning for Online Ofﬂoading in Wireless Powered Mobile-Edge Computing Networks
#
#  This file contains the main code of DROO. It loads the training samples saved in ./data/data_#.mat, splits the samples into two parts (training and testing data constitutes 80% and 20%), trains the DNN with training and validation samples, and finally tests the DNN with test data.
#
#  Input: ./data/data_#.mat
#    Data samples are generated according to the CD method presented in [2]. There are 30,000 samples saved in each ./data/data_#.mat, where # is the user number. Each data sample includes
#  -----------------------------------------------------------------
#  |       wireless channel gain           |    input_h            |
#  -----------------------------------------------------------------
#  |       computing mode selection        |    output_mode        |
#  -----------------------------------------------------------------
#  |       energy broadcasting parameter   |    output_a           |
#  -----------------------------------------------------------------
#  |     transmit time of wireless device  |    output_tau         |
#  -----------------------------------------------------------------
#  |      weighted sum computation rate    |    output_obj         |
#  -----------------------------------------------------------------


import time
import numpy as np  # import numpy
import scipy.io as sio  # import scipy.io for .mat file I/
from progressbar import *
import os
from memory import MemoryDNN
from optimization import bisection

os.makedirs(os.path.join(os.getcwd(), 'model'), exist_ok=True)


def plot_rate(rate_his, title, rolling_intv=50):
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib as mpl

    rate_array = np.asarray(rate_his)
    df = pd.DataFrame(rate_his)

    mpl.style.use('seaborn')
    #    rolling_intv = 20
    if title == 'train':
        plt.figure(dpi=150)
        plt.plot(np.arange(len(rate_array)) + 1, df.rolling(rolling_intv, min_periods=1).mean(), 'b')
        plt.fill_between(np.arange(len(rate_array)) + 1, df.rolling(rolling_intv, min_periods=1).min()[0],
                         df.rolling(rolling_intv, min_periods=1).max()[0], color='c', alpha=0.2)
        plt.title('Performance after training', loc='center', fontsize=15)
        plt.ylabel('Normalized Computation Rate')
        plt.xlabel('Time Frames')
        plt.show()
        plt.savefig('Normalized computation rate(training).jpeg')
    if title == 'test':
        plt.figure(dpi=150)
        plt.plot((np.arange(len(rate_array)) + 1) * test_interval, df.rolling(rolling_intv, min_periods=1).mean(), 'xkcd:orange')
        plt.fill_between((np.arange(len(rate_array)) + 1) * test_interval, df.rolling(rolling_intv, min_periods=1).min()[0],
                         df.rolling(rolling_intv, min_periods=1).max()[0], color='xkcd:peach', alpha=0.2)
        #plt.xticks(np.arange(1, n + 1, step=test_interval))
        plt.title('Performance at different time frame', loc='center', fontsize=15)
        plt.ylabel('Normalized Computation Rate')
        plt.xlabel('Time Frames')
        plt.show()
        plt.savefig('Normalized computation rate(testing).jpeg')


def save_to_txt(rate_his, file_path):
    with open(file_path, 'w') as f:
        for rate in rate_his:
            f.write("%s \n" % rate)


if __name__ == "__main__":
    '''
        This algorithm generates K modes from DNN, and chooses with largest
        reward. The mode with largest reward is stored in the memory, which is
        further used to train the DNN.
        Adaptive K is implemented. 
    '''

    N = 10  # number of users
    n = 30000  # number of time frames
    K = 4  # initialize K = N
    decoder_mode = 'KNN'  # the quantization mode could be 'OP' (Order-preserving) or 'KNN'
    memory_size = 4096  # capacity of memory structure
    Delta = 32  # Update interval for adaptive K
    batch_size = 64
    training_interval = 25
    test_interval = 100  # Test the performance of the model every #test interval frames

    # Load data
    channel = sio.loadmat('./data/data_%d' % N)['input_h']
    rate = sio.loadmat('./data/data_%d' % N)[
        'output_obj']  # this rate is only used to plot figures; never used to train DROO. Generate by CD method in
    # optimization.py
    mode = sio.loadmat('./data/data_%d' % N)['output_mode']
    # increase h to close to 1 for better training; it is a trick widely adopted in deep learning
    channel = channel * 1000000

    # generate the train and test data sample index
    # data are split as 80:20
    # training data are randomly sampled with duplication if n > total data size

    split_idx = int(.9 * len(channel))  # channel size: 30000*10
    num_test = len(channel) - split_idx
    mem = MemoryDNN(net=[N, 60, 40, N],
                    learning_rate=0.001,
                    training_interval=training_interval,
                    test_interval=test_interval,
                    batch_size=batch_size,
                    memory_size=memory_size,
                    output_graph=False
                    )

    start_time = time.time()
    print('\n' + '=' * 20 + 'Start training' + '=' * 20)
    print(
        'User num:{}\nChannel num(Time frames):{:,}\nK:{}\nDecode mode:{}\nMemory size:{}\nDelta(K update interval):{}'
            .format(N, n, K, decoder_mode, memory_size, Delta))
    rate_his = []
    rate_his_ratio = []
    mode_his = []
    k_idx_his = []
    K_his = []
    test_rate_his = []
    avg_test_rate_ratio = []
    widgets = ['DROO: ', Percentage(), ' ', Bar(marker='0', left='[', right=']'),
               ' ', ETA(), ' ', FileTransferSpeed()]  # see docs for other options
    pbar = ProgressBar(widgets=widgets, maxval=n)
    for i in pbar(range(n)):
        # print(i)
        # for i in range(n):  # n = 30000(time frame)
        #     # Similar to the progressbar
        #     if i % (n // 10) == 0:  # n // 10 returns result in integer
        #         print("%0.1f" % (i / n))  # print out the finished percentage
        if i > 0 and i % Delta == 0:  # Update interval for adaptive K = 32
            # index counts from 0
            if Delta > 1:
                max_k = max(k_idx_his[-(Delta - 1)::]) + 1
            else:
                max_k = k_idx_his[-1] + 1
            K = min(max_k + 1, N)

        # training
        i_idx = i % split_idx  # split_index = 24000

        h = channel[i_idx, :]

        # the action selection must be either 'OP' or 'KNN'
        m_list = mem.decode(h, K, decoder_mode)

        r_list = []
        for m in m_list:
            # Compute the reward
            r_list.append(bisection(h / 1e6, m)[0])  # Because we've multiplied channel gain by 1e6 to for better
            # training result, so h need to divided by 1e6

        # encode the mode with largest reward
        mem.encode(h, m_list[np.argmax(r_list)])
        # the main code for DROO training ends here
        if (i+1) % test_interval == 0:
            test_rate_ratio = []
            m_best_arr = np.empty((0, N))
            test_idx = np.random.choice(range(split_idx, len(channel)), size=batch_size)
            for tidx in test_idx:
                h_test = channel[tidx, :]
                m_list_test = mem.decode(h_test, K, decoder_mode)
                test_rlist = []
                for m in m_list_test:
                    test_rlist.append(bisection(h_test / 1e6, m)[0])
                test_rate_his.append(max(test_rlist))
                m_best = m_list_test[np.argmax(test_rlist)]
                m_best_arr = np.vstack((m_best_arr, m_best))
                test_rate_ratio.append(test_rlist[-1] / rate[tidx][0])
            avg_test_rate_ratio.append(np.real(np.mean(test_rate_ratio[-batch_size::])))
            mem.calc_loss(m=mode[test_idx, :], m_pred=m_best_arr)
            print('\n' + '=' * 15 + 'Time frame:{:0>5d}'.format(i+1) + '=' * 15 + '\nAverage test rate ratio:{:.5f}'.format(
                np.real(np.mean(test_rate_ratio[-batch_size::]))))
        # memorize the largest reward
        rate_his.append(np.max(r_list))
        rate_his_ratio.append(rate_his[-1] / rate[i_idx][0])
        # record the index of largest reward
        k_idx_his.append(np.argmax(r_list))
        # record K in case of adaptive K
        K_his.append(K)
        mode_his.append(m_list[np.argmax(r_list)])

    total_time = time.time() - start_time
    mem.plot_cost()
    plot_rate(rate_his_ratio, title='train')
    plot_rate(avg_test_rate_ratio, title='test')
    print("Averaged normalized computation rate:", sum(rate_his_ratio[-num_test: -1]) / num_test)
    print('Total time consumed:%s' % total_time)
    print('Average time per channel:%s' % (total_time / n))

    # save data into txt
    save_to_txt(k_idx_his, "k_idx_his.txt")
    save_to_txt(K_his, "K_his.txt")
    save_to_txt(mem.cost_his, "cost_his.txt")
    save_to_txt(rate_his_ratio, "rate_his_ratio.txt")
    save_to_txt(mode_his, "mode_his.txt")
