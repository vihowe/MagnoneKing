import numpy as np
import sys
from scipy.io import loadmat
from dipoleMathplain import *
from LocationStarv2 import *
from models import *
from gen_train_data import gen_data
import time
import argparse


# 均方误差
def mse(a1, a2):
    error = a1 - a2
    squaredError = []
    for val in error:
        squaredError.append(val * val)
    mean_square_error = sum(squaredError) / len(squaredError)
    return mean_square_error


# 均方根误差计算
def rmse(error):
    return np.sqrt((error**2).mean())


# istar定位程序
def location_program_istar(a_st, a_ed, x, y, sample_index):

    ## ------------------- SYSTEM PARAMETERS-------------------------
    rb = 0.353
    pi = 3.1415926
    mu0 = 4 * pi * 10**(-7)

    Be = np.multiply([33865, -3074, 34371], 10**(-9))

    rb1 = 2 * rb / 3

    M = np.dot(np.multiply(np.dot(np.multiply(
        (2000 - 1), 4.0), pi), rb**3.0), Be) / 3.0 / mu0 - np.dot(
            np.multiply(np.dot(np.multiply(
                (2000 - 1), 4.0), pi), rb1**3.0), Be) / 3.0 / mu0

    print('moment = {}'.format(np.linalg.norm(M)))

    d = 0.5  # baseline
    # d = 1       # baseline

    sigma1 = 500 * 10**(-12)  # std of Gauss noise

    ## -------------- short distance detection ----------------------
    z0 = 1

    ## ------------------ plain of sensor ---------------------------
    a = np.arange(a_st, a_ed, 0.4)  # 传感器的x坐标
    b = np.arange(-1, 1, 0.4)  # 传感器的y坐标
    c = 0

    threshold = 10**(-9)  # minimum signal can receive

    ## ------------------- error ------------------------------
    error = np.zeros(
        (len(x), len(y)))  # ISTAR (improved STAR):bearing vector error
    errorm = np.zeros((len(x), len(y)))  # ISTAR (improved STAR):moment error

    test_data_in = []
    istar = np.zeros((len(x) * len(y), 3))  # ISTAR算法预测的位置向量
    istar_moment = np.zeros((len(x) * len(y), 3))  # ISTAR算法预测的磁矩向量

    ## ----------------------- ITERATIONS ---------------------------
    iter = 0

    for i in np.arange(len(x)):
        for j in np.arange(len(y)):

            iter = iter + 1

            ## ------------------ fixed sensor -----------------------
            B, pos = dipoleMathplain_v2(M, x[i], y[j], z0,
                                        d)  # 8个传感器的磁感应强度和目标的真实位置

            #print(pos)

            pos = np.array(pos)

            B = np.multiply(B, (abs(B) > threshold))  # 传感器的分辨率

            for ii in np.arange(8):
                for jj in np.arange(3):
                    B[ii, jj] = round(B[ii, jj], 10)

            for ii in np.arange(8):
                B[ii, :] = B[ii, :] + Be + sigma1 * np.random.randn(
                    1, 3)  # 添加高斯白噪声

            r_est1, nr_opt, r_estori, nr_ori, G = LocationStarv2(
                B, d)  # 根据传感器信号进行定位

            # 计算误差
            error[i, j] = np.linalg.norm(pos - np.multiply(r_est1, nr_opt))

            #print(error[i,j])

            # 计算磁矩
            m_est = np.dot(np.multiply(np.dot(
                -4.0, pi), r_est1**4.0), (np.dot(G, nr_opt.T) - np.multiply(
                    np.multiply(1.5, (np.dot(np.dot(nr_opt, G), nr_opt.T))),
                    nr_opt.T))) / 3.0 / mu0

            # 计算磁矩误差
            errorm[i, j] = np.linalg.norm(M - m_est.T)

            #print(errorm[i,j])

            tmp1 = np.multiply(r_est1, nr_opt)
            tmp1 = tmp1.reshape(1, -1)  # 1 x 3

            temp = tmp1.tolist()  # 1x3

            test_data_in.append(temp)

    ## ----------------  evaluation -----------------------------
    #---------------------- bearing vector --------------------------
    mean_err = np.mean(error.reshape(1, np.dot(len(x), len(y))))
    std_err = np.std(error.reshape(1, np.dot(len(x), len(y))))
    rms_err = rmse(error.reshape(1, np.dot(len(x), len(y))))
    print(
        'ISTAR(bearing vector): mean error = {:.4}, std = {:.4}, RMSE = {:.4}'.
        format(mean_err, std_err, rms_err))

    ## ------------------- moment ----------------------------------
    errorm = errorm / np.linalg.norm(M)  # 磁矩的相对误差

    mean_err = np.mean(errorm.reshape(1, np.dot(len(x), len(y))))
    std_err = np.std(errorm.reshape(1, np.dot(len(x), len(y))))
    rms_err = rmse(errorm.reshape(1, np.dot(len(x), len(y))))
    print(
        'ISTAR(moment): mean error = {:.4}, std = {:.4}, RMSE = {:.4}'.format(
            mean_err, std_err, rms_err))

    ##-------------------- change to array ---------------------
    #      nosample x noinput
    ##----------------------------------------------------------
    test_data_in = np.array(test_data_in)  # (225, 1, 15)
    test_data_in = test_data_in[:, 0, :]

    ## ---------------- return value -----------------------------------
    istar = test_data_in[:, 0:3]

    return istar


# aistar定位程序
def location_program_aistar(a_st, a_ed, x, y, sample_index):

    ## ------------------- SYSTEM PARAMETERS-------------------------
    rb = 0.353
    pi = 3.1415926
    mu0 = 4 * pi * 10**(-7)

    Be = np.multiply([33865, -3074, 34371], 10**(-9))

    rb1 = 2 * rb / 3

    M = np.dot(np.multiply(np.dot(np.multiply(
        (2000 - 1), 4.0), pi), rb**3.0), Be) / 3.0 / mu0 - np.dot(
            np.multiply(np.dot(np.multiply(
                (2000 - 1), 4.0), pi), rb1**3.0), Be) / 3.0 / mu0

    print('moment = {}'.format(np.linalg.norm(M)))

    d = 0.5  # baseline
    # d = 1       # baseline

    sigma1 = 500 * 10**(-12)  # std of Gauss noise

    ## -------------- short distance detection ----------------------
    z0 = 1

    ## ------------------ plain of sensor ---------------------------
    a = np.arange(a_st, a_ed, 0.4)  # 传感器的x坐标
    b = np.arange(-1, 1, 0.4)  # 传感器的y坐标
    c = 0

    threshold = 10**(-9)  # minimum signal can receive

    ## ------------------- error ------------------------------
    error_multi = np.zeros(
        (len(x),
         len(y)))  # Avg-ISTAR (average improved STAR):bearing vector error
    errorm_multi = np.zeros(
        (len(x), len(y)))  # Avg-ISTAR (average improved STAR):moment error

    test_data_in1 = []
    pos_true_rec = []

    aistar = np.zeros((len(x) * len(y), 3))  # Avg-ISTAR算法预测的位置向量
    aistar_moment = np.zeros((len(x) * len(y), 3))  # Avg-ISTAR算法预测的磁矩向量

    ## ----------------------- ITERATIONS ---------------------------
    iter = 0

    for i in np.arange(len(x)):
        for j in np.arange(len(y)):

            iter = iter + 1

            ## ------------------ fixed sensor -----------------------
            B, pos = dipoleMathplain_v2(M, x[i], y[j], z0,
                                        d)  # 8个传感器的磁感应强度和目标的真实位置

            #print(pos)
            pos_true_rec.append(pos)  # true position
            pos = np.array(pos)

            B = np.multiply(B, (abs(B) > threshold))  # 传感器的分辨率

            for ii in np.arange(8):
                for jj in np.arange(3):
                    B[ii, jj] = round(B[ii, jj], 10)

            for ii in np.arange(8):
                B[ii, :] = B[ii, :] + Be + sigma1 * np.random.randn(
                    1, 3)  # 添加高斯白噪声

            ## --------------------- moving sensor ----------------------------
            r_rec = []  # 每个位置ISTAR预测的位置向量
            m_rec = []  # 每个位置预测的磁矩

            for k in np.arange(len(a)):
                for h in np.arange(len(b)):
                    B, pos_true = dipoleMathplain_v3(M, x[i], y[j], z0, a[k],
                                                     b[h], c, d)  # 得到磁感应强度
                    B = np.multiply(B, (abs(B) > threshold))

                    for ii in np.arange(8):
                        for jj in np.arange(3):
                            B[ii, jj] = round(B[ii, jj], 10)

                    for ii in np.arange(8):
                        B[ii, :] = B[ii, :] + Be + sigma1 * np.random.randn(
                            1, 3)

                    r_est1, nr_opt, r_estori, nr_ori, G = LocationStarv2(
                        B, d)  # 不同定位算法得到的位置向量

                    temp = np.multiply(r_est1, nr_opt) + np.array(
                        [a[k], b[h], c])
                    temp = temp.tolist()
                    r_rec.append(temp)

                    temp = (np.dot(
                        np.multiply(np.dot(-4.0, pi), r_est1**4.0),
                        (np.dot(G, nr_opt.T) - np.multiply(
                            np.multiply(1.5,
                                        (np.dot(np.dot(nr_opt, G), nr_opt.T))),
                            nr_opt.T))) / 3.0 / mu0).T
                    temp = temp.tolist()
                    m_rec.append(temp)

            r_rec = np.array(r_rec)  #
            m_rec = np.array(m_rec)  #

            r_avg = np.mean(r_rec, 0)  # Avg ISTAR算法预测的位置向量
            m_avg = np.mean(m_rec, 0)  # Avg ISTAR算法预测的磁矩向量

            # 误差计算
            error_multi[i, j] = np.linalg.norm(pos - r_avg)
            #print(error_multi[i,j])

            errorm_multi[i, j] = np.linalg.norm(M - m_avg)
            #print(errorm_multi[i,j])

            # Average ISTAR algorithm prediction result
            aistar[iter - 1, :] = r_avg
            aistar_moment[iter - 1, :] = m_avg

    ## ----------------  evaluation -----------------------------
    #---------------------- bearing vector --------------------------
    mean_err_multi = np.mean(error_multi.reshape(1, np.dot(len(x), len(y))))
    std_err_multi = np.std(error_multi.reshape(1, np.dot(len(x), len(y))))
    rms_err_multi = rmse(error_multi.reshape(1, np.dot(len(x), len(y))))
    print(
        'Average ISTAR(bearing vector): mean error = {:.4}, std = {:.4}, RMSE = {:.4}'
        .format(mean_err_multi, std_err_multi, rms_err_multi))

    ## ------------------- moment ----------------------------------
    errorm_multi = errorm_multi / np.linalg.norm(M)

    mean_err_multi = np.mean(errorm_multi.reshape(1, np.dot(len(x), len(y))))
    std_err_multi = np.std(errorm_multi.reshape(1, np.dot(len(x), len(y))))
    rms_err_multi = rmse(errorm_multi.reshape(1, np.dot(len(x), len(y))))
    print(
        'Average ISTAR(moment): mean error = {:.4}, std = {:.4}, RMSE = {:.4}'.
        format(mean_err_multi, std_err_multi, rms_err_multi))

    return aistar


# nn_istar定位程序
def location_program_nn_istar(a_st, a_ed, x, y, sample_index):

    ######################################################################
    #        train network:
    #                   data is generated by MATLAB
    #
    #####################################################################

    train_flag = 0  # 是否重新训练
    genrate_dataset = 0  # 是否需要生成数据

    if train_flag:

        # 生成训练的数据集
        if genrate_dataset:
            train_data_in, train_data_in1, train_data_out, train_data_out1 = gen_data(
                a_st, a_ed)
            np.save('train_data_in_{}'.format(sample_index), train_data_in)
            np.save('train_data_in1_{}'.format(sample_index), train_data_in1)
            np.save('train_data_out_{}'.format(sample_index), train_data_out)
            np.save('train_data_out1_{}'.format(sample_index), train_data_out1)
        else:
            train_data_in = np.load(
                'train_data_in_{}.npy'.format(sample_index))
            train_data_in1 = np.load(
                'train_data_in1_{}.npy'.format(sample_index))
            train_data_out = np.load(
                'train_data_out_{}.npy'.format(sample_index))
            train_data_out1 = np.load(
                'train_data_out1_{}.npy'.format(sample_index))

        no_pos = int(train_data_in1.shape[1] / 15)  # 25

        ## ====================== train network =========================
        batch_size = 1024
        lr = 0.001

        train(train_data_in, train_data_out, batch_size, lr, 0,
              sample_index)  # 训练网络(传感器在原点不移动)

    ## ------------------- SYSTEM PARAMETERS-------------------------
    rb = 0.353
    pi = 3.1415926
    mu0 = 4 * pi * 10**(-7)

    Be = np.multiply([33865, -3074, 34371], 10**(-9))

    rb1 = 2 * rb / 3

    M = np.dot(np.multiply(np.dot(np.multiply(
        (2000 - 1), 4.0), pi), rb**3.0), Be) / 3.0 / mu0 - np.dot(
            np.multiply(np.dot(np.multiply(
                (2000 - 1), 4.0), pi), rb1**3.0), Be) / 3.0 / mu0

    print('moment = {}'.format(np.linalg.norm(M)))

    d = 0.5  # baseline
    # d = 1       # baseline

    sigma1 = 500 * 10**(-12)  # std of Gauss noise

    ## -------------- short distance detection ----------------------
    z0 = 1

    ## ------------------ plain of sensor ---------------------------
    a = np.arange(a_st, a_ed, 0.4)  # 传感器的x坐标
    b = np.arange(-1, 1, 0.4)  # 传感器的y坐标
    c = 0

    threshold = 10**(-9)  # minimum signal can receive

    ## ------------------- error ------------------------------
    test_data_in = []
    pos_true_rec = []

    nn_istar = np.zeros((len(x) * len(y), 3))  # NN-ISTAR算法预测的位置向量

    ## ----------------------- ITERATIONS ---------------------------
    iter = 0

    for i in np.arange(len(x)):
        for j in np.arange(len(y)):

            iter = iter + 1

            ## ------------------ fixed sensor -----------------------
            B, pos = dipoleMathplain_v2(M, x[i], y[j], z0,
                                        d)  # 8个传感器的磁感应强度和目标的真实位置

            #print(pos)

            pos_true_rec.append(pos)  # true position

            pos = np.array(pos)

            B = np.multiply(B, (abs(B) > threshold))  # 传感器的分辨率

            for ii in np.arange(8):
                for jj in np.arange(3):
                    B[ii, jj] = round(B[ii, jj], 10)

            for ii in np.arange(8):
                B[ii, :] = B[ii, :] + Be + sigma1 * np.random.randn(
                    1, 3)  # 添加高斯白噪声

            r_est1, nr_opt, r_estori, nr_ori, G = LocationStarv2(
                B, d)  # 根据传感器信号进行定位

            # 计算磁矩
            m_est = np.dot(np.multiply(np.dot(
                -4.0, pi), r_est1**4.0), (np.dot(G, nr_opt.T) - np.multiply(
                    np.multiply(1.5, (np.dot(np.dot(nr_opt, G), nr_opt.T))),
                    nr_opt.T))) / 3.0 / mu0

            tmp1 = np.multiply(r_est1, nr_opt)
            tmp1 = tmp1.reshape(1, -1)  # 1 x 3

            tmp2 = m_est.T
            tmp2 = tmp2.reshape(1, -1)  # 1 x 3

            temp = np.hstack((G.reshape(1, 9), tmp1, tmp2))
            temp = temp.tolist()  # 1x15

            test_data_in.append(temp)  # 神经网络的训练数据

    ###############################################################################
    #
    #                         nNN-ISTAR Testing
    #                                 & Evaluation
    #
    ###############################################################################

    ##-------------------- change to array ---------------------
    #      nosample x noinput
    ##----------------------------------------------------------
    test_data_in = np.array(test_data_in)  # (225, 1, 15)
    test_data_in = test_data_in[:, 0, :]

    pos_true_rec = np.array(pos_true_rec)

    test_data_in = test_data_in[:, 0:12]

    ## ---------------- testing process -------------------------
    #  local_nn_test(data, flag)
    #  flag = 0: ISTAR
    #  flag >0: avg ISTAR
    ## ----------------------------------------------------------
    ## improved STAR
    NN_output_test = test(test_data_in, 0, sample_index)

    Rest_nn = NN_output_test[:, 0:3]

    error_net = []

    for k in np.arange(NN_output_test.shape[0]):
        error_net.append(np.linalg.norm(pos_true_rec[k, :] - Rest_nn[k, :]))

    ## ---------------------- evaluation --------------------------
    # bearing vector
    error_net = np.array(error_net)

    mean_err_net = np.mean(error_net)  # 平均误差
    std_err_net = np.std(error_net)  # 误差的标准差
    rms_err_net = rmse(error_net)  # 均方根误差
    print(
        'NN-ISTAR(bearing vector): mean error = {:.4}, std = {:.4}, RMSE = {:.4}'
        .format(mean_err_net, std_err_net, rms_err_net))

    nn_istar = Rest_nn

    return nn_istar


# nn_aistar定位程序
def location_program_nn_aistar(a_st, a_ed, x, y, sample_index):

    ######################################################################
    #        train network:
    #                   data is generated by MATLAB
    #
    #####################################################################
    train_flag = 0  # 是否重新训练
    genrate_dataset = 0  # 是否需要生成数据

    if train_flag:

        # 生成训练的数据集
        if genrate_dataset:
            train_data_in, train_data_in1, train_data_out, train_data_out1 = gen_data(
                a_st, a_ed)
            np.save('train_data_in_{}'.format(sample_index), train_data_in)
            np.save('train_data_in1_{}'.format(sample_index), train_data_in1)
            np.save('train_data_out_{}'.format(sample_index), train_data_out)
            np.save('train_data_out1_{}'.format(sample_index), train_data_out1)
        else:
            train_data_in = np.load(
                'train_data_in_{}.npy'.format(sample_index))
            train_data_in1 = np.load(
                'train_data_in1_{}.npy'.format(sample_index))
            train_data_out = np.load(
                'train_data_out_{}.npy'.format(sample_index))
            train_data_out1 = np.load(
                'train_data_out1_{}.npy'.format(sample_index))

        no_pos = int(train_data_in1.shape[1] / 15)  # 25

        ## ====================== train network =========================
        batch_size = 1024
        lr = 0.001

        # train(train_data_in, train_data_out, batch_size, lr, 0, sample_index)  # 训练网络(传感器在原点不移动)

        # 训练网络(传感器在不同位置)
        for kk in np.arange(no_pos):
            print(kk + 1)
            train(train_data_in1[:, 15 * kk:15 * (1 + kk)], train_data_out1,
                  batch_size, lr, kk + 1, sample_index)

    ## ------------------- SYSTEM PARAMETERS-------------------------
    rb = 0.353
    pi = 3.1415926
    mu0 = 4 * pi * 10**(-7)

    Be = np.multiply([33865, -3074, 34371], 10**(-9))

    rb1 = 2 * rb / 3

    M = np.dot(np.multiply(np.dot(np.multiply(
        (2000 - 1), 4.0), pi), rb**3.0), Be) / 3.0 / mu0 - np.dot(
            np.multiply(np.dot(np.multiply(
                (2000 - 1), 4.0), pi), rb1**3.0), Be) / 3.0 / mu0

    print('moment = {}'.format(np.linalg.norm(M)))

    d = 0.5  # baseline
    # d = 1       # baseline

    sigma1 = 500 * 10**(-12)  # std of Gauss noise

    ## -------------- short distance detection ----------------------
    z0 = 1

    ## ------------------ plain of sensor ---------------------------
    a = np.arange(a_st, a_ed, 0.4)  # 传感器的x坐标
    b = np.arange(-1, 1, 0.4)  # 传感器的y坐标
    c = 0

    threshold = 10**(-9)  # minimum signal can receive

    test_data_in1 = []
    pos_true_rec = []

    nn_aistar = np.zeros((len(x) * len(y), 3))  # Avg NN-ISTAR算法预测的位置向量

    ## ----------------------- ITERATIONS ---------------------------
    iter = 0

    for i in np.arange(len(x)):
        for j in np.arange(len(y)):

            iter = iter + 1

            ## ------------------ fixed sensor -----------------------
            B, pos = dipoleMathplain_v2(M, x[i], y[j], z0,
                                        d)  # 8个传感器的磁感应强度和目标的真实位置

            #print(pos)
            pos_true_rec.append(pos)  # true position
            pos = np.array(pos)

            B = np.multiply(B, (abs(B) > threshold))  # 传感器的分辨率

            for ii in np.arange(8):
                for jj in np.arange(3):
                    B[ii, jj] = round(B[ii, jj], 10)

            for ii in np.arange(8):
                B[ii, :] = B[ii, :] + Be + sigma1 * np.random.randn(
                    1, 3)  # 添加高斯白噪声

            ## --------------------- moving sensor ----------------------------
            r_rec = []  # 每个位置ISTAR预测的位置向量
            m_rec = []  # 每个位置预测的磁矩
            G_rec = []  # 每个位置的磁梯度矩阵

            for k in np.arange(len(a)):
                for h in np.arange(len(b)):
                    B, pos_true = dipoleMathplain_v3(M, x[i], y[j], z0, a[k],
                                                     b[h], c, d)  # 得到磁感应强度
                    B = np.multiply(B, (abs(B) > threshold))

                    for ii in np.arange(8):
                        for jj in np.arange(3):
                            B[ii, jj] = round(B[ii, jj], 10)

                    for ii in np.arange(8):
                        B[ii, :] = B[ii, :] + Be + sigma1 * np.random.randn(
                            1, 3)

                    r_est1, nr_opt, r_estori, nr_ori, G = LocationStarv2(
                        B, d)  # 不同定位算法得到的位置向量

                    temp = G.reshape(1, 9)
                    temp = temp[0, :].tolist()
                    G_rec.append(temp)

                    temp = np.multiply(r_est1, nr_opt) + np.array(
                        [a[k], b[h], c])
                    temp = temp.tolist()
                    r_rec.append(temp)

                    temp = (np.dot(
                        np.multiply(np.dot(-4.0, pi), r_est1**4.0),
                        (np.dot(G, nr_opt.T) - np.multiply(
                            np.multiply(1.5,
                                        (np.dot(np.dot(nr_opt, G), nr_opt.T))),
                            nr_opt.T))) / 3.0 / mu0).T
                    temp = temp.tolist()
                    m_rec.append(temp)

            r_rec = np.array(r_rec)  #
            m_rec = np.array(m_rec)  #
            G_rec = np.array(G_rec)  #  25x9

            r_avg = np.mean(r_rec, 0)  # Avg ISTAR算法预测的位置向量

            m_avg = np.mean(m_rec, 0)  # Avg ISTAR算法预测的磁矩向量

            # 神经网络的测试数据
            test_data_in_tmp = np.hstack((G_rec, r_rec, m_rec))  #  25x15
            test_data_in_tmp = test_data_in_tmp.reshape(
                -1, np.dot(np.dot(len(a), len(b)), 15))  # 1x375
            test_data_in_tmp = test_data_in_tmp.tolist()
            test_data_in1.append(test_data_in_tmp)

    ###############################################################################
    #
    #                         nNN-ISTAR Testing
    #                                 & Evaluation
    #
    ###############################################################################

    ##-------------------- change to array ---------------------
    #      nosample x noinput
    ##----------------------------------------------------------

    test_data_in1 = np.array(test_data_in1)  # (225, 1, 375)
    test_data_in1 = test_data_in1[:, 0, :]

    pos_true_rec = np.array(pos_true_rec)

    ## Average nn_ISTAR
    Rest_nn1 = np.zeros((np.dot(len(x), len(y)), 3))

    for kk in np.arange(np.dot(len(a), len(b))):
        test_data = test_data_in1[:, 15 * kk:15 * (kk + 1)]
        test_data = test_data[:, 0:12]

        NN_output_test1 = test(test_data, kk + 1, sample_index)
        Rest_nn1 = Rest_nn1 + NN_output_test1[:, 0:3]

    Rest_nn1 = Rest_nn1 / np.dot(len(a), len(b))

    ## ---------------- return value -----------------------------------
    nn_aistar = Rest_nn1

    ## ---------------- error calculation-------------------------------
    error_multi_net = []

    for k in np.arange(NN_output_test1.shape[0]):
        error_multi_net.append(
            np.linalg.norm(pos_true_rec[k, :] - Rest_nn1[k, :]))

    ## ---------------------- evaluation --------------------------
    # bearing vector
    error_multi_net = np.array(error_multi_net)

    mean_err_multi_net = np.mean(error_multi_net)
    std_err_multi_net = np.std(error_multi_net)
    rms_err_multi_net = rmse(error_multi_net)

    print(
        'Average NN-ISTAR(bearing vector): mean error = {:.4}, std = {:.4}, RMSE = {:.4}'
        .format(mean_err_multi_net, std_err_multi_net, rms_err_multi_net))

    return nn_aistar


# 输入参数 sample_rate = 1000    # 次/秒  (采样率，改变该参数可以提高定位的数据量)
def main(sample_rate=1):

    ######################################################################
    #                      Main program
    #       sample rate : 1000 samples per second
    #       velocity : 0.1m/s
    #
    ######################################################################
    grid_size = 0.4  # 0.4m
    velocity = 0.1  # 0.1m/s

    nosample = int(grid_size / velocity * sample_rate)  # 4000

    delta_ = velocity / sample_rate  # every data moving distance

    x = np.arange(0, 15, 1)
    y = np.arange(0, 15, 1)

    # 不同算法预测的位置向量
    istar_ = np.zeros((len(x) * len(y), 3 * nosample))  # ISTAR
    nn_istar_ = np.zeros((len(x) * len(y), 3 * nosample))  # NN-ISTAR
    aistar_ = np.zeros((len(x) * len(y), 3 * nosample))  # Avg ISTAR
    nn_aistar_ = np.zeros((len(x) * len(y), 3 * nosample))  # Avg NN-ISTAR

    for i in np.arange(nosample):
        sample_index = i + 1
        print('sample index = {}'.format(sample_index))
        a_st = -1 + delta_ * i
        a_ed = 1 + delta_ * i

        # no position x 3
        # istar, nn_istar, aistar, nn_aistar = location_program(a_st,a_ed,x,y,sample_index)
        istar = location_program_istar(a_st, a_ed, x, y, sample_index)
        nn_istar = location_program_nn_istar(a_st, a_ed, x, y, sample_index)
        aistar = location_program_aistar(a_st, a_ed, x, y, sample_index)
        nn_aistar = location_program_nn_aistar(a_st, a_ed, x, y, sample_index)

        istar_[:, i * 3:(i + 1) * 3] = istar
        nn_istar_[:, i * 3:(i + 1) * 3] = nn_istar
        aistar_[:, i * 3:(i + 1) * 3] = aistar
        nn_aistar_[:, i * 3:(i + 1) * 3] = nn_aistar

    # 最后将预测的位置向量保存在csv文件中
    # np.savetxt('istar.csv', istar_, delimiter = ',')
    # np.savetxt('nn_istar.csv',nn_istar_, delimiter = ',')
    # np.savetxt('aistar.csv',aistar_, delimiter = ',')
    # np.savetxt('nn_aistar.csv',nn_aistar_, delimiter = ',')
    return "finished"


def test_loc(location_func, sample_rate=1,):
    ######################################################################
    #                      Test program
    #       sample rate : 1000 samples per second
    #       velocity : 0.1m/s
    #
    ######################################################################
    grid_size = 0.4  # 0.4m
    velocity = 0.1  # 0.1m/s

    nosample = int(grid_size / velocity * sample_rate)  # 4000

    delta_ = velocity / sample_rate  # every data moving distance

    x = np.arange(0, 5, 1)
    y = np.arange(0, 5, 1)

    # 不同算法预测的位置向量
    ret_ = np.zeros((len(x) * len(y), 3 * nosample))
    # istar_ = np.zeros((len(x) * len(y), 3 * nosample))  # ISTAR
    # nn_istar_ = np.zeros((len(x) * len(y), 3 * nosample))  # NN-ISTAR
    # aistar_ = np.zeros((len(x) * len(y), 3 * nosample))  # Avg ISTAR
    # nn_aistar_ = np.zeros((len(x) * len(y), 3 * nosample))  # Avg NN-ISTAR

    for i in np.arange(nosample):
        sample_index = i + 1
        print('sample index = {}'.format(sample_index))
        a_st = -1 + delta_ * i
        a_ed = 1 + delta_ * i

        # no position x 3
        # istar, nn_istar, aistar, nn_aistar = location_program(a_st,a_ed,x,y,sample_index)
        ret = location_func(a_st, a_ed, x, y, sample_index)
        # istar = location_program_istar(a_st, a_ed, x, y, sample_index)
        # nn_istar = location_program_nn_istar(a_st, a_ed, x, y, sample_index)
        # aistar = location_program_aistar(a_st, a_ed, x, y, sample_index)
        # nn_aistar = location_program_nn_aistar(a_st, a_ed, x, y, sample_index)
        ret_[:, i * 3:(i + 1) * 3] = ret

        # istar_[:, i * 3:(i + 1) * 3] = istar
        # nn_istar_[:, i * 3:(i + 1) * 3] = nn_istar
        # aistar_[:, i * 3:(i + 1) * 3] = aistar
        # nn_aistar_[:, i * 3:(i + 1) * 3] = nn_aistar

    # 最后将预测的位置向量保存在csv文件中
    # np.savetxt('istar.csv', istar_, delimiter = ',')
    # np.savetxt('nn_istar.csv',nn_istar_, delimiter = ',')
    # np.savetxt('aistar.csv',aistar_, delimiter = ',')
    # np.savetxt('nn_aistar.csv',nn_aistar_, delimiter = ',')
    return "finished"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=int, default=0)
    args = parser.parse_args()


    sample_rate = 1  # 次/秒  (采样率，改变该参数可以提高定位的数据量)

    # 注意修改采样率之后，需要在定位程序里重新生成训练数据和重新训练神经网络
    # 即将下面两个参数设置为1
    # train_flag = 1   # 是否重新训练
    #     genrate_dataset = 1  # 是否需要生成数据
    location_funcs = [location_program_istar, location_program_aistar, location_program_nn_istar, location_program_nn_aistar]
    t_start = time.perf_counter()
    # main(sample_rate)
    test_loc(location_funcs[args.task])
    print(f"elapsed time: {(time.perf_counter() - t_start)*1000/25} ms")
