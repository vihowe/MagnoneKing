import numpy as np
from dipoleMathplain import *
from LocationStarv2 import *


def gen_data(a_st,a_ed):
    ## ------------------- SYSTEM PARAMETERS-------------------------
    rb = 0.353
    pi = 3.1415926
    mu0 = 4*pi*10**(-7)

    Be = np.multiply([33865,-3074,34371],10**(-9))

    rb1 = 2*rb/3

    M = np.dot(np.multiply(np.dot(np.multiply((2000-1),4.0),pi),rb**3.0),Be)/3.0/mu0-np.dot(np.multiply(np.dot(np.multiply((2000-1),4.0),pi),rb1**3.0),Be)/3.0/mu0

    print('moment = {}'.format(np.linalg.norm(M)))

    d = 0.5       # baseline

    sigma1 = 500*10**(-12)  # std of Gauss noise


    ## -------------- short distance detection ----------------------
    x = np.arange(0,15,0.1)
    y = np.arange(0,15,0.1)
    z0 = 1

    ## ------------------ plain of sensor ---------------------------
    a = np.arange(a_st,a_ed,0.4)  
    b = np.arange(-1,1,0.4)         # -1, -0.6, -0.2, 0.2, 0.6
    c = 0

    threshold = 10**(-9)     # minimum signal can receive

    ## ------------------- error ------------------------------

    train_data_in = []
    train_data_out = []

    train_data_in1 = []
    train_data_out1 = []


    ## ----------------------- ITERATIONS ---------------------------
    iter = 0

    for i in np.arange(len(x)):
        for j in np.arange(len(y)):

            iter = iter + 1
            print('iteration {}'.format(iter))

            ## ------------------ fixed sensor -----------------------
            B, pos = dipoleMathplain_v2(M,x[i],y[j],z0,d)
            
            #print(pos)
            
            pos = np.array(pos)

            B = np.multiply(B,(abs(B)>threshold))
            
            for ii in np.arange(8):
                for jj in np.arange(3):
                    B[ii,jj] = round(B[ii,jj],10)

            for ii in np.arange(8):
                B[ii,:] = B[ii,:] + Be + sigma1*np.random.randn(1,3)
                  
            r_est1, nr_opt, r_estori, nr_ori, G = LocationStarv2(B, d)
            
            m_est = np.dot(np.multiply(np.dot(-4.0,pi),r_est1**4.0),(np.dot(G,nr_opt.T) - np.multiply(np.multiply(1.5,(np.dot(np.dot(nr_opt,G),nr_opt.T))),nr_opt.T))) / 3.0 / mu0

            tmp1 = np.multiply(r_est1,nr_opt)
            tmp1 = tmp1.reshape(1,-1)  # 1 x 3

            tmp2 = m_est.T
            tmp2 = tmp2.reshape(1,-1)  # 1 x 3

            temp = np.hstack((G.reshape(1,9), tmp1, tmp2))
            temp = temp.tolist()           # 1x15
               
            train_data_in.append(temp)
            
            tmp = pos.tolist()+M.tolist()
            train_data_out.append(tmp)
            
            ## --------------------- moving sensor ----------------------------
            r_rec = []
            r_rec_ori = []
            m_rec = []
            G_rec = []

            for k in np.arange(len(a)):
                for h in np.arange(len(b)):
                    B, pos_true = dipoleMathplain_v3(M,x[i],y[j],z0,a[k],b[h],c,d)
                    B = np.multiply(B,(abs(B) > threshold))
                   
                    for ii in np.arange(8):
                        for jj in np.arange(3):
                            B[ii,jj] = round(B[ii,jj],10)

                    for ii in np.arange(8):
                        B[ii,:] = B[ii,:] + Be + sigma1*np.random.randn(1,3)

                    r_est1, nr_opt, r_estori, nr_ori, G = LocationStarv2(B,d)
                    
                    temp = G.reshape(1,9)
                    temp = temp[0,:].tolist()
                    G_rec.append(temp)
                    
                    temp = np.multiply(r_est1,nr_opt) + np.array([a[k],b[h],c])
                    temp = temp.tolist()
                    r_rec.append(temp)

                    temp = np.multiply(r_estori,nr_ori) + np.array([a[k],b[h],c])
                    temp = temp.tolist()
                    r_rec_ori.append(temp)
                    
                    temp = (np.dot(np.multiply(np.dot(-4.0,pi),r_est1**4.0),(np.dot(G,nr_opt.T)-np.multiply(np.multiply(1.5,(np.dot(np.dot(nr_opt,G),nr_opt.T))),nr_opt.T)))/3.0/mu0).T
                    temp = temp.tolist()
                    m_rec.append(temp)
            
            r_rec_ori = np.array(r_rec_ori)  # 25x3    
            r_rec = np.array(r_rec)  #
            m_rec = np.array(m_rec)  #
            G_rec = np.array(G_rec)  #  25x9
            
            r_avg_ori = np.mean(r_rec_ori,0)

            r_avg = np.mean(r_rec,0)

            m_avg = np.mean(m_rec,0)

         
            train_data_in_tmp = np.hstack((G_rec,r_rec,m_rec))                 #  25x15      
            train_data_in_tmp = train_data_in_tmp.reshape(-1,np.dot(np.dot(len(a),len(b)),15))  # 1x375
            train_data_in_tmp = train_data_in_tmp.tolist()
            train_data_in1.append(train_data_in_tmp)

    ##-------------------- change to array ---------------------
    #      nosample x noinput
    ##----------------------------------------------------------
    train_data_in = np.array(train_data_in)  # (225, 1, 15)
    train_data_in = train_data_in[:,0,:]

    train_data_in1 = np.array(train_data_in1)  # (225, 1, 375)
    train_data_in1 = train_data_in1[:,0,:]

    train_data_out = np.array(train_data_out)

    train_data_out1 = train_data_out

    return train_data_in,train_data_in1,train_data_out,train_data_out1
