import numpy as np

def mse(a1,a2):
    error = a1-a2
    squaredError = []
    for val in error:
        squaredError.append(val*val)
    mean_square_error = sum(squaredError) / len(squaredError)
    return mean_square_error

def LocationStarv2(B, d):

    Gx1 = np.array([(-(-B[0,1]-B[4,1]+B[3,1]+B[7,1])-(B[0,2]-B[4,2]+B[3,2]-B[7,2])),(-B[0,0]-B[4,0]+B[3,0]+B[7,0]),(B[0,0]-B[4,0]+B[3,0]-B[7,0]),(-B[0,0]-B[4,0]+B[3,0]+B[7,0]),(-B[0,1]-B[4,1]+B[3,1]+B[7,1]),(B[0,1]-B[4,1]+B[3,1]-B[7,1]),(B[0,0]-B[4,0]+B[3,0]-B[7,0]),(-B[0,2]-B[4,2]+B[3,2]+B[7,2]),(B[0,2]-B[4,2]+B[3,2]-B[7,2])])/2/d
    Gx1 = Gx1.reshape(3,3)

    Gx2 = np.array([(-(-B[1,1]-B[5,1]+B[2,1]+B[6,1])-(B[1,2]-B[5,2]+B[2,2]-B[6,2])),(-B[1,0]-B[5,0]+B[2,0]+B[6,0]),(B[1,0]-B[5,0]+B[2,0]-B[6,0]),(-B[1,0]-B[5,0]+B[2,0]+B[6,0]),(-B[1,1]-B[5,1]+B[2,1]+B[6,1]),(B[1,1]-B[5,1]+B[2,1]-B[6,1]),(B[1,0]-B[5,0]+B[2,0]-B[6,0]),(-B[1,2]-B[5,2]+B[2,2]+B[6,2]),(B[1,2]-B[5,2]+B[2,2]-B[6,2])])/2/d
    Gx2 = Gx2.reshape(3,3)

    Gy1 = np.array([(B[3,0]+B[7,0]-B[2,0]-B[6,0]),(B[3,1]+B[7,1]-B[2,1]-B[6,1]),(B[3,0]-B[7,0]+B[2,0]-B[6,0]),(B[3,1]+B[7,1]-B[2,1]-B[6,1]),(-(B[3,0]+B[7,0]-B[2,0]-B[6,0])-(B[3,2]-B[7,2]+B[2,2]-B[6,2])),(B[3,1]-B[7,1]+B[2,1]-B[6,1]),(B[3,2]+B[7,2]-B[2,2]-B[6,2]),(B[3,1]-B[7,1]+B[2,1]-B[6,1]),(B[3,2]-B[7,2]+B[2,2]-B[6,2])])/2/d
    Gy1 = Gy1.reshape(3,3)

    Gy2 = np.array([(B[0,0]+B[4,0]-B[1,0]-B[5,0]),(B[0,1]+B[4,1]-B[1,1]-B[5,1]),(B[0,0]-B[4,0]+B[1,0]-B[5,0]),(B[0,1]+B[4,1]-B[1,1]-B[5,1]),(-(B[0,0]+B[4,0]-B[1,0]-B[5,0])-(B[0,2]-B[4,2]+B[1,2]-B[5,2])),(B[0,1]-B[4,1]+B[1,1]-B[5,1]),(B[0,2]+B[4,2]-B[1,2]-B[5,2]),(B[0,1]-B[4,1]+B[1,1]-B[5,1]),(B[0,2]-B[4,2]+B[1,2]-B[5,2])])/2/d
    Gy2 = Gy2.reshape(3,3)

    Gz1 = np.array([(B[0,0]-B[1,0]+B[3,0]-B[2,0]),(-B[0,0]-B[1,0]+B[3,0]+B[2,0]),(B[0,2]-B[1,2]+B[3,2]-B[2,2]),(B[0,1]-B[1,1]+B[3,1]-B[2,1]),(-B[0,1]-B[1,1]+B[3,1]+B[2,1]),(-B[0,2]-B[1,2]+B[3,2]+B[2,2]),(B[0,2]-B[1,2]+B[3,2]-B[2,2]),(-B[0,2]-B[1,2]+B[3,2]+B[2,2]),(-(B[0,0]-B[1,0]+B[3,0]-B[2,0])-(-B[0,1]-B[1,1]+B[3,1]+B[2,1]))])/2/d
    Gz1 = Gz1.reshape(3,3)

    Gz2 = np.array([(B[4,0]-B[5,0]+B[7,0]-B[6,0]),(-B[4,0]-B[5,0]+B[7,0]+B[6,0]),(B[4,2]-B[5,2]+B[7,2]-B[6,2]),(B[4,1]-B[5,1]+B[7,1]-B[6,1]),(-B[4,1]-B[5,1]+B[7,1]+B[6,1]),(-B[4,2]-B[5,2]+B[7,2]+B[6,2]),(B[4,2]-B[5,2]+B[7,2]-B[6,2]),(-B[4,2]-B[5,2]+B[7,2]+B[6,2]),(-(B[4,0]-B[5,0]+B[7,0]-B[6,0])-(-B[4,1]-B[5,1]+B[7,1]+B[6,1]))])/2/d
    Gz2 = Gz2.reshape(3,3)


    # Frobenius norm
    CTx1 = np.linalg.norm(Gx1,'fro')
    CTx2 = np.linalg.norm(Gx2,'fro')
    CTy1 = np.linalg.norm(Gy1,'fro')
    CTy2 = np.linalg.norm(Gy2,'fro')
    CTz1 = np.linalg.norm(Gz1,'fro')
    CTz2 = np.linalg.norm(Gz2,'fro')

    CT_ori = (CTx1+CTx2+CTy1+CTy2+CTz1+CTz2)/6

    dCT_ori = np.array([(CTx1-CTx2)/d,(CTy1-CTy2)/d,(CTz1-CTz2)/d])

    nr_ori = dCT_ori/np.linalg.norm(CT_ori)

    r_estori = abs(np.dot(nr_ori,np.array([0,0,1]))*d/((CTz2/CTz1)**0.25-1)-np.dot(nr_ori,np.array([0,0,1]))*d/2)

    if (CTz2/CTz1 ==  1):
        print('CTz2/CTz1 = 1')
    
    # the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
    Dx1, Vx1 = np.linalg.eig(Gx1)
    Dx2, Vx2 = np.linalg.eig(Gx2)
    Dy1, Vy1 = np.linalg.eig(Gy1)
    Dy2, Vy2 = np.linalg.eig(Gy2)
    Dz1, Vz1 = np.linalg.eig(Gz1)
    Dz2, Vz2 = np.linalg.eig(Gz2)

    lambda_ =  np.zeros((6,3))

    tmp = [Dx1[0],Dx1[1],Dx1[2]]
    tmp.sort()
    lambda_[0,:] = np.array(tmp)

    tmp = [Dx2[0],Dx2[1],Dx2[2]]
    tmp.sort()    
    lambda_[1,:] = np.array(tmp)

    tmp = [Dy1[0],Dy1[1],Dy1[2]]
    tmp.sort()
    lambda_[2,:] = np.array(tmp)

    tmp = [Dy2[0],Dy2[1],Dy2[2]]
    tmp.sort()
    lambda_[3,:] = np.array(tmp)

    tmp = [Dz1[0],Dz1[1],Dz1[2]]
    tmp.sort()
    lambda_[4,:] = np.array(tmp)

    tmp = [Dz2[0],Dz2[1],Dz2[2]]
    tmp.sort()
    lambda_[5,:] = np.array(tmp)

    CTtemp = np.zeros((1,6))

    for jj in np.arange(6):
        CTtemp[0,jj] = (-lambda_[jj,1]**2-np.dot(lambda_[jj,0],lambda_[jj,2]))**0.5
    
    CTnew = np.mean(CTtemp)

    dCTnew = np.array([(CTtemp[0,0]-CTtemp[0,1])/d,(CTtemp[0,2]-CTtemp[0,3])/d,(CTtemp[0,4]-CTtemp[0,5])/d])
    
    nr1 = dCTnew/np.linalg.norm(dCTnew)

    r_est1 = np.dot(4,CTnew)/np.linalg.norm(dCTnew)

    if (np.linalg.norm(dCTnew) ==  0):
        print('norm(dCTnew) = 0')

    # center G
    G = np.zeros((3,3))

    Bxb = (B[1,0]+B[5,0]+B[2,0]+B[6,0])/4
    Bxf = (B[0,0]+B[3,0]+B[4,0]+B[7,0])/4
    Bxr = (B[2,0]+B[3,0]+B[7,0]+B[6,0])/4
    Bxl = (B[0,0]+B[1,0]+B[4,0]+B[5,0])/4
    Bxu = (B[0,0]+B[1,0]+B[2,0]+B[3,0])/4
    Bxd = (B[4,0]+B[5,0]+B[6,0]+B[7,0])/4

    Byb = (B[1,1]+B[5,1]+B[2,1]+B[6,1])/4
    Byf = (B[0,1]+B[3,1]+B[4,1]+B[7,1])/4
    Byr = (B[2,1]+B[3,1]+B[7,1]+B[6,1])/4
    Byl = (B[0,1]+B[1,1]+B[4,1]+B[5,1])/4
    Byu = (B[0,1]+B[1,1]+B[2,1]+B[3,1])/4
    Byd = (B[4,1]+B[5,1]+B[6,1]+B[7,1])/4

    Bzb = (B[1,2]+B[5,2]+B[2,2]+B[6,2])/4
    Bzf = (B[0,2]+B[3,2]+B[4,2]+B[7,2])/4
    Bzr = (B[2,2]+B[3,2]+B[7,2]+B[6,2])/4
    Bzl = (B[0,2]+B[1,2]+B[4,2]+B[5,2])/4
    Bzu = (B[0,2]+B[1,2]+B[2,2]+B[3,2])/4
    Bzd = (B[4,2]+B[5,2]+B[6,2]+B[7,2])/4

    G[0,0] = (Bxf-Bxb)/d
    G[0,1] = (Bxr-Bxl+Byf-Byb)/2/d
    G[1,0] = G[0,1]
    G[0,2] = (Bzf-Bzb+Bxu-Bxd)/2/d
    G[2,0] = G[0,2]
    G[1,1] = (Byr-Byl)/d
    G[1,2] = (Bzr-Bzl+Byu-Byd)/2/d
    G[2,1] = G[1,2]
    G[2,2] = (-G[0,0]-G[1,1]+(Bzu-Bzd)/d)/2

    D,V = np.linalg.eig(G)
    
    # np.argsort(x): sort array, return index
    temp = np.array([D[0],D[1],D[2]])
    idx_sort = np.argsort(temp)
    lambda_yin = temp[idx_sort]

    V = V[:,idx_sort]

    alpha1 = ((lambda_yin[1]-lambda_yin[0])/(lambda_yin[2]-lambda_yin[0]))**0.5

    alpha2 = ((lambda_yin[2]-lambda_yin[1])/(lambda_yin[2]-lambda_yin[0]))**0.5

    
    nr2 = np.multiply(alpha1,V[:,0].T)+np.multiply(alpha2,V[:,2].T)   # 1x3

    nr3 = np.multiply(-alpha1,V[:,0].T)+np.multiply(alpha2,V[:,2].T)

    nr4 = np.multiply(alpha1,V[:,0].T)-np.multiply(alpha2,V[:,2].T)

    nr5 = np.multiply(-alpha1,V[:,0].T)-np.multiply(alpha2,V[:,2].T)

    min_err = np.array([mse(nr1,nr2),mse(nr1,nr3),mse(nr1,nr4),mse(nr1,nr5)])

    id = np.argsort(min_err)

    if (id[0] ==  0):
        nr_opt = nr2
    else:
        if (id[0] ==  1):
            nr_opt = nr3
        else:
            if (id[0] ==  2):
                nr_opt = nr4
            else:
                nr_opt = nr5

    return r_est1,nr_opt,r_estori,nr_ori,G