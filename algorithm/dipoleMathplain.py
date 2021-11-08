import numpy as np

def dipoleMathplain_v2(M,x,y,z0,d):

    B = np.zeros((8,3))
    pos = [x,y,z0]
    pos = np.array(pos)
    
    r = np.zeros((8,3))

    r[0,:]=pos-np.array([d/2,-d/2,d/2])
    r[1,:]=pos-np.array([-d/2,-d/2,d/2])
    r[2,:]=pos-np.array([-d/2,d/2,d/2])
    r[3,:]=pos-np.array([d/2,d/2,d/2])
    r[4,:]=pos-np.array([d/2,-d/2,-d/2])
    r[5,:]=pos-np.array([-d/2,-d/2,-d/2])
    r[6,:]=pos-np.array([-d/2,d/2,-d/2])
    r[7,:]=pos-np.array([d/2,d/2,-d/2])
    
    pi = 3.1415926
    mu0 = 4*pi*10**(-7)
    
    for ii in np.arange(8):
        B[ii,:]=np.dot(mu0,(np.multiply(np.dot(3.0,np.dot(M,r[ii,:])),r[ii,:])-np.multiply(M,(np.linalg.norm(r[ii,:])**2))))/4.0/pi/(np.linalg.norm(r[ii,:])**5)
    
    pos = pos.tolist()
    return B,pos


def dipoleMathplain_v3(M,x,y,z,a,b,c,d):

    B = np.zeros((8,3))

    pos_dipole = np.array([x,y,z])

    pos_sensor = np.array([a,b,c])

    pos_true = pos_dipole-pos_sensor
    pos_true = pos_true.tolist()  # array to list

    r = np.zeros((8,3))

    r[0,:]=pos_dipole-pos_sensor-np.array([d/2,-d/2,d/2])

    r[1,:]=pos_dipole-pos_sensor-np.array([-d/2,-d/2,d/2])

    r[2,:]=pos_dipole-pos_sensor-np.array([-d/2,d/2,d/2])

    r[3,:]=pos_dipole-pos_sensor-np.array([d/2,d/2,d/2])

    r[4,:]=pos_dipole-pos_sensor-np.array([d/2,-d/2,-d/2])

    r[5,:]=pos_dipole-pos_sensor-np.array([-d/2,-d/2,-d/2])

    r[6,:]=pos_dipole-pos_sensor-np.array([-d/2,d/2,-d/2])

    r[7,:]=pos_dipole-pos_sensor-np.array([d/2,d/2,-d/2])
    
    pi = 3.1415926
    mu0 = 4*pi*10**(-7)
    
    for ii in np.arange(8):
        B[ii,:]=np.dot(mu0,(np.multiply(np.dot(3.0,np.dot(M,r[ii,:])),r[ii,:])-np.multiply(M,(np.linalg.norm(r[ii,:])**2))))/4.0/pi/(np.linalg.norm(r[ii,:])**5)

    return B,pos_true