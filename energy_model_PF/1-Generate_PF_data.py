import numpy as np
import pandas as pd

np.random.seed(5)

#Load csv data
data = pd.read_csv("../HMX_data/Data_denoised_PF.csv")

#Assign strain and stress components
F11 = data["f11"].values
F22 = data["f22"].values
F33 = data["f33"].values
F12 = data["f12"].values
F23 = data["f23"].values
F13 = data["f13"].values
F21 = data["f21"].values
F32 = data["f32"].values
F31 = data["f31"].values

P11 = data["p11"].values
P22 = data["p22"].values
P33 = data["p33"].values
P12 = data["p12"].values
P23 = data["p23"].values
P13 = data["p13"].values
P21 = data["p21"].values
P32 = data["p32"].values
P31 = data["p31"].values

stress_data = np.zeros((P11.shape[0],9))
stress_data[:,0] = P11
stress_data[:,1] = P22
stress_data[:,2] = P33
stress_data[:,3] = P12
stress_data[:,4] = P23
stress_data[:,5] = P13
stress_data[:,6] = P21
stress_data[:,7] = P32
stress_data[:,8] = P31

strain_data = np.zeros((P11.shape[0],9))
strain_data[:,0] = F11 
strain_data[:,1] = F22 
strain_data[:,2] = F33 
strain_data[:,3] = F12
strain_data[:,4] = F23
strain_data[:,5] = F13
strain_data[:,6] = F21
strain_data[:,7] = F32
strain_data[:,8] = F31

strain_data0 = np.copy(strain_data)
stress_data0 = np.copy(stress_data)

import math

#Define component-wise tensor rotation function
def apply_rotation(q1,q2,q3,q4,q5,q6,q7,q8,q9,a1,a2,a3,a4,a5,a6,a7,a8,a9):
    ar1 = a1*q1 + a4*q2 + a7*q3
    ar2 = a2*q1 + a5*q2 + a8*q3
    ar3 = a3*q1 + a6*q2 + a9*q3

    ar4 = a1*q4 + a4*q5 + a7*q6
    ar5 = a2*q4 + a5*q5 + a8*q6
    ar6 = a3*q4 + a6*q5 + a9*q6

    ar7 = a1*q7 + a4*q8 + a7*q9
    ar8 = a2*q7 + a5*q8 + a8*q9
    ar9 = a3*q7 + a6*q8 + a9*q9

    return ar1,ar2,ar3,ar4,ar5,ar6,ar7,ar8,ar9

#Define function to get rotation tensor given angles
def get_general_rotation_matrix(a,b,c):
    q1 = np.cos(a) * np.cos(b)
    q2 = np.cos(a) * np.sin(b) * np.sin(c) - np.sin(a) * np.cos(c)
    q3 = np.cos(a) * np.sin(b) * np.cos(c) + np.sin(a) * np.sin(c)

    q4 = np.sin(a) * np.cos(b)
    q5 = np.sin(a) * np.sin(b) * np.sin(c) + np.cos(a) * np.cos(c)
    q6 = np.sin(a) * np.sin(b) * np.cos(c) - np.cos(a) * np.sin(c)

    q7 = -np.sin(b)
    q8 = np.cos(b) * np.sin(c)
    q9 = np.cos(b) * np.cos(c)


    R = np.zeros((3,3))
    R[0,0] = q1
    R[0,1] = q2
    R[0,2] = q3

    R[1,0] = q4
    R[1,1] = q5
    R[1,2] = q6

    R[2,0] = q7
    R[2,1] = q8
    R[2,2] = q9

    is_id = np.dot(np.transpose(R),R)
    tr = np.trace(is_id)

    #Check if rotation tensor is orthogonal
    if math.isclose(tr , 3.) == False:
        print("Not a rotation matrix! The trace of QT*Q should be 3. Trace is:", tr)
        exit()

    return q1,q2,q3,q4,q5,q6,q7,q8,q9

#Placeholder for rotated tensor data
strain_data1 = np.zeros_like(strain_data0)
stress_data1 = np.zeros_like(stress_data0)

#Iterate through original tensor data and apply rotatation
for k in range(strain_data0.shape[0]):
    #Apply a rotation:
    a = 0.
    b = np.pi
    c = 0.

    q1,q2,q3,q4,q5,q6,q7,q8,q9 = get_general_rotation_matrix(a,b,c)

    a1  = strain_data0[k,0]#F11 
    a5  = strain_data0[k,1]#F22 
    a9  = strain_data0[k,2]#F33 
    a2  = strain_data0[k,3]#F12
    a6  = strain_data0[k,4]#F23
    a3  = strain_data0[k,5]#F13
    a4  = strain_data0[k,6]#F21
    a8  = strain_data0[k,7]#F32
    a7  = strain_data0[k,8]#F31

    ar1,ar2,ar3,ar4,ar5,ar6,ar7,ar8,ar9 = apply_rotation(q1,q2,q3,q4,q5,q6,q7,q8,q9,a1,a2,a3,a4,a5,a6,a7,a8,a9)

    strain_data1[k,0] = ar1 #F11 
    strain_data1[k,1] = ar5 #F22 
    strain_data1[k,2] = ar9 #F33 
    strain_data1[k,3] = ar2 #F12
    strain_data1[k,4] = ar6 #F23
    strain_data1[k,5] = ar3 #F13
    strain_data1[k,6] = ar4 #F21
    strain_data1[k,7] = ar8 #F32
    strain_data1[k,8] = ar7 #F31


    a1  = stress_data0[k,0]#F11 
    a5  = stress_data0[k,1]#F22 
    a9  = stress_data0[k,2]#F33 
    a2  = stress_data0[k,3]#F12
    a6  = stress_data0[k,4]#F23
    a3  = stress_data0[k,5]#F13
    a4  = stress_data0[k,6]#F21
    a8  = stress_data0[k,7]#F32
    a7  = stress_data0[k,8]#F31

    ar1,ar2,ar3,ar4,ar5,ar6,ar7,ar8,ar9 = apply_rotation(q1,q2,q3,q4,q5,q6,q7,q8,q9,a1,a2,a3,a4,a5,a6,a7,a8,a9)

    stress_data1[k,0] = ar1 #F11 
    stress_data1[k,1] = ar5 #F22 
    stress_data1[k,2] = ar9 #F33 
    stress_data1[k,3] = ar2 #F12
    stress_data1[k,4] = ar6 #F23
    stress_data1[k,5] = ar3 #F13
    stress_data1[k,6] = ar4 #F21
    stress_data1[k,7] = ar8 #F32
    stress_data1[k,8] = ar7 #F31

#Concatenate orginal and rotated data
strain_data = np.concatenate([strain_data0, strain_data1], axis = 0)
stress_data = np.concatenate([stress_data0, stress_data1], axis = 0)

#Save data as .npy
np.save("PF_stress_data_saved.npy", stress_data)
np.save("PF_strain_data_saved.npy", strain_data)
print("Saved PF model data.")
