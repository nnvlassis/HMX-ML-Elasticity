import numpy as np


import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
from keras import Model
import keras

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten, Input, Multiply, Reshape, Subtract, Dropout, Concatenate, Lambda
from keras.optimizers import Adam, Nadam
import pandas as pd
from keras.models import load_model
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K

np.random.seed(5)

K.set_floatx('float64')
K.set_epsilon(1e-16)

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

#Copy of original not rotated data
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

    if math.isclose(tr , 3.) == False:
        print("Not a rotation matrix! The trace of QT*Q should be 3. Trace is:", tr)
        exit()

    return q1,q2,q3,q4,q5,q6,q7,q8,q9

#Load .npy data
strain_data = np.load("PF_strain_data_saved.npy")
stress_data = np.load("PF_stress_data_saved.npy")

strain_data = strain_data
stress_data = stress_data /100. #Simple data scaling

# ------------------------------
# Input Data
# ------------------------------

#Preprocess strain data
INPUT = np.copy(strain_data)

#Prepare rotation matrix
a = 0.
b = np.pi
c = 0.
q1,q2,q3,q4,q5,q6,q7,q8,q9 = get_general_rotation_matrix(a,b,c)

#Define zero strain point to evaluate in loss function (unit F)
model_output_size =1
x_00 = np.zeros((2,9))
x_00[0,0] = 1.
x_00[0,1] = 1.
x_00[0,2] = 1.

#Rotate F -- implemented here for general case use
a1  = x_00[0,0]#F11 
a5  = x_00[0,1]#F22 
a9  = x_00[0,2]#F33
a2  = x_00[0,3]#F12
a6  = x_00[0,4]#F23
a3  = x_00[0,5]#F13
a4  = x_00[0,6]#F21
a8  = x_00[0,7]#F32
a7  = x_00[0,8]#F31
ar1,ar2,ar3,ar4,ar5,ar6,ar7,ar8,ar9 = apply_rotation(q1,q2,q3,q4,q5,q6,q7,q8,q9,a1,a2,a3,a4,a5,a6,a7,a8,a9)

x_00[1,0] = ar1 #F11 
x_00[1,1] = ar5 #F22 
x_00[1,2] = ar9 #F33
x_00[1,3] = ar2 #F12
x_00[1,4] = ar6 #F23
x_00[1,5] = ar3 #F13
x_00[1,6] = ar4 #F21
x_00[1,7] = ar8 #F32
x_00[1,8] = ar7 #F31

# ------------------------------
# Output Data
# ------------------------------
#Preprocess stress data
OUTPUT = np.copy(stress_data)

#Define zero stress point to evaluate in loss function
grad_00 = np.zeros((2,9))
grad_00[0] = stress_data[0]

#Rotate stress -- implemented here for general case use / residual stress
a1  = grad_00[0,0]#F11 
a5  = grad_00[0,1]#F22 
a9  = grad_00[0,2]#F33 
a2  = grad_00[0,3]#F12
a6  = grad_00[0,4]#F23
a3  = grad_00[0,5]#F13
a4  = grad_00[0,6]#F21
a8  = grad_00[0,7]#F32
a7  = grad_00[0,8]#F31
ar1,ar2,ar3,ar4,ar5,ar6,ar7,ar8,ar9 = apply_rotation(q1,q2,q3,q4,q5,q6,q7,q8,q9,a1,a2,a3,a4,a5,a6,a7,a8,a9)

grad_00[1,0] = ar1 #F11 
grad_00[1,1] = ar5 #F22 
grad_00[1,2] = ar9 #F33 
grad_00[1,3] = ar2 #F12
grad_00[1,4] = ar6 #F23
grad_00[1,5] = ar3 #F13
grad_00[1,6] = ar4 #F21
grad_00[1,7] = ar8 #F32
grad_00[1,8] = ar7 #F31
        
# ------------------------------
# Read Data
# ------------------------------
#Prepare NN inputs
X = np.copy(INPUT)
Y = np.copy(OUTPUT)

#Subselect number of points and their index to apply the rotation metric to 
data0 = np.concatenate((X,Y), axis =-1)
data_randoms = np.random.rand(20000,14)

data = data0

index_mask1 = np.ones((data0.shape[0],1))
index_mask2 = np.ones((data_randoms.shape[0],1)) 
index_mask = np.concatenate([index_mask1,index_mask2],axis = 0)


index_mask1 = np.zeros((strain_data0.shape[0],1))
index_mask2 = np.ones((strain_data0.shape[0],1))
index_mask_metric = np.concatenate([index_mask1,index_mask2],axis = 0)


#Shuffle input data
s = np.arange(data.shape[0])
np.random.shuffle(s)

#Strain data
X = np.copy(data[s,:9])

#Index data
index_mask = np.copy(index_mask[s])
index_mask_metric = np.copy(index_mask_metric[s])

#Define train/test split index
train_index = int(0.7*X.shape[0])

E11 = X[:,0].reshape(-1,1)
E22 = X[:,1].reshape(-1,1)
E33 = X[:,2].reshape(-1,1)
E12 = X[:,3].reshape(-1,1)
E23 = X[:,4].reshape(-1,1)
E13 = X[:,5].reshape(-1,1)
E21 = X[:,6].reshape(-1,1)
E32 = X[:,7].reshape(-1,1)
E31 = X[:,8].reshape(-1,1)

#Stress data
Y = np.copy(data[s,9:])

#Define a dummy energy potential input of the correct size -- the NN is not optimized for this value -- it is just a placeholder for the correct size
fake_potential = np.copy(data[s,6]) 

#Total loss function metric
def grad_diff(gradient_diff,rotated_energy_loss):

    def grad_loss(ytrue, ypred):

        #Stress gradient loss
        gras_zeros = K.zeros_like(gradient_diff)
        grad_part = keras.losses.mean_squared_error(gras_zeros,gradient_diff)

        #Define zero strain TF tensor (unit F)
        x_00_tensor  = K.constant(x_00)
        x_00_tensor0 = K.constant(x_00[:,0].reshape(-1,1))
        x_00_tensor1 = K.constant(x_00[:,1].reshape(-1,1))
        x_00_tensor2 = K.constant(x_00[:,2].reshape(-1,1))
        x_00_tensor3 = K.constant(x_00[:,3].reshape(-1,1))
        x_00_tensor4 = K.constant(x_00[:,4].reshape(-1,1))
        x_00_tensor5 = K.constant(x_00[:,5].reshape(-1,1))
        x_00_tensor6 = K.constant(x_00[:,6].reshape(-1,1))
        x_00_tensor7 = K.constant(x_00[:,7].reshape(-1,1))
        x_00_tensor8 = K.constant(x_00[:,8].reshape(-1,1))

        #Define index placeholder TF tensor -- not necessary for gradient calculation
        index_placeholder = K.ones_like(x_00_tensor5)
        
        #Define zero energy TF tensor
        y_00_tensor = K.constant(np.zeros((model_output_size,1)))

        #Define zero stress TF tensor
        grad_00_tensor0 = K.constant(grad_00[:,0].reshape(-1,1))
        grad_00_tensor1 = K.constant(grad_00[:,1].reshape(-1,1))
        grad_00_tensor2 = K.constant(grad_00[:,2].reshape(-1,1))
        grad_00_tensor3 = K.constant(grad_00[:,3].reshape(-1,1))
        grad_00_tensor4 = K.constant(grad_00[:,4].reshape(-1,1))
        grad_00_tensor5 = K.constant(grad_00[:,5].reshape(-1,1))
        grad_00_tensor6 = K.constant(grad_00[:,6].reshape(-1,1))
        grad_00_tensor7 = K.constant(grad_00[:,7].reshape(-1,1))
        grad_00_tensor8 = K.constant(grad_00[:,8].reshape(-1,1))

        #Energy prediction at zero strain
        y_00_pred = model([x_00_tensor0, x_00_tensor1, x_00_tensor2, x_00_tensor3, x_00_tensor4, x_00_tensor5, x_00_tensor6, x_00_tensor7, x_00_tensor8, x_00_tensor,index_placeholder,index_placeholder])

        #Energy loss predition at zero strain
        zero_point = keras.losses.mean_squared_error(y_00_tensor,y_00_pred)
        zero_point = K.mean(zero_point,axis = 0)

        #Stress prediction at zero strain
        gradients_zero0 =  K.gradients(y_00_pred, x_00_tensor0)[0]
        gradients_zero1 =  K.gradients(y_00_pred, x_00_tensor1)[0]
        gradients_zero2 =  K.gradients(y_00_pred, x_00_tensor2)[0]
        gradients_zero3 =  K.gradients(y_00_pred, x_00_tensor3)[0]
        gradients_zero4 =  K.gradients(y_00_pred, x_00_tensor4)[0]
        gradients_zero5 =  K.gradients(y_00_pred, x_00_tensor5)[0]
        gradients_zero6 =  K.gradients(y_00_pred, x_00_tensor6)[0]
        gradients_zero7 =  K.gradients(y_00_pred, x_00_tensor7)[0]
        gradients_zero8 =  K.gradients(y_00_pred, x_00_tensor8)[0]

        #Stress loss predition at zero strain
        gradients_zero_point0 = keras.losses.mean_squared_error(gradients_zero0,grad_00_tensor0)
        gradients_zero_point1 = keras.losses.mean_squared_error(gradients_zero1,grad_00_tensor1)
        gradients_zero_point2 = keras.losses.mean_squared_error(gradients_zero2,grad_00_tensor2)
        gradients_zero_point3 = keras.losses.mean_squared_error(gradients_zero3,grad_00_tensor3)
        gradients_zero_point4 = keras.losses.mean_squared_error(gradients_zero4,grad_00_tensor4)
        gradients_zero_point5 = keras.losses.mean_squared_error(gradients_zero5,grad_00_tensor5)
        gradients_zero_point6 = keras.losses.mean_squared_error(gradients_zero6,grad_00_tensor6)
        gradients_zero_point7 = keras.losses.mean_squared_error(gradients_zero7,grad_00_tensor7)
        gradients_zero_point8 = keras.losses.mean_squared_error(gradients_zero8,grad_00_tensor8)


        gradients_zero_point = gradients_zero_point0 + gradients_zero_point1 + gradients_zero_point2 + gradients_zero_point3 + gradients_zero_point4 + gradients_zero_point5 + gradients_zero_point6 + gradients_zero_point7 + gradients_zero_point8 

        gradients_zero_point = K.mean(gradients_zero_point,axis = 0)

        #Total loss prediction
        loss = 1. * grad_part + 1.* zero_point + 1. *  gradients_zero_point + 1. * rotated_energy_loss
        
        return loss
    return grad_loss




#Stress gradient prediction loss function metric -- similar to above and defined separately to record partial loss changes during training
def grad_loss(grad_diff):

    def grad_part_loss(ytrue, ypred):
        gras_zeros = K.zeros_like(gradient_diff)
        grad_part = keras.losses.mean_squared_error(gras_zeros,gradient_diff)
        return grad_part
    return grad_part_loss

#Energy at zero strain prediction loss function metric -- similar to above and defined separately to record partial loss changes during training
def energy_zero_point_check():

    def energy_zero(ytrue, ypred):

        x_00_tensor  = K.constant(x_00)
        x_00_tensor0 = K.constant(x_00[:,0].reshape(-1,1))
        x_00_tensor1 = K.constant(x_00[:,1].reshape(-1,1))
        x_00_tensor2 = K.constant(x_00[:,2].reshape(-1,1))
        x_00_tensor3 = K.constant(x_00[:,3].reshape(-1,1))
        x_00_tensor4 = K.constant(x_00[:,4].reshape(-1,1))
        x_00_tensor5 = K.constant(x_00[:,5].reshape(-1,1))
        x_00_tensor6 = K.constant(x_00[:,6].reshape(-1,1))
        x_00_tensor7 = K.constant(x_00[:,7].reshape(-1,1))
        x_00_tensor8 = K.constant(x_00[:,8].reshape(-1,1))

        index_placeholder = K.ones_like(x_00_tensor5)

        y_00_tensor = K.constant(np.zeros((model_output_size,1)))
        grad_00_tensor0 = K.constant(grad_00[:,0].reshape(-1,1))
        grad_00_tensor1 = K.constant(grad_00[:,1].reshape(-1,1))
        grad_00_tensor2 = K.constant(grad_00[:,2].reshape(-1,1))
        grad_00_tensor3 = K.constant(grad_00[:,3].reshape(-1,1))
        grad_00_tensor4 = K.constant(grad_00[:,4].reshape(-1,1))
        grad_00_tensor5 = K.constant(grad_00[:,5].reshape(-1,1))
        grad_00_tensor6 = K.constant(grad_00[:,6].reshape(-1,1))
        grad_00_tensor7 = K.constant(grad_00[:,7].reshape(-1,1))
        grad_00_tensor8 = K.constant(grad_00[:,8].reshape(-1,1))


        y_00_pred = model([x_00_tensor0, x_00_tensor1, x_00_tensor2, x_00_tensor3, x_00_tensor4, x_00_tensor5, x_00_tensor6, x_00_tensor7, x_00_tensor8, x_00_tensor,index_placeholder,index_placeholder])

        zero_point = keras.losses.mean_squared_error(y_00_tensor,y_00_pred)
        zero_point = K.mean(zero_point,axis = 0)
        return zero_point
    return energy_zero

#Stress at zero strain prediction loss function metric -- similar to above and defined separately to record partial loss changes during training
def gradient_zero_point_check():

    def grad_zero(ytrue, ypred):

        x_00_tensor  = K.constant(x_00)
        x_00_tensor0 = K.constant(x_00[:,0].reshape(-1,1))
        x_00_tensor1 = K.constant(x_00[:,1].reshape(-1,1))
        x_00_tensor2 = K.constant(x_00[:,2].reshape(-1,1))
        x_00_tensor3 = K.constant(x_00[:,3].reshape(-1,1))
        x_00_tensor4 = K.constant(x_00[:,4].reshape(-1,1))
        x_00_tensor5 = K.constant(x_00[:,5].reshape(-1,1))
        x_00_tensor6 = K.constant(x_00[:,6].reshape(-1,1))
        x_00_tensor7 = K.constant(x_00[:,7].reshape(-1,1))
        x_00_tensor8 = K.constant(x_00[:,8].reshape(-1,1))

        index_placeholder = K.ones_like(x_00_tensor5)

        grad_00_tensor0 = K.constant(grad_00[:,0].reshape(-1,1))
        grad_00_tensor1 = K.constant(grad_00[:,1].reshape(-1,1))
        grad_00_tensor2 = K.constant(grad_00[:,2].reshape(-1,1))
        grad_00_tensor3 = K.constant(grad_00[:,3].reshape(-1,1))
        grad_00_tensor4 = K.constant(grad_00[:,4].reshape(-1,1))
        grad_00_tensor5 = K.constant(grad_00[:,5].reshape(-1,1))
        grad_00_tensor6 = K.constant(grad_00[:,6].reshape(-1,1))
        grad_00_tensor7 = K.constant(grad_00[:,7].reshape(-1,1))
        grad_00_tensor8 = K.constant(grad_00[:,8].reshape(-1,1))


        y_00_pred = model([x_00_tensor0, x_00_tensor1, x_00_tensor2, x_00_tensor3, x_00_tensor4, x_00_tensor5, x_00_tensor6, x_00_tensor7, x_00_tensor8, x_00_tensor,index_placeholder,index_placeholder])


        gradients_zero0 =  K.gradients(y_00_pred, x_00_tensor0)[0]
        gradients_zero1 =  K.gradients(y_00_pred, x_00_tensor1)[0]
        gradients_zero2 =  K.gradients(y_00_pred, x_00_tensor2)[0]
        gradients_zero3 =  K.gradients(y_00_pred, x_00_tensor3)[0]
        gradients_zero4 =  K.gradients(y_00_pred, x_00_tensor4)[0]
        gradients_zero5 =  K.gradients(y_00_pred, x_00_tensor5)[0]
        gradients_zero6 =  K.gradients(y_00_pred, x_00_tensor6)[0]
        gradients_zero7 =  K.gradients(y_00_pred, x_00_tensor7)[0]
        gradients_zero8 =  K.gradients(y_00_pred, x_00_tensor8)[0]

        gradients_zero_point0 = keras.losses.mean_squared_error(gradients_zero0,grad_00_tensor0)
        gradients_zero_point1 = keras.losses.mean_squared_error(gradients_zero1,grad_00_tensor1)
        gradients_zero_point2 = keras.losses.mean_squared_error(gradients_zero2,grad_00_tensor2)
        gradients_zero_point3 = keras.losses.mean_squared_error(gradients_zero3,grad_00_tensor3)
        gradients_zero_point4 = keras.losses.mean_squared_error(gradients_zero4,grad_00_tensor4)
        gradients_zero_point5 = keras.losses.mean_squared_error(gradients_zero5,grad_00_tensor5)
        gradients_zero_point6 = keras.losses.mean_squared_error(gradients_zero6,grad_00_tensor6)
        gradients_zero_point7 = keras.losses.mean_squared_error(gradients_zero7,grad_00_tensor7)
        gradients_zero_point8 = keras.losses.mean_squared_error(gradients_zero8,grad_00_tensor8)


        gradients_zero_point = gradients_zero_point0 + gradients_zero_point1 + gradients_zero_point2 + gradients_zero_point3 + gradients_zero_point4 + gradients_zero_point5 + gradients_zero_point6 + gradients_zero_point7 + gradients_zero_point8 
        gradients_zero_point = K.mean(gradients_zero_point,axis = 0)

        return gradients_zero_point
    return grad_zero

#Symmetric stress prediction loss function metric -- similar to above and defined separately to record partial loss changes during training
def sym_grad_loss(gradient_diff_symmetry):

    def sym_grad_part_loss(ytrue, ypred):
        gras_zeros = K.zeros_like(gradient_diff_symmetry)
        grad_part = keras.losses.mean_squared_error(gras_zeros,gradient_diff_symmetry)
        return grad_part
    return sym_grad_part_loss

#Symmetric energy prediction loss function metric -- similar to above and defined separately to record partial loss changes during training
def sym_energy_loss(energy_loss):

    def sym_energy_part_loss(ytrue, ypred):
        
        return energy_loss
    return sym_energy_part_loss

#Define TF gradient function
def getgradient(x):
    return tf.gradients(x[1], x[0])

#Get the rotation tensor components
a = 0.
b = np.pi
c = 0.

q1,q2,q3,q4,q5,q6,q7,q8,q9 = get_general_rotation_matrix(a,b,c)

# Forward Mapping Model - - - - - - - - - - - - - - - - - - - -
width = 100 # NN hidden layer neurons

# Define NN strain inputs separately to take individual derivatives if needed
e11 = Input((1,)) 
e22 = Input((1,)) 
e33 = Input((1,)) 
e12 = Input((1,)) 
e23 = Input((1,)) 
e13 = Input((1,)) 
e21 = Input((1,)) 
e32 = Input((1,)) 
e31 = Input((1,)) 

# Define NN true gradient inputs -- will be used to calulcate loss
true_gradient = Input((9,))

# Define NN index inputs for metrics
ind_mask = Input((1,))
ind_mask_metric = Input((1,))

#Concatenate strain inputs
conc_inp = Concatenate(axis=-1)([e11,e22,e33,e12,e23,e13,e21,e32,e31])

#Define NN architecture
h = Dense(width, activation="relu")(conc_inp)
h = Multiply()([h,h])
h = Multiply()([h,h])
h = Dense(width, activation="relu")(h)
h = Multiply()([h,h])
out = Dense(1, activation="linear", name = 'for')(h)

#Define NN model
model = Model(inputs=[e11,e22,e33,e12,e23,e13,e21,e32,e31,true_gradient, ind_mask, ind_mask_metric], outputs=[out] )

#Define gradient function
gradient = Lambda(getgradient,output_shape=(9,))([conc_inp,out]) 

#Calculate true and predicted stress gradient difference
gradient_diff = gradient - true_gradient

#Select rotated and not rotated metrics
gradient_diff = Multiply()([gradient_diff, ind_mask])
gradient_diff_symmetry = Multiply()([gradient_diff, ind_mask_metric])

#Apply rotation
re11, re12, re13, re21, re22, re23, re31, re32, re33 = apply_rotation(q1,q2,q3,q4,q5,q6,q7,q8,q9, e11, e12, e13, e21, e22, e23, e31, e32, e33 )

#Predict energy for rotated input
rotated_energy = model([re11, re22, re33, re12, re23, re13, re21, re32, re31,true_gradient, ind_mask, ind_mask_metric])

#Calculate loss for predicted energy for rotated input
rotated_energy_loss = keras.losses.mean_squared_error(rotated_energy, out)
rotated_energy_loss = K.mean(rotated_energy_loss,axis = 0)

#Define optimizer
opt = Nadam(lr = 0.01)

#Compile model
model.compile(optimizer='nadam', loss=grad_diff(gradient_diff,rotated_energy_loss),metrics = [grad_loss(gradient_diff), energy_zero_point_check() ,gradient_zero_point_check(), sym_energy_loss(rotated_energy_loss),sym_grad_loss(gradient_diff_symmetry)])

# Print model summary
model.summary()


from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

#Define model callbacks
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.90, patience=5, min_lr=1e-9,verbose = 1)
earlystop = EarlyStopping(monitor='loss', patience=1000)
checkpoint = ModelCheckpoint('trained_models/energy_model_PF.h5',monitor='loss',verbose=1,save_best_only=True)

#Train model
history = model.fit([E11[:train_index],E22[:train_index],E33[:train_index],E12[:train_index],E23[:train_index],E13[:train_index],E21[:train_index],E32[:train_index],E31[:train_index],Y[:train_index], index_mask[:train_index], index_mask_metric[:train_index]], fake_potential[:train_index], 
    epochs=1000, batch_size = 512, callbacks = [reduce_lr,checkpoint,earlystop], shuffle = True, 
    validation_data=[[E11[train_index:],E22[train_index:],E33[train_index:],E12[train_index:],E23[train_index:],E13[train_index:],E21[train_index:],E32[train_index:],E31[train_index:],Y[train_index:], index_mask[train_index:], index_mask_metric[train_index:]], fake_potential[train_index:] ]) 



print("Training Complete! Model trained as trained_models/energy_model_PF.h5")
