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
from sklearn import preprocessing
from sklearn.externals import joblib
from keras.models import load_model
import tensorflow as tf
import keras.backend as K


np.random.seed(5)

K.set_floatx('float64')
K.set_epsilon(1e-16)

#Load .npy data
strain_data = np.load("SE_strain_data_saved.npy")
stress_data = np.load("SE_stress_data_saved.npy")

#Preprocess strain data
INPUT = np.copy(strain_data)

#Scaling from 0 to 1
INPUT_Scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
INPUT_Scaled = INPUT_Scaler.fit_transform(INPUT) 

#Define zero strain point to evaluate in loss function
model_output_size = 1
x_00 = np.zeros((model_output_size,6))
x_00 = INPUT_Scaler.transform(x_00)

#Extract scaling factor to use in loss metrics
A_factor = INPUT_Scaler.scale_

#Save input scaler
joblib.dump(INPUT_Scaler, 'trained_models/INPUT_Scaler.pkl')

# ------------------------------
# Output Data
# ------------------------------
#Preprocess stress data
OUTPUT = np.copy(stress_data)
OUTPUT_Scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

#Scaling from 0 to 1
OUTPUT_Scaled = OUTPUT_Scaler.fit_transform(OUTPUT) #

#Extract scaling factor to use in loss metrics
B_factor = 1. / OUTPUT_Scaler.scale_

#Define zero stress point to evaluate in loss function
grad_00 = np.zeros((model_output_size,6)) #
grad_00 = OUTPUT_Scaler.transform(grad_00)

#Extract scaling factors to use in loss metrics
C_factor = OUTPUT_Scaler.scale_
D_factor =  OUTPUT_Scaler.min_

#Save output scaler
joblib.dump(OUTPUT_Scaler, 'trained_models/OUTPUT_Scaler.pkl')

# ------------------------------
# Read Data
# ------------------------------
#Prepare NN inputs
X = np.copy(INPUT_Scaled)
Y = np.copy(OUTPUT_Scaled)

data = np.concatenate((X,Y), axis =-1)

#Shuffle input data
s = np.arange(data.shape[0])
np.random.shuffle(s)

#Strain data
X = np.copy(data[s,:6])

E11 = X[:,0].reshape(-1,1)
E22 = X[:,1].reshape(-1,1)
E33 = X[:,2].reshape(-1,1)
E12 = X[:,3].reshape(-1,1)
E23 = X[:,4].reshape(-1,1)
E13 = X[:,5].reshape(-1,1)

#Stress data
Y = np.copy(data[s,6:])

#Define a dummy energy potential input of the correct size -- the NN is not optimized for this value -- it is just a placeholder for the correct size
fake_potential = np.copy(X[:,0].reshape(-1,1)) 

#Define train/test split index
train_index = int(0.7*X.shape[0])

# ------------------------------
# Custom Loss
# ------------------------------
#Total loss function metric
def grad_diff(gradient_diff):

    def grad_loss(ytrue, ypred):

        #Stress gradient loss
        gras_zeros = K.zeros_like(gradient_diff)
        grad_part = keras.losses.mean_squared_error(gras_zeros,gradient_diff)

        #Define zero strain TF tensor
        x_00_tensor  = K.constant(x_00)
        x_00_tensor0 = K.constant(x_00[:,0].reshape(-1,1))
        x_00_tensor1 = K.constant(x_00[:,1].reshape(-1,1))
        x_00_tensor2 = K.constant(x_00[:,2].reshape(-1,1))
        x_00_tensor3 = K.constant(x_00[:,3].reshape(-1,1))
        x_00_tensor4 = K.constant(x_00[:,4].reshape(-1,1))
        x_00_tensor5 = K.constant(x_00[:,5].reshape(-1,1))

        #Define zero energy TF tensor
        y_00_tensor = K.constant(np.zeros((model_output_size,1)))

        #Define zero stress TF tensor
        grad_00_tensor0 = K.constant(grad_00[:,0].reshape(-1,1))
        grad_00_tensor1 = K.constant(grad_00[:,1].reshape(-1,1))
        grad_00_tensor2 = K.constant(grad_00[:,2].reshape(-1,1))
        grad_00_tensor3 = K.constant(grad_00[:,3].reshape(-1,1))
        grad_00_tensor4 = K.constant(grad_00[:,4].reshape(-1,1))
        grad_00_tensor5 = K.constant(grad_00[:,5].reshape(-1,1))

        #Energy prediction at zero strain
        y_00_pred = model([x_00_tensor0, x_00_tensor1, x_00_tensor2, x_00_tensor3, x_00_tensor4, x_00_tensor5, x_00_tensor])

        #Energy loss predition at zero strain
        zero_point = keras.losses.mean_squared_error(y_00_tensor,y_00_pred)
        zero_point = K.mean(zero_point,axis = 0)

        #Stress prediction at zero strain
        gradients_zero0 =  K.gradients(y_00_pred, x_00_tensor0)[0]
        gradients_zero0 = C_factor[0] * gradients_zero0 + D_factor[0] #Chain rule / Scale to 0,1

        gradients_zero1 =  K.gradients(y_00_pred, x_00_tensor1)[0]
        gradients_zero1 = C_factor[1] * gradients_zero1 + D_factor[1] #Chain rule / Scale to 0,1

        gradients_zero2 =  K.gradients(y_00_pred, x_00_tensor2)[0]
        gradients_zero2 = C_factor[2] * gradients_zero2 + D_factor[2] #Chain rule / Scale to 0,1

        gradients_zero3 =  K.gradients(y_00_pred, x_00_tensor3)[0]
        gradients_zero3 = C_factor[3] * gradients_zero3 + D_factor[3] #Chain rule / Scale to 0,1

        gradients_zero4 =  K.gradients(y_00_pred, x_00_tensor4)[0]
        gradients_zero4 = C_factor[4] * gradients_zero4 + D_factor[4] #Chain rule / Scale to 0,1

        gradients_zero5 =  K.gradients(y_00_pred, x_00_tensor5)[0]
        gradients_zero5 = C_factor[5] * gradients_zero5 + D_factor[5] #Chain rule / Scale to 0,1

        #Stress loss predition at zero strain
        gradients_zero_point0 = keras.losses.mean_squared_error(gradients_zero0,grad_00_tensor0)

        gradients_zero_point1 = keras.losses.mean_squared_error(gradients_zero1,grad_00_tensor1)

        gradients_zero_point2 = keras.losses.mean_squared_error(gradients_zero2,grad_00_tensor2)

        gradients_zero_point3 = keras.losses.mean_squared_error(gradients_zero3,grad_00_tensor3)

        gradients_zero_point4 = keras.losses.mean_squared_error(gradients_zero4,grad_00_tensor4)

        gradients_zero_point5 = keras.losses.mean_squared_error(gradients_zero5,grad_00_tensor5)

        gradients_zero_point = gradients_zero_point0 + gradients_zero_point1 + gradients_zero_point2 + gradients_zero_point3 + gradients_zero_point4 + gradients_zero_point5 

        #Total loss prediction
        loss = 1. * grad_part + 1.* zero_point + 1. *  gradients_zero_point 

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

        y_00_tensor = K.constant(np.zeros((model_output_size,1)))
        grad_00_tensor0 = K.constant(grad_00[:,0].reshape(-1,1))
        grad_00_tensor1 = K.constant(grad_00[:,1].reshape(-1,1))
        grad_00_tensor2 = K.constant(grad_00[:,2].reshape(-1,1))
        grad_00_tensor3 = K.constant(grad_00[:,3].reshape(-1,1))
        grad_00_tensor4 = K.constant(grad_00[:,4].reshape(-1,1))
        grad_00_tensor5 = K.constant(grad_00[:,5].reshape(-1,1))

        y_00_pred = model([x_00_tensor0, x_00_tensor1, x_00_tensor2, x_00_tensor3, x_00_tensor4, x_00_tensor5, x_00_tensor])

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

        y_00_tensor = K.constant(np.zeros((model_output_size,1)))
        grad_00_tensor0 = K.constant(grad_00[:,0].reshape(-1,1))
        grad_00_tensor1 = K.constant(grad_00[:,1].reshape(-1,1))
        grad_00_tensor2 = K.constant(grad_00[:,2].reshape(-1,1))
        grad_00_tensor3 = K.constant(grad_00[:,3].reshape(-1,1))
        grad_00_tensor4 = K.constant(grad_00[:,4].reshape(-1,1))
        grad_00_tensor5 = K.constant(grad_00[:,5].reshape(-1,1))

        y_00_pred = model([x_00_tensor0, x_00_tensor1, x_00_tensor2, x_00_tensor3, x_00_tensor4, x_00_tensor5, x_00_tensor])

        gradients_zero0 =  K.gradients(y_00_pred, x_00_tensor0)[0]
        gradients_zero0 = C_factor[0] * gradients_zero0 + D_factor[0]

        gradients_zero1 =  K.gradients(y_00_pred, x_00_tensor1)[0]
        gradients_zero1 = C_factor[1] * gradients_zero1 + D_factor[1]

        gradients_zero2 =  K.gradients(y_00_pred, x_00_tensor2)[0]
        gradients_zero2 = C_factor[2] * gradients_zero2 + D_factor[2]

        gradients_zero3 =  K.gradients(y_00_pred, x_00_tensor3)[0]
        gradients_zero3 = C_factor[3] * gradients_zero3 + D_factor[3]

        gradients_zero4 =  K.gradients(y_00_pred, x_00_tensor4)[0]
        gradients_zero4 = C_factor[4] * gradients_zero4 + D_factor[4]

        gradients_zero5 =  K.gradients(y_00_pred, x_00_tensor5)[0]
        gradients_zero5 = C_factor[5] * gradients_zero5 + D_factor[5]

        gradients_zero_point0 = keras.losses.mean_squared_error(gradients_zero0,grad_00_tensor0)

        gradients_zero_point1 = keras.losses.mean_squared_error(gradients_zero1,grad_00_tensor1)

        gradients_zero_point2 = keras.losses.mean_squared_error(gradients_zero2,grad_00_tensor2)

        gradients_zero_point3 = keras.losses.mean_squared_error(gradients_zero3,grad_00_tensor3)

        gradients_zero_point4 = keras.losses.mean_squared_error(gradients_zero4,grad_00_tensor4)

        gradients_zero_point5 = keras.losses.mean_squared_error(gradients_zero5,grad_00_tensor5)

        gradients_zero_point = gradients_zero_point0 + gradients_zero_point1 + gradients_zero_point2 + gradients_zero_point3 + gradients_zero_point4 + gradients_zero_point5 

        gradients_zero_point = gradients_zero_point0 + gradients_zero_point1 + gradients_zero_point2 + gradients_zero_point3 + gradients_zero_point4 + gradients_zero_point5  
        gradients_zero_point = K.mean(gradients_zero_point,axis = 0)

        return gradients_zero_point
    return grad_zero


#Define TF gradient function
def getgradient(x):
    return tf.gradients(x[1], x[0])



# Forward Mapping Model - - - - - - - - - - - - - - - - - - - -
width = 100 # NN hidden layer neurons

# Define NN strain inputs separately to take individual derivatives if needed
e11 = Input((1,)) #1
e22 = Input((1,)) #2
e33 = Input((1,)) #3
e12 = Input((1,)) #4
e23 = Input((1,)) #5
e13 = Input((1,)) #6

# Define NN true gradient inputs -- will be used to calulcate loss
true_gradient = Input((6,))

#Concatenate strain inputs
conc_inp = Concatenate(axis=-1)([e11,e22,e33,e12,e23,e13])

#Define NN architecture
h = Dense(width, activation="relu")(conc_inp)
h = Multiply()([h,h])
h = Multiply()([h,h])
h = Dense(width, activation="relu")(h)
h = Multiply()([h,h])
out = Dense(1, activation="linear", name = 'for')(h)

#Define NN model
model = Model(inputs=[e11,e22,e33,e12,e23,e13,true_gradient], outputs=[out] )

#Define gradient function
gradient = Lambda(getgradient,output_shape=(6,))([conc_inp,out]) 

#Split stress gradient to components
gradient_split = Lambda( lambda x: tf.split(x,num_or_size_splits=6,axis=1) )(gradient) 

s11 = gradient_split[0] #1
s22 = gradient_split[1] #2
s33 = gradient_split[2] #3
s12 = gradient_split[3] #4
s23 = gradient_split[4] #5
s13 = gradient_split[5] #6

gradient_scaled = C_factor * gradient + D_factor #Chain rule / Scale to 0,1

#Calculate true and predicted stress gradient difference
gradient_diff = gradient_scaled - true_gradient

#Define optimizer
opt = Nadam(lr = 0.01)

#Compile model
model.compile(optimizer='nadam', loss=grad_diff(gradient_diff),metrics = [grad_loss(gradient_diff), energy_zero_point_check() ,gradient_zero_point_check()])

# Print model summary
model.summary()

from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

#Define model callbacks
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.90, patience=5, min_lr=1e-9,verbose = 1)
earlystop = EarlyStopping(monitor='loss', patience=1000)
checkpoint = ModelCheckpoint('trained_models/energy_model.h5',monitor='loss',verbose=1,save_best_only=True)

#Train model
history = model.fit([E11[:train_index],E22[:train_index],E33[:train_index],E12[:train_index],E23[:train_index],E13[:train_index],Y[:train_index]], fake_potential[:train_index], 
epochs=1000, batch_size = 512, callbacks = [reduce_lr,checkpoint,earlystop], shuffle = True, 
validation_data=[[E11[train_index:],E22[train_index:],E33[train_index:],E12[train_index:],E23[train_index:],E13[train_index:],Y[train_index:]], fake_potential[train_index:] ]) 

print("Finished training SE model.")

