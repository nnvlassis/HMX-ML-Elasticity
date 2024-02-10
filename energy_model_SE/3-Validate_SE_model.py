
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
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K


np.random.seed(5)

K.set_floatx('float64')
K.set_epsilon(1e-16)

# Plotting function
def plot_multicurve(inp_strain_paths, pred_stress_paths, true_strain_paths,true_stress_paths, inp_index, name,title):
    strain_labels = [r"$E_{11}$",r"$E_{22}$",r"$E_{33}$",r"$E_{12}$",r"$E_{23}$",r"$E_{13}$" ]
    stress_labels = [r"$S_{11}$",r"$S_{22}$",r"$S_{33}$",r"$S_{12}$",r"$S_{23}$",r"$S_{13}$" ]
    m = ["s--", "o--", "d--","^--","v--","*--"]
    m2 = ["s", "o", "d","^","v","*"]
    plt.axhline(0,color = 'k', alpha  = 0.5)
    plt.axvline(0,color = 'k', alpha  = 0.5)
    for k in range(pred_stress_paths.shape[1]):
        p = plt.plot(true_strain_paths[:,inp_index], true_stress_paths[:,k],lw =1 ) 

        color = p[0].get_color()
        plt.plot(inp_strain_paths[:,inp_index], pred_stress_paths[:,k], m[k],label = stress_labels[k],markevery = 0.1, markersize = 10)


    plt.xlabel(strain_labels[inp_index], fontsize = 22)
    plt.ylabel(r"$S_{ij}$ ($\frac{\partial \psi}{\partial E_{ij}}$) (GPa)", fontsize = 22)
    plt.xticks(fontsize=17, rotation=0)
    plt.yticks(fontsize=17, rotation=0)

    plt.legend(fontsize = 14,ncol=1)
    plt.grid()
    plt.title(title, fontsize = 17)
    plt.tight_layout()
    # plt.savefig(name)
    plt.show()


import matplotlib
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

#Load model
model = load_model("./trained_models/energy_model.h5", compile = False)

#Load scalers
INPUT_Scaler = joblib.load("./trained_models/INPUT_Scaler.pkl")
OUTPUT_Scaler = joblib.load("./trained_models/OUTPUT_Scaler.pkl")

#Set learning phase to False to freeze weights
K.set_learning_phase(0)

#Define gradient function
gradient_node = tf.gradients(model.output, model.layers[6].output)
session = K.get_session()


#Plotting function
def plot_all(data,inp_index,title, name):

    #Get true strain and stress values
    e11 = data["e11"].values
    e22 = data["e22"].values
    e33 = data["e33"].values
    e12 = data["e12"].values
    e23 = data["e23"].values
    e13 = data["e13"].values

    Sxx = data["s11"].values
    Syy = data["s22"].values
    Szz = data["s33"].values
    Sxy = data["s12"].values
    Syz = data["s23"].values
    Sxz = data["s13"].values


    stress_data_0 = np.zeros((Sxx.shape[0],6))
    stress_data_0[:,0] = Sxx
    stress_data_0[:,1] = Syy
    stress_data_0[:,2] = Szz
    stress_data_0[:,3] = Sxy
    stress_data_0[:,4] = Syz
    stress_data_0[:,5] = Sxz


    strain_data_0 = np.zeros((Sxx.shape[0],6))
    strain_data_0[:,0] = e11
    strain_data_0[:,1] = e22
    strain_data_0[:,2] = e33
    strain_data_0[:,3] = e12
    strain_data_0[:,4] = e23
    strain_data_0[:,5] = e13

    #Scale input strain
    XX = INPUT_Scaler.transform(strain_data_0)
    #Calculate predicted gradient
    evaluated_grads0 = session.run(gradient_node,feed_dict={model.input[0]: XX[:,0].reshape(-1,1),model.input[1]: XX[:,1].reshape(-1,1),model.input[2]: XX[:,2].reshape(-1,1),
        model.input[3]: XX[:,3].reshape(-1,1),model.input[4]: XX[:,4].reshape(-1,1),model.input[5]: XX[:,5].reshape(-1,1)})[0]   

    #Invert input scaling
    XX = INPUT_Scaler.inverse_transform(XX)

    plot_multicurve(XX, evaluated_grads0, strain_data_0, stress_data_0, inp_index = inp_index, name = name , title = title)


# ------------------------------
# Plot predictions
# ------------------------------
data = pd.read_csv("./MD_simulation_curves/compression_along_x_and_y_Data_noisy_SE.csv")
plot_all(data,0,"Compression along x and y axes","compression_along_x_and_y.pdf")

data = pd.read_csv("./MD_simulation_curves/compression_along_y_and_z_Data_noisy_SE.csv")
plot_all(data,1,"Compression along y and z axes","compression_along_y_and_z.pdf")

data = pd.read_csv("./MD_simulation_curves/compression_along_x_and_z_Data_noisy_SE.csv")
plot_all(data,0,"Compression along x and z axes","compression_along_x_and_z.pdf")


data1 = pd.read_csv("./MD_simulation_curves/shear_along_neg_xy_Data_noisy_SE.csv").iloc[::-1]
data2 = pd.read_csv("./MD_simulation_curves/shear_along_xy_Data_noisy_SE.csv")
data = pd.concat([data1, data2], ignore_index=True)
plot_all(data,3,"Shear along xy axes","shear_along_xy.pdf")


data1 = pd.read_csv("./MD_simulation_curves/shear_along_neg_yz_Data_noisy_SE.csv").iloc[::-1]
data2 = pd.read_csv("./MD_simulation_curves/shear_along_yz_Data_noisy_SE.csv")
data = pd.concat([data1, data2], ignore_index=True)
plot_all(data,4,"Shear along yz axes","shear_along_yz.pdf")


data1 = pd.read_csv("./MD_simulation_curves/shear_along_neg_xz_Data_noisy_SE.csv").iloc[::-1]
data2 = pd.read_csv("./MD_simulation_curves/shear_along_xz_Data_noisy_SE.csv")
data = pd.concat([data1, data2], ignore_index=True)
plot_all(data,5,"Shear along xz axes","shear_along_xz.pdf")


data1 = pd.read_csv("./MD_simulation_curves/compression_along_x_Data_noisy_SE.csv").iloc[::-1]
data2 = pd.read_csv("./MD_simulation_curves/tension_along_x_Data_noisy_SE.csv")
data = pd.concat([data1, data2], ignore_index=True)
plot_all(data,0,"Tension along x axis","uniaxial_along_x.pdf")


data1 = pd.read_csv("./MD_simulation_curves/compression_along_y_Data_noisy_SE.csv").iloc[::-1]
data2 = pd.read_csv("./MD_simulation_curves/tension_along_y_Data_noisy_SE.csv")
data = pd.concat([data1, data2], ignore_index=True)
plot_all(data,1,"Tension along y axis","uniaxial_along_y.pdf")


data1 = pd.read_csv("./MD_simulation_curves/compression_along_z_Data_noisy_SE.csv").iloc[::-1]
data2 = pd.read_csv("./MD_simulation_curves/tension_along_z_Data_noisy_SE.csv")
data = pd.concat([data1, data2], ignore_index=True)
plot_all(data,2,"Tension along z axis","uniaxial_along_z.pdf")











