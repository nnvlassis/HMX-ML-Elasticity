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
import matplotlib
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

np.random.seed(5)

K.set_floatx('float64')
K.set_epsilon(1e-16)

#Load model
model = load_model("./trained_models/energy_model_PF.h5", compile = False)

#Set learning phase to False to freeze weights
K.set_learning_phase(0)

#Define gradient function
gradient_node = tf.gradients(model.output, model.layers[9].output)
session = K.get_session()

#Plotting function
def plot_multicurve(inp_strain_paths, pred_stress_paths, true_strain_paths,true_stress_paths, inp_index, name,title):
    strain_labels = [r"$F_{11}$",r"$F_{22}$",r"$F_{33}$",r"$F_{12}$",r"$F_{23}$",r"$F_{13}$",r"$F_{21}$",r"$F_{32}$",r"$F_{32}$" ]
    stress_labels = [r"$P_{11}$",r"$P_{22}$",r"$P_{33}$",r"$P_{12}$",r"$P_{23}$",r"$P_{13}$",r"$P_{21}$",r"$P_{32}$",r"$P_{31}$" ]
    m = ["s--", "o--", "d--","^--","v--","*--","x--","<--",">--"]
    m2 = ["s", "o", "d","^","v","*"]
    plt.axhline(0,color = 'k', alpha  = 0.5)
    for k in range(pred_stress_paths.shape[1]):

        p = plt.plot(true_strain_paths[:,inp_index], true_stress_paths[:,k],lw =1 ) 

        color = p[0].get_color()
        plt.plot(inp_strain_paths[:,inp_index], pred_stress_paths[:,k], m[k],label = stress_labels[k],markevery = 0.1, markersize = 10)


    plt.xlabel(strain_labels[inp_index], fontsize = 22)
    plt.ylabel(r"$P_{iJ}$ ($\frac{\partial \psi}{\partial F_{iJ}}$) (GPa)", fontsize = 22)
    plt.xticks(fontsize=17, rotation=0)
    plt.yticks(fontsize=17, rotation=0)
    plt.legend()

    plt.grid()
    plt.title(title, fontsize = 17)

    plt.tight_layout()
    # plt.savefig(name)
    plt.show()


#Plotting function
def plot_all(data,inp_index, title,name):
    #Get true strain and stress values
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

    stress_data_0 = np.zeros((P11.shape[0],9))
    stress_data_0[:,0] = P11
    stress_data_0[:,1] = P22
    stress_data_0[:,2] = P33
    stress_data_0[:,3] = P12
    stress_data_0[:,4] = P23
    stress_data_0[:,5] = P13
    stress_data_0[:,6] = P21
    stress_data_0[:,7] = P32
    stress_data_0[:,8] = P31

    strain_data_0 = np.zeros((P11.shape[0],9))
    strain_data_0[:,0] = F11 
    strain_data_0[:,1] = F22 
    strain_data_0[:,2] = F33 
    strain_data_0[:,3] = F12
    strain_data_0[:,4] = F23
    strain_data_0[:,5] = F13
    strain_data_0[:,6] = F21
    strain_data_0[:,7] = F32
    strain_data_0[:,8] = F31

    XX = strain_data_0
    #Calculate predicted gradient
    evaluated_grads0 = session.run(gradient_node,feed_dict={model.input[0]: XX[:,0].reshape(-1,1),model.input[1]: XX[:,1].reshape(-1,1),model.input[2]: XX[:,2].reshape(-1,1),
        model.input[3]: XX[:,3].reshape(-1,1),model.input[4]: XX[:,4].reshape(-1,1),model.input[5]: XX[:,5].reshape(-1,1),model.input[6]: XX[:,6].reshape(-1,1),model.input[7]: XX[:,7].reshape(-1,1),model.input[8]: XX[:,8].reshape(-1,1)})[0]    # evaluated_grads1 = session.run(gradient_node,feed_dict={model.input[0]: X1, model.input[1]: Xt1.reshape(-1,1)})[0]

    energy = model.predict([XX[:,0],XX[:,1],XX[:,2],XX[:,3],XX[:,4],XX[:,5],XX[:,6],XX[:,7],XX[:,8]])

    plot_multicurve(XX, evaluated_grads0*100., strain_data_0, stress_data_0, inp_index = inp_index, name = name , title = title) #invert simple stress scaling

# ------------------------------
# Plot predictions
# ------------------------------
data = pd.read_csv("./MD_simulation_curves/compression_along_x_and_y_Data_noisy_PF.csv")
plot_all(data,0,"Compression along x and y axes","compression_along_x_and_y_PF.pdf")

data = pd.read_csv("./MD_simulation_curves/compression_along_y_and_z_Data_noisy_PF.csv")
plot_all(data,1,"Compression along y and z axes","compression_along_y_and_PF.pdf")

data = pd.read_csv("./MD_simulation_curves/compression_along_x_and_z_Data_noisy_PF.csv")
plot_all(data,0,"Compression along x and z axes","compression_along_x_and_PF.pdf")


data1 = pd.read_csv("./MD_simulation_curves/shear_along_neg_xy_Data_noisy_PF.csv").iloc[::-1]
data2 = pd.read_csv("./MD_simulation_curves/shear_along_xy_Data_noisy_PF.csv")
data = pd.concat([data1, data2], ignore_index=True)
plot_all(data,3,"Shear along xy axes","shear_along_xy_PF.pdf")


data1 = pd.read_csv("./MD_simulation_curves/shear_along_neg_yz_Data_noisy_PF.csv").iloc[::-1]
data2 = pd.read_csv("./MD_simulation_curves/shear_along_yz_Data_noisy_PF.csv")
data = pd.concat([data1, data2], ignore_index=True)
plot_all(data,4,"Shear along yz axes","shear_along_yz_PF.pdf")


data1 = pd.read_csv("./MD_simulation_curves/shear_along_neg_xz_Data_noisy_PF.csv").iloc[::-1]
data2 = pd.read_csv("./MD_simulation_curves/shear_along_xz_Data_noisy_PF.csv")
data = pd.concat([data1, data2], ignore_index=True)
plot_all(data,5,"Shear along xz axes","shear_along_xz_PF.pdf")


data1 = pd.read_csv("./MD_simulation_curves/compression_along_x_Data_noisy_PF.csv").iloc[::-1]
data2 = pd.read_csv("./MD_simulation_curves/tension_along_x_Data_noisy_PF.csv")
data = pd.concat([data1, data2], ignore_index=True)
plot_all(data,0,"Tension along x axis","uniaxial_along_x_PF.pdf")


data1 = pd.read_csv("./MD_simulation_curves/compression_along_y_Data_noisy_PF.csv").iloc[::-1]
data2 = pd.read_csv("./MD_simulation_curves/tension_along_y_Data_noisy_PF.csv")
data = pd.concat([data1, data2], ignore_index=True)
plot_all(data,1,"Tension along y axis","uniaxial_along_y_PF.pdf")


data1 = pd.read_csv("./MD_simulation_curves/compression_along_z_Data_noisy_PF.csv").iloc[::-1]
data2 = pd.read_csv("./MD_simulation_curves/tension_along_z_Data_noisy_PF.csv")
data = pd.concat([data1, data2], ignore_index=True)
plot_all(data,2,"Tension along z axis","uniaxial_along_z_PF.pdf")




exit()

