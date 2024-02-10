import numpy as np
import pandas as pd

#Load csv data
data = pd.read_csv("../HMX_Data/Data_denoised_SE.csv")

#Assign strain and stress components
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

stress_data = np.zeros((Sxx.shape[0],6))
stress_data[:,0] = Sxx
stress_data[:,1] = Syy
stress_data[:,2] = Szz
stress_data[:,3] = Sxy
stress_data[:,4] = Syz
stress_data[:,5] = Sxz

strain_data = np.zeros((Sxx.shape[0],6))
strain_data[:,0] = e11
strain_data[:,1] = e22
strain_data[:,2] = e33
strain_data[:,3] = e12
strain_data[:,4] = e23
strain_data[:,5] = e13

#Save data as .npy
np.save("SE_stress_data_saved.npy", stress_data)
np.save("SE_strain_data_saved.npy", strain_data)
print("Saved SE model data.")

