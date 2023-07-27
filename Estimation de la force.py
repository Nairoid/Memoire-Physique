# -*- coding: utf-8 -*-

import os
import os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as sts
import csv
import math



#plt font size
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rc('figure', titlesize= BIGGER_SIZE)
plt.rc('axes', titlesize= BIGGER_SIZE) 
plt.rc('axes', labelsize=MEDIUM_SIZE) 
plt.rc('xtick', labelsize=MEDIUM_SIZE)
plt.rc('ytick', labelsize=MEDIUM_SIZE)
plt.rcParams['figure.figsize'] = (8.5, 6)
plt.rcParams["figure.dpi"] = 144

#Fichier Delta_moy_a0 final
df_Delta_moy = pd.read_csv(r'C:\Users\nguye\Desktop\Master 2\Mémoire\Resultats\Delta_moy_a0.csv')

#Calcul theta


df_Delta_moy['dz/dx'] = 0.29*0.01/(1.7*0.01)**2 * df_Delta_moy['Delta_moyen']*0.01/np.sqrt(1-(df_Delta_moy['Delta_moyen']*0.01)**2/((1.7*0.01)**2))
df_Delta_moy['Theta'] = np.arccos(1/np.sqrt(1+(df_Delta_moy['dz/dx'])**2))

#Force poids
df_Delta_moy['Force poids'] = np.sin(df_Delta_moy['Theta'])*9.81*0.00421



#graphique menisque

x1 = []

x1 = np.arange(-2.,2. ,0.01)

def ellipse(X) :
    return -np.sqrt((0.29)**2*(1-(X**2/(1.7)** 2)))+0.29

def derivee(X) :
    return (0.29/(1.7)**2 *0.163/np.sqrt(1-(0.163)**2/((1.7)**2)))*(X-0.163*0.7**2)
    
plt.plot(x1,ellipse(x1),label = 'Menisque')
#plt.plot(x1,derivee(x1),label = 'Tangente')
plt.xlim(-2.,2.)
plt.ylim(0,1)
#plt.title("Hauteur du menisque h en fonction de la distance $\Delta$",y=1.0, pad=+15)
plt.xlabel("$\Delta$ (cm) ")
plt.ylabel("h  (cm)")
#☺plt.legend(bbox_to_anchor = (1.0, 1), loc = 'upper right')

plt.savefig(r"C:\Users\nguye\Desktop\Master 2\Mémoire\Resultats\Menisque")
plt.show()


#graphique force poids
Volt_variation = 100
Volt_step = 7
Number_of_experiment = 6

i=0
x=[]
y = []
y_errorbar = []
for i in range(0, Volt_step):
    data_per_voltage = []
    j = 0
    k=0
    for j in range(0, df_Delta_moy.shape[0]):
        if df_Delta_moy['Volt'][j] == (i)*Volt_variation :
            data_per_voltage.append(df_Delta_moy['Force poids'][j])
            k+=1
    
    x.append(round((i)*Volt_variation/3.4,0))
    y.append(sts.mean(data_per_voltage))
    y_errorbar.append(sts.stdev(data_per_voltage))


#plt.title("Force électroosmotique F$_{eo}$ moyenne en fonction de la tension")
plt.ticklabel_format(axis='y', style='sci',scilimits=(0,4))
plt.xlabel("$E_{ext}$(V cm$^{-1}$)")
plt.ylabel("<F$_{eo}$> (N)")
plt.errorbar(x,y,y_errorbar,capsize=4)

plt.savefig(r"C:\Users\nguye\Desktop\Master 2\Mémoire\Resultats\Force moyenne")
plt.show()
