# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 01:29:06 2023

@author: nguye
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as sts

#Position moyen en fonction de la tension

df = pd.read_csv(r'C:\Users\nguye\Desktop\Master 2\Mémoire\Resultats\Position moyenne.csv')

#print(df['Volt'],df['Distance moyenne'])
#moyenne en fonction de la tension

Volt_variation = 100
Volt_step = 7
#print(df.shape[0])
i=0
x=[]
y = []
y_errorbar = []
for i in range(0, Volt_step):
    data_per_voltage = []
    j = 0
    k=0
    for j in range(0, df.shape[0]):
        if df['Volt'][j] == (i)*Volt_variation :
            data_per_voltage.append(df['Distance moyenne'][j])
            k+=1
    
    x.append((i)*Volt_variation)
    y.append(sts.mean(data_per_voltage))
    y_errorbar.append(sts.stdev(data_per_voltage))


plt.title("Distance moyenne en fonction de la tension")
plt.xlabel("Tension(V)")
plt.ylabel("Distance moyenne (cm)")
plt.errorbar(x,y,y_errorbar,capsize=4)

plt.savefig(r"C:\Users\nguye\Desktop\Master 2\Mémoire\Resultats\Position moyenne")
plt.show()