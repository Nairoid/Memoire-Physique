# -*- coding: utf-8 -*-
"""

@author: nguye
"""
import os
import os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as sts
import csv
from numpy.fft import fft, ifft
import math

from sklearn.linear_model import LinearRegression

#plt font size
SMALL_SIZE = 12
MEDIUM_SIZE = 20
BIGGER_SIZE = 22

plt.rc('figure', titlesize= BIGGER_SIZE)
plt.rc('axes', titlesize= BIGGER_SIZE) 
plt.rc('axes', labelsize=MEDIUM_SIZE) 
plt.rc('xtick', labelsize=MEDIUM_SIZE)
plt.rc('ytick', labelsize=MEDIUM_SIZE)
plt.rcParams['figure.figsize'] = (8.5, 6)
plt.rcParams["figure.dpi"] = 144


#Position moyenne, creation of the beginning of the doc
with open(r'C:\Users\nguye\Desktop\Master 2\Mémoire\Resultats\Position moyenne.csv', 'w') as f:
    #create the csv writer
    writer = csv.writer(f)
    
    #Cree le point (0,0)
    writer.writerow(('Volt','Distance_moyenne','Ecart_type_moyen'))
    writer.writerow((0,0,0))
    writer.writerow((0,0,0))
    
#Vitesse moyenne, creation of the beginning of the doc
with open(r'C:\Users\nguye\Desktop\Master 2\Mémoire\Resultats\Vitesse moyenne.csv', 'w') as f:
    #create the csv writer
    writer = csv.writer(f)
    
    #Cree le point (0,0)
    writer.writerow(('Volt','Vitesse_moyenne','Ecart_type_moyen'))
    writer.writerow((0,0,0))
    writer.writerow((0,0,0))
    
#Acceleration moyenne, creation of the beginning of the doc
with open(r'C:\Users\nguye\Desktop\Master 2\Mémoire\Resultats\Acceleration moyenne.csv', 'w') as f:
    #create the csv writer
    writer = csv.writer(f)
    
    #Cree le point (0,0)
    writer.writerow(('Volt','Acc_moyenne','Ecart_type_moyen'))
    writer.writerow((0,0,0))
    writer.writerow((0,0,0))

#Delta_moy_a0 moyenne, creation of the beginning of the doc
with open(r'C:\Users\nguye\Desktop\Master 2\Mémoire\Resultats\Delta_moy_a0.csv', 'w') as f:
    #create the csv writer
    writer = csv.writer(f)
    
    #Cree le point (0,0)
    writer.writerow(('Volt','Delta_moyen'))
    writer.writerow((0,0))
    writer.writerow((0,0))

Volt_variation = 100
Volt_step = 7
Number_of_experiment = 6

for s in range(0,Volt_step):
    for t in range(0,Number_of_experiment):
        '''
        #skip 400V/600V 1-2-3
        if ((s==4 or s==6) and (t==0 or t==1 or t==2)):
            continue
        '''
        #data
        directing_file =r'C:\Users\nguye\Desktop\Master 2\Mémoire\Video_data'
        Volt = str(s) + '00'
        Number = '_' + str(t+1)
        
        opening_file = os.path.join(directing_file, 'Results_'+ Volt +'V' + Number + '.csv')
        
        #intensite du champ electrique
        intensite_E_float = float(Volt)/3.4  #diamètre du becher = 3.4 cm
        intensite_E = "{:.0f}".format(intensite_E_float)
        
        #si le fichier n'existe pas
        if os.path.isfile(opening_file) == 0 :
            print(opening_file + ' non trouvé')
            continue
        
        df = pd.read_csv(opening_file)
        
        
        x_ini = df.iloc[0]['XM']
        y_ini = df.iloc[0]['YM']
        
        df['X_corrected'] = df['XM'] - x_ini
        
        df['Y_corrected'] = df['YM'] - y_ini
        
        df['distance'] = np.sqrt(df['X_corrected']**2 + df['Y_corrected']**2)
                                 
        '''
        df['Voltage switch'] = (df['Area'] < 0.07)
        
        df['Corrected distance'] = df['distance']* df['Voltage switch']
        
        #Data with voltage
        pace =100
        n=0
        
        df['Voltage'] = 0
        
        for f in df['Frame']:  
            if f < len(df['Frame'])-100 :
                df['Voltage'][f] = 0 + n*pace
                
                if(df['Voltage switch'][f] == 1 and df['Voltage switch'][f+1] == 0 and df['Voltage switch'][f+2] == 0 and df['Voltage switch'][f+3] == 0 and df['Voltage switch'][f+4] == 0 ):
                    n+=1
        
        #Average and std
        m=0
        data = []
        voltage = []
        mean = []
        std = []
        for f in df['Frame']:
            if f < len(df['Frame'])-100 :
                if df['Voltage'][f] == 100+m*pace:
                    
                    data.append(df['Corrected distance'][f])
                    
                    
                    if(df['Voltage switch'][f] == 1 and df['Voltage switch'][f+1] == 0 and df['Voltage switch'][f+2] == 0 and df['Voltage switch'][f+3] == 0 and df['Voltage switch'][f+4] == 0 ):
                        voltage.append([df['Voltage'][f]])
                        mean.append(sts.mean(data))
                        std.append(sts.stdev(data))
                        m+=1
                        data = []
        '''                
        #Results file
        end_file = r'Resultats'
        save_file = os.path.join(r'C:\Users\nguye\Desktop\Master 2\Mémoire', end_file)
        
        '''#plot
        df.plot(x = 'Voltage' , y = 'Corrected distance', kind = 'scatter')
        plt.title("Distance cartesienne en fonction de la tension")
        
        plt.savefig(save_file + "_dis")
        plt.show()'''
        
        
        
        '''plt.title("Distance cartesienne moyenne en fonction de la tension")
        plt.errorbar(voltage, mean , std , linestyle='None', marker='o')
        
        plt.savefig(save_file + "_avgdis")
        plt.show()'''
        
         
        
        
        ################            X            #################
        #Distance x en fonction du temps
        df['Temps'] = df['Frame']* 0.1
        
        df.plot(x = 'Temps' , y = 'X_corrected', kind = 'scatter', s = 3)
        plt.title("Distance selon X en fonction du temps ("+ intensite_E +'V.$cm^{-1}$' + ")")
        plt.xlabel("Temps(s)")
        plt.ylabel("Distance selon X (cm)")
        
        final_file_X = os.path.join(save_file,'X', 'Results_'+ Volt + 'V' + Number )
        plt.savefig(final_file_X + "_X")
        plt.show()
        
        #Fourier transform X
        four_trans_X = fft(df['X_corrected'])
        N = len(four_trans_X)
        n = np.arange(N)
        sr = 10
        T = N/sr
        freq = n/T
        
        # Get the one-sided spectrum
        n_oneside = N//2
        # get the one side frequency
        f_oneside = freq[:n_oneside]
        '''
        print(f_oneside)
        print(np.abs(four_trans_X[:n_oneside]))'''
        
        plt.plot(f_oneside ,np.abs(four_trans_X[:n_oneside]))
        plt.title("FFT amplitude de X en fonction de la fréquence("+ intensite_E +'V.$cm^{-1}$' + ")")
        plt.xlabel("Freq(Hz)")
        plt.ylabel("FFT amplitude (cm^-1)")
        
        #zoom sur le graphe
        x_min = 0
        x_max = 1
        
        plt.xlim(x_min,x_max)
        
        final_file_FFTX = os.path.join(save_file,'FFT_X', 'Results_'+ Volt + 'V' + Number )
        plt.savefig(final_file_FFTX + "_FFT_X")
        plt.show()
        
        
        
        
        ###############################      Y  ##################
        #Distance y en fonction du temps
        
        df.plot(x = 'Temps' , y = 'Y_corrected', kind = 'scatter', s = 5)
        plt.title("Distance selon Y en fonction du temps (" + intensite_E +'V.$cm^{-1}$'  + ")")
        plt.xlabel("Temps(s)")
        plt.ylabel("Distance selon Y (cm)")
        
        final_file_Y = os.path.join(save_file,'Y', 'Results_'+ Volt + 'V' + Number )
        plt.savefig(final_file_Y + "_Y")
        plt.show()
        
        #Fourier transform X
        four_trans_Y = fft(df['Y_corrected'])
        
        plt.plot(f_oneside ,np.abs(four_trans_Y[:n_oneside]))
        plt.title("FFT de Y en fonction de la fréquence("+ intensite_E +'V.$cm^{-1}$' + ")")
        plt.xlabel("Freq(Hz)")
        plt.ylabel("FFT amplitude (cm^-1)")
        plt.xlim(x_min,x_max)
        
        final_file_FFTY = os.path.join(save_file,'FFT_Y', 'Results_'+ Volt + 'V' + Number )
        plt.savefig(final_file_FFTY +   "_FFT_Y")
        plt.show()
        
        ####################    DELTA               #############         
        #Distance en fonction du temps
        
        df.plot(x = 'Temps' , y = 'distance', kind = 'scatter', s = 5)
        plt.title("<$\Delta$> en fonction du temps ("+ intensite_E +'V.$cm^{-1}$' + ")")
        plt.xlabel("Temps(s)")
        plt.ylabel("Delta moyen (cm)")
        
        final_file_deltamoy = os.path.join(save_file,'Delta_moy', 'Results_'+ Volt + 'V' + Number )
        plt.savefig(final_file_deltamoy   + "_Delta_moy")
        plt.show()
        
        #Fourier transform Delta
        four_trans_delta = fft(df['distance'])
        
        plt.plot(f_oneside ,np.abs(four_trans_delta[:n_oneside]))
        plt.title("FFT de $\Delta$ en fonction de la fréquence("+ intensite_E +'V.$cm^{-1}$' + ")")
        plt.xlabel("Freq(Hz)")
        plt.ylabel("FFT amplitude (cm^-1)")
        plt.xlim(x_min,x_max)
        
        final_file_FFTdelta = os.path.join(save_file,'FFT_delta', 'Results_'+ Volt + 'V' + Number )
        plt.savefig(final_file_FFTdelta   + "_FFT_delta")
        plt.show()
        
        #ecriture dans le doc position moyenne    
        with open(r'C:\Users\nguye\Desktop\Master 2\Mémoire\Resultats\Position moyenne.csv', 'a') as f:
            
            writer = csv.writer(f)    
            writer.writerow((Volt,df['distance'].mean(),df['distance'].std()))
            
        
        
        
        #####################            Trajectoire                    ##################
        
        plt.title("Trajectoire ("+ intensite_E +'V.$cm^{-1}$'+ ")",y=1.0, pad=+15)
        plt.plot(df['X_corrected'],df['Y_corrected'])    
        plt.xlabel("X (cm)")
        plt.ylabel("Y (cm)")
        
        final_file_traj = os.path.join(save_file,'traj', 'Results_'+ Volt + 'V' + Number)
        plt.savefig(final_file_traj + "_traj")
        plt.show()
        
        ############## Angle correction ###################
        
        
        #Angle moyen
        somme_sin = 0
        somme_cos = 0
        angle_moy = 0
        for a in df['Angle'] :
            #On va virer les angles
            somme_sin += math.sin(a*math.pi/180)
            somme_cos += math.cos(a*math.pi/180)
        sin_moy = somme_sin/len(df['Angle'])
        cos_moy = somme_cos/len(df['Angle'])
        
        #print('Results_', Volt , 'V' , Number , ' sin moy = ' , sin_moy)
        #print('Results_', Volt , 'V' , Number , ' cos moy = ' , cos_moy)

        
        #Circular mean
        if sin_moy < 0 and cos_moy > 0 :
            angle_moy = math.atan(sin_moy/cos_moy)/math.pi*180 + 180
        elif cos_moy < 0 and sin_moy > 0 :
            angle_moy = math.atan(sin_moy/cos_moy)/math.pi*180 + 180
        else :
            angle_moy = math.atan(sin_moy/cos_moy)/math.pi*180
            
        #Coordonnées sous rotation
        df['X_rotation'] = df['X_corrected'] * math.cos(angle_moy*math.pi/180) - df['Y_corrected'] * math.sin(angle_moy*math.pi/180)
        df['Y_rotation'] = df['X_corrected'] * math.sin(angle_moy*math.pi/180) + df['Y_corrected'] * math.cos(angle_moy*math.pi/180)
        
        #print('Results_', Volt , 'V' , Number , ' angle moy = ' , angle_moy)
        
        #Graphique angle + moyenne

        
        df.plot(x = 'Temps', y = 'Angle', kind = 'line')
       
        #plt.title("$\\alpha$ en fonction du temps ("+ intensite_E +' V.$cm^{-1}$' + ")",y=1.0, pad=+15)
        
        plt.axhline(y = angle_moy, color = 'red', label = '<$\\alpha$>')
        plt.xlabel("Temps (s) ")
        plt.ylabel("Angle (°)")
        plt.ylim(0,180)
        plt.legend(bbox_to_anchor = (1.0, 1), fontsize="20",loc = 'upper right')
        
        final_file_FFTdelta = os.path.join(save_file,'Angle', 'Results_'+ Volt + 'V' + Number )
        plt.savefig(final_file_FFTdelta + "_Angle")
        plt.show()
        
        ############# X rotation ###########
        #Distance x rotation en fonction du temps
        
        df.plot(x = 'Temps' , y = 'X_rotation')
        #plt.title("Distance selon X' en fonction du temps (" + intensite_E +'V.$cm^{-1}$' + ")")
        plt.xlabel("Temps(s)")
        plt.ylabel("Coordonnée selon X' (cm)")
        
        final_file_Y = os.path.join(save_file,'X_prime', 'Results_'+ Volt + 'V' + Number )
        plt.savefig(final_file_Y + "_X_prime")
        plt.show()
        
        #Fourier transform X
        four_trans_X = fft(df['X_rotation'])
        
        plt.plot(f_oneside ,np.abs(four_trans_Y[:n_oneside]))
        #plt.title("FFT de X' en fonction de la fréquence("+ intensite_E +'V.$cm^{-1}$'+ ")")
        plt.xlabel("Freq(Hz)")
        plt.ylabel("Amplitude de la FFT (cm.s)")
        plt.xlim(x_min,x_max)
        
        final_file_FFTX = os.path.join(save_file,'FFT_X_rotation', 'Results_'+ Volt + 'V' + Number )
        plt.savefig(final_file_FFTX +   "_FFT_X_rotation")
        plt.show()
        
        #Trajectoire (rotation)
        plt.rcParams['figure.figsize'] = (8, 6)
        #plt.title("Trajectoire ("+ intensite_E +' V.$cm^{-1}$' + ")")
        plt.plot(df['X_rotation'],df['Y_rotation'])
        plt.xlabel("X' (cm)")
        plt.ylabel("Y' (cm)")
        
        #"centre de masse"
        x_m = df['X_rotation'].mean()
        y_m = df['Y_rotation'].mean()
        plt.plot(x_m, y_m, marker="o", markersize=8, markeredgecolor="red", markerfacecolor="red",label = '(<X>,<Y>)')
        plt.legend(bbox_to_anchor = (1.0, 1),fontsize="20", loc = 'upper right')
        
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        
        final_file_traj = os.path.join(save_file,'traj_rotation', 'Results_'+ Volt + 'V' + Number)
        plt.savefig(final_file_traj + "_traj_rotation")
        plt.show()
        
        #Angle (rotation)
        df['Angle_rotation'] = df['Angle'] - angle_moy
        df.plot(x = 'Temps', y = 'Angle_rotation', kind = 'line')
        plt.title("Angle en fonction du temps("+ intensite_E +'V.$cm^{-1}$' + ")")
        plt.xlabel("Temps (s) ")
        plt.ylabel("Angle' (°)")
        
        final_file_FFTdelta = os.path.join(save_file,'Angle_rotation', 'Results_'+ Volt + 'V' + Number )
        plt.savefig(final_file_FFTdelta + "_Angle_rotation")
        plt.show()
        
        #Vitesse
        df['Vitesse'] = df['distance']*0
        df['Vitesse_all'] = df['distance']*0
        deplacement = 0
        temps_totale = 0
        
        for i in range(0,df['Vitesse'].shape[0]):
            if i== df['Vitesse'].shape[0]-1:
                #vitesse finale vaut la vitesse précédente
                df['Vitesse'].iat[i] = df['Vitesse'][i-1]
                df['Vitesse_all'].iat[i] = df['Vitesse_all'][i-1]
            else :
                df['Vitesse'].iat[i] = abs((df['distance'][i+1]-df['distance'][i])/(df['Temps'][i+1]-df['Temps'][i]))
                df['Vitesse_all'].iat[i] = (df['distance'][i+1]-df['distance'][i])/(df['Temps'][i+1]-df['Temps'][i])
        
        #plot vitesse temps
        df.plot(x = 'Temps', y = 'Vitesse')
        plt.title("$u_{diode}$ instantanée en fonction du temps("+ intensite_E +'V.$cm^{-1}$' + ")")
        plt.xlabel("Temps (s) ")
        plt.ylabel("$u_{diode}$  (cm/s)")
        
        
        final_file_vitesse = os.path.join(save_file,'Vitesse_temps', 'Results_'+ Volt + 'V' + Number )
        plt.savefig(final_file_vitesse + "_Vitesse_temps")
        plt.show()
        
        #ecriture dans le doc vitesse moyenne    
        with open(r'C:\Users\nguye\Desktop\Master 2\Mémoire\Resultats\Vitesse moyenne.csv', 'a') as f:
            
            writer = csv.writer(f)    
            writer.writerow((Volt,df['Vitesse'].mean(),df['Vitesse'].std()))
        
        #clean data
        vitesse_mean = df['Vitesse'].mean()
        vitesse_std = df['Vitesse'].std()
        
        for i in range(0,df['Vitesse'].shape[0]):
            if df['Vitesse'][i] > vitesse_mean + 2*vitesse_std:
                df['Vitesse'].iat[i] = np.nan
        
        #plot 
        df.plot(x = 'distance', y = 'Vitesse', kind = 'scatter')
        plt.axvline(x = df['distance'].mean(), color = 'red', label = '<$\Delta$>')
        plt.title("$u_{diode}$ instantanée en fonction de $\Delta$("+ intensite_E +'V.$cm^{-1}$' + ")")
        plt.xlabel("$\Delta$ (cm) ")
        plt.ylabel("$u_{diode}$  (cm/s)")
        plt.legend(bbox_to_anchor = (1.0, 1), loc = 'upper right')
        
        final_file_vitesse = os.path.join(save_file,'Vitesse', 'Results_'+ Volt + 'V' + Number )
        plt.savefig(final_file_vitesse + "_Vitesse")
        plt.show()
        
        #acceleration
        df['Acceleration'] = df['distance']*0
        vitesse = 0
        temps_totale_acc = 0
        
        for i in range(0,df['Acceleration'].shape[0]):
            if i== df['Acceleration'].shape[0]-1:
                #vitesse finale vaut la vitesse précédente
                df['Acceleration'].iat[i] = df['Acceleration'][i-1]
            else :
                df['Acceleration'].iat[i] = (df['Vitesse_all'][i+1]-df['Vitesse_all'][i])/(df['Temps'][i+1]-df['Temps'][i])
        
        #plot en fonction du temps
        df.plot(x = 'Temps', y = 'Acceleration')
        plt.title("$a_{diode}$ instantanée en fonction du temps("+ intensite_E +'V.$cm^{-1}$' + ")")
        plt.xlabel("Temps (s) ")
        plt.ylabel("$a_{diode}$  (cm/s)")
        
        final_file_vitesse = os.path.join(save_file,'Acceleration_temps', 'Results_'+ Volt + 'V' + Number )
        plt.savefig(final_file_vitesse + "_Acceleration_temps")
        plt.show()        
             
        #ecriture dans le doc acc moyenne    
        with open(r'C:\Users\nguye\Desktop\Master 2\Mémoire\Resultats\Acceleration moyenne.csv', 'a') as f:
            
            writer = csv.writer(f)    
            writer.writerow((Volt,df['Acceleration'].mean(),df['Acceleration'].std()))
        
        #clean data
        acc_mean = df['Acceleration'].mean()
        acc_std = df['Acceleration'].std()
        
        for i in range(0,df['Acceleration'].shape[0]):
            if (df['Acceleration'][i] > acc_mean + 2*acc_std or df['Acceleration'][i] < acc_mean - 2*acc_std):
                df['Acceleration'].iat[i] = np.nan
                
        #Recherche des delta lorsque a=0
        delta_a0 = 0
        nbr_delta = 0
        delta_moy_a0 = 0
        for i in range(0,df['Acceleration'].shape[0]):
            if (df['Acceleration'][i] < 0.02 and df['Acceleration'][i] > -0.02):
                delta_a0 += df['distance'][i]
                nbr_delta +=1
        
        delta_moy_a0 = delta_a0/nbr_delta
        
        #ecriture dans le doc delta moy a=0    
        with open(r'C:\Users\nguye\Desktop\Master 2\Mémoire\Resultats\Delta_moy_a0.csv', 'a') as f:
            
            writer = csv.writer(f)    
            writer.writerow((Volt,delta_moy_a0))
        
        #plot
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams["figure.dpi"] = 144
        df.plot(x = 'distance', y = 'Acceleration', kind = 'scatter')
        plt.axvline(x = df['distance'].mean(), color = 'red', label = '<$\Delta$>')
        plt.axvline(x = delta_moy_a0, color = 'green', label = '<$\Delta$> pour a~0')
        #plt.title("$a_{diode}$ instantanée en fonction de $\Delta$("+ intensite_E +'V.$cm^{-1}$' + ")")
        plt.xlabel("$\Delta$ (cm) ")
        plt.ylabel("$a_{inst}$  (cm/s$^2$)")
        plt.legend(bbox_to_anchor = (1.0, 1), loc = 'upper right')
        
        final_file_vitesse = os.path.join(save_file,'Acceleration', 'Results_'+ Volt + 'V' + Number )
        plt.savefig(final_file_vitesse + "_Acceleration")
        plt.show()
        
#Fichier distance moyenne final
df_pos_moyenne = pd.read_csv(r'C:\Users\nguye\Desktop\Master 2\Mémoire\Resultats\Position moyenne.csv')

#Graphique de la distance moyenne + ecart type
i=0
x=[]
y = []
y_errorbar = []
for i in range(0, Volt_step):
    data_per_voltage = []
    j = 0
    k=0
    for j in range(0, df_pos_moyenne.shape[0]):
        if df_pos_moyenne['Volt'][j] == (i)*Volt_variation :
            data_per_voltage.append(df_pos_moyenne['Distance_moyenne'][j])
            k+=1
    
    x.append(round((i)*Volt_variation/3.4,0))
    y.append(sts.mean(data_per_voltage))
    y_errorbar.append(sts.stdev(data_per_voltage))


#plt.title("Distance moyenne <$\Delta$> en fonction de $E_{ext}$")
plt.xlabel("$E_{ext}$(V cm$^{-1}$)")
plt.ylabel("<$\Delta$> (cm)")
plt.errorbar(x,y,y_errorbar,capsize=4)

plt.savefig(r"C:\Users\nguye\Desktop\Master 2\Mémoire\Resultats\Position moyenne")
plt.show()

#Graphique de l'écart-type
i=0
x_e =[]
y_e = []
y_e_errorbar = []
for i in range(0, Volt_step):
    data_per_voltage = []
    j = 0
    k=0
    for j in range(0, df_pos_moyenne.shape[0]):
        if df_pos_moyenne['Volt'][j] == (i)*Volt_variation :
            data_per_voltage.append(df_pos_moyenne['Ecart_type_moyen'][j])
            k+=1
    
    x_e.append(round((i)*Volt_variation/3.4,0))
    y_e.append(sts.mean(data_per_voltage))
    y_e_errorbar.append(sts.stdev(data_per_voltage))


plt.title("Ecart-type moyen en fonction de $E_{ext}$")
plt.xlabel("$E_{ext}$(V)")
plt.ylabel("Ecart-type moyen (cm)")
plt.errorbar(x_e,y_e,y_e_errorbar,capsize=4)

plt.savefig(r"C:\Users\nguye\Desktop\Master 2\Mémoire\Resultats\Ecart-type moyen")
plt.show()



#Fichier vitesse moyenne final
df_vit_moyenne = pd.read_csv(r'C:\Users\nguye\Desktop\Master 2\Mémoire\Resultats\Vitesse moyenne.csv')

#Graphique de la vitesse moyenne + ecart type
plt.rcParams['figure.figsize'] = (7, 6)


i=0
x_vitesse=[]
y_vitesse = []
y_errorbar_vitesse = []
for i in range(0, Volt_step):
    data_per_voltage_vitesse = []
    j = 0
    k=0
    for j in range(0, df_vit_moyenne.shape[0]):
        if df_vit_moyenne['Volt'][j] == (i)*Volt_variation :
            data_per_voltage_vitesse.append(df_vit_moyenne['Vitesse_moyenne'][j])
            k+=1
    
    x_vitesse.append(round((i)*Volt_variation/3.4,0))
    y_vitesse.append(sts.mean(data_per_voltage_vitesse))
    y_errorbar_vitesse.append(sts.stdev(data_per_voltage_vitesse))

x_vitesse_m = np.array(x_vitesse) * 100
y_vitesse_m = np.array(y_vitesse) / 100
y_errorbar_vitesse_m = np.array(y_errorbar_vitesse) / 100

#Linear regression
reg = LinearRegression(fit_intercept=False)

eta = 10**(-3)
epsilon = 81 * 8.854 * 10**(-12)
zeta = 80*10**(-3)
E_0 = 0.67/(2.6 * 10**(-3))

x_fit = np.array(np.delete(x_vitesse_m, 0)) - E_0
x_fit = x_fit[:, np.newaxis]
y_fit = np.array(np.delete(y_vitesse_m, 0)) * eta /(epsilon * zeta)

#fit
fit_var = reg.fit(x_fit,y_fit)

plt.scatter(x_fit,y_fit,color = 'green')
plt.plot(x_fit,fit_var.predict(x_fit), color = 'black')
plt.show()

print("Le coefficient de détermination r^2 est :" , fit_var.score(x_fit,y_fit))
print("Beta vaut :" ,  fit_var.coef_)

coef_reel = float(fit_var.coef_ *epsilon*zeta/eta)*10000

fit_reel = coef_reel * (np.array(x_vitesse) - E_0/100)
#plot final

#Data
#plt.title("Vitesse moyenne en fonction de $E_{ext}$")
plt.xlabel("$E_{ext}$(V cm$^{-1}$)")
plt.ylabel("Vitesse moyenne (cm.$s^{-1}$)")
plt.errorbar(x_vitesse,y_vitesse,y_errorbar_vitesse,capsize=4, fmt='o', label = 'Données mesurées')
plt.ylim((-0.002,0.15))
#fit
plt.plot(x_vitesse,fit_reel, color = 'black', label = 'Courbe de tendance $u_{fit}$')

#plt.legend(bbox_to_anchor = (1.0, 0), loc = 'lower right')

plt.savefig(r"C:\Users\nguye\Desktop\Master 2\Mémoire\Resultats\Vitesse moyenne et fit")
plt.show()