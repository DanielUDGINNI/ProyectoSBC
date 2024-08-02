# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 14:22:57 2023

@author: DanielVazquez
"""

#Librerias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#Carga del dataset
data =  pd.read_csv('heart.csv')

headers = data.columns
df = data.to_numpy()
#Definir X
x = df[:, :-1]
#Definir Y
y = df[:, -1]
#Inicilizar pesos de forma aleatoria (13 + W0)
w = np.random.uniform(0, 10, size=14)
alpha = 8
#print("Pesos Iniciales: \n",w)

#-------------------------------------------------------------------------------#
#Estandarizacion
scaler = StandardScaler()

#train y test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
yp_train = np.zeros(len(y_train))

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

def calc(a,b):
    z = 0
    for j in range(len(a)):
        z = z + (a[j] * b[j+1])
    return z + b[0]

def sigmoide(c):
    return 1 / (1 + np.exp(-c))

def cross_entropy(yi,yj):
    m = len(yi)
    epsilon = 1e-15
    b = 0
    for i in range(m):
        b = b + (yi[i] * np.log(yj[i] + epsilon) + (1 - yi[i]) * np.log(1 - yj[i] + epsilon)) 
    z = (-1/m) * b
    return z

def train():
    for i in range(0,len(X_train)):
        yp_train[i] = sigmoide(calc(X_train[i, :], w))
        
#et = etiquetas, pr = prediccion
def derivadas(et, pr):
    #parametros
    b = np.zeros(len(w))
    for k in range(0, len(et)):
        b[0] = b[0] + (et[k] - pr[k])
        b[1] = b[1] + ((X_train[k, 0] *(et[k] - pr[k])))
        b[2] = b[2] + ((X_train[k, 1] *(et[k] - pr[k])))
        b[3] = b[3] + ((X_train[k, 2] *(et[k] - pr[k])))
        b[4] = b[4] + ((X_train[k, 3] *(et[k] - pr[k])))
        b[5] = b[5] + ((X_train[k, 4] *(et[k] - pr[k])))
        b[6] = b[6] + ((X_train[k, 5] *(et[k] - pr[k])))
        b[7] = b[7] + ((X_train[k, 6] *(et[k] - pr[k])))
        b[8] = b[8] + ((X_train[k, 7] *(et[k] - pr[k])))
        b[9] = b[9] + ((X_train[k, 8] *(et[k] - pr[k])))
        b[10] = b[10] + ((X_train[k, 9] *(et[k] - pr[k])))
        b[11] = b[11] + ((X_train[k, 10] *(et[k] - pr[k])))
        b[12] = b[12] + ((X_train[k, 12] *(et[k] - pr[k])))
        
        
    b[0] = (-1/len(et)) * b[0]
    b[1] = (-1/len(et)) * b[1]
    b[2] = (-1/len(et)) * b[2]
    b[3] = (-1/len(et)) * b[3]
    b[4] = (-1/len(et)) * b[4]
    b[5] = (-1/len(et)) * b[5]
    b[6] = (-1/len(et)) * b[6]
    b[7] = (-1/len(et)) * b[7]
    b[8] = (-1/len(et)) * b[8]
    b[9] = (-1/len(et)) * b[9]
    b[10] = (-1/len(et)) * b[10]
    b[11] = (-1/len(et)) * b[11]
    b[12] = (-1/len(et)) * b[12]
    return b        

#Actualizacion de pesos    
def actualizacion(pesos):
    for j in range(0, len(pesos)):
        w[j] = w[j] - (alpha * pesos[j])
        
        
yp_test = np.zeros(len(y_test))
def test():
    #print(w)
    for i in range(0, len(X_test)):
        yp_test[i] = sigmoide(calc(X_test[i, :], w))
        #print(yp_test[i], y_test[i])

#Metricas de evaluacion----------------------------------
def accuracy(pred, etiq):
    correctos = 0
    for r in range(0, len(pred)):
        if(pred[r] == etiq[r]):
            correctos = correctos + 1
    return correctos/len(pred)

def precision(pred, etiq):
    tp = 0
    fp = 0
    for r in range(0, len(pred)):
        if(etiq[r] == 1 and pred[r] == 1):
            tp = tp + 1
        if(etiq[r] == 0 and pred[r] == 1):
            fp = fp + 1
    return tp/(tp + fp)

def sensitivity(pred, etiq):
    tp = 0
    fn = 0
    for r in range(0, len(pred)):
        if(etiq[r] == 1 and pred[r] == 1):
            tp = tp + 1
        if(etiq[r] == 1 and pred[r] == 0):
            fn = fn + 1
            
    return tp/(tp + fn)

def f1_score(p, s):
    res = (p * s)/(p + s)
    res = res * 2
    return res
    
def matriz_confusion(pred, etiq):
    tp = 0
    fp = 0 
    tn = 0 
    fn = 0 
    
    for r in range(0, len(pred)):
        if(etiq[r] == 1 and pred[r] == 1):
            tp = tp + 1  #Verdaderos positivos
        if(etiq[r] == 0 and pred[r] == 0):
            tn = tn + 1  #Verdaderos negativos
        if(etiq[r] == 0 and pred[r] == 1):
            fp = fp + 1  #Falsos Positivos
        if(etiq[r] == 1 and pred[r] == 0):
            fn = fn + 1  #Falsos negativos
    print("TP: ", tp)
    print("FP: ", fp)
    print("TN: ", tn)
    print("FN: ", fn)

n = 10
iterations = np.zeros(n)
opc = 0
arror = np.zeros(n)
while opc < n:
    train()
    error = cross_entropy(y_train, yp_train)
    arror[opc] = error
    iterations[opc] = opc
    print("Error entrenamiento: ", error)
    bi = derivadas(y_train, yp_train)
    #print(bi)
    actualizacion(bi)
    #print(w)
    opc = opc + 1
        
        

test()
yp_test = np.round(yp_test)
yp_test = np.array(yp_test, dtype = int)
#print("Pesos finales: \n",w)
#print(yp_test)
ac = accuracy(yp_test, y_test)
pr = precision(yp_test, y_test)
sn = sensitivity(yp_test, y_test)
fs = f1_score(pr, sn)

print("Accuracy: ",ac * 100)
print("Precision: ",pr * 100)
print("Sensitivity: ",sn * 100)
print("F1 Score: ",fs * 100)

matriz_confusion(yp_test, y_test)

plt.plot(iterations, arror, label='Perdida')

plt.xlabel('iter')
plt.ylabel('Error')
plt.title('Logistic Regresion - Heart')

plt.legend()

plt.show()


        
        
        
        
        
        
        
        
        
