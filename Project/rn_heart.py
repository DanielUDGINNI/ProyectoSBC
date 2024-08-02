# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 17:56:08 2023

@author: DanielVazquez
"""

#Librerias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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

alpha = 5
#print("Pesos Iniciales: ",w)
#print("Pesos Iniciales: \n",w)

#Estandarizacion
scaler = StandardScaler()
#train y test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
yp_train = np.zeros(len(y_train))

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

ncapas1 = 8
#ncapas2 = 4
capa1 = np.zeros(ncapas1)
#capa2 = np.zeros(ncapas2)
#Inicilizar pesos de forma aleatoria (8 + W0)
w1 = np.random.uniform(0, 10, size=(ncapas1,X_train.shape[1] + 1))
#w2 = np.random.uniform(0, 10, size=(ncapas2,len(capa1) + 1))
w3 = np.random.uniform(0, 10, size=(ncapas1 + 1))
w3[0] = 1
w1[:,0] = 1
error = 0
#print(w1)
def sigmoide(c):
    return 1 / (1 + np.exp(-c))

def calculo(x1, w):
    n = np.zeros(ncapas1)
    for j in range(ncapas1):
        for k in range(len(x1)):
            n[j] = n[j] + x1[k] * w1[j, k + 1]
            
        n[j] = n[j] + w[j, 0]
    return n
    
def salida(c):
    for i in range(len(c)):
        c[i] = sigmoide(c[i])
    return c

def salir(x1,w):
    n = 0
    for i in range(len(x1)):
        n = n + (x1[i] * w[i + 1] )
    n = n + w[0]
    return n
    
def cross_entropy(yi,yj):
    m = len(yi)
    epsilon = 1e-15
    b = 0
    for i in range(m):
        b = b + (yi[i] * np.log(yj[i] + epsilon) + (1 - yi[i]) * np.log(1 - yj[i] + epsilon)) 
    z = (-1/m) * b
    return z

def mse(real, pred):
    return((real - pred)**2)

def mseGeneral(real, pred):
    s = 0
    for i in range(len(real)):
        s = s + (real[i] - pred[i])**2
    return (1/len(real)) * s


def actualizacion1(capa, y, yreal, x):
    for i in range(len(capa)):
        d = (y - yreal) #derivadas error cuadratico medio
        #d = - ((yreal/y) + ((1 - yreal)/(1 - y))) #entrop
        d = d * (y * (1 - y)) #sigm
        #error de la ultima capa
        a = actualizacion2(d, w3[i+1], capa[i])
        for j in range(13):
            w1[i, j + 1] = w1[i, j + 1] - (alpha * (a * x[j]))

        w1[i,0] = w1[i,0] - (alpha * a) 
        d = d * capa[i] #salida de capa
        w3[i+1] = w3[i+1] - (alpha * d) #actualizacion
    w3[0] = w3[0] - (alpha * ((y - yreal)*(y * (1 - y))))
    #return aux
    
def actualizacion2(e, wa, capa):
    res = (capa * (1 - capa))
    res = res * wa
    return res
    
def train():
    #error = 0
    for i in range(len(X_train)): #len(X_train)
        capa1 = calculo(X_train[i, :],w1)
        capa1s = salida(capa1)
        y = salir(capa1s, w3)
        yp_train[i] = sigmoide(y)
        #print(yp_train[i], y_train[i])
        actualizacion1(capa1s, yp_train[i], y_train[i], X_train[i, :])
        #error = error + mse(y_train[i],yp_train[i])
        #print(y)
    #print(1/(len(y_train)) * error)
    
    
yp_test = np.zeros(len(y_test))
def test():    
    for i in range(0, len(X_test)):
        capa1 = calculo(X_test[i, :],w1)
        capa1s = salida(capa1)
        y = salir(capa1s, w3)
        yp_test[i] = sigmoide(y)
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

n = 5
iterations = np.zeros(n)
arror = np.zeros(n)
#print(calculo(X_train[0, :],w1))-----------------
for i in range(n):
    train()
    er = mseGeneral(y_train, yp_train)
    arror[i] = er
    iterations[i] = i
    #print(er)

#print(yp_train)
test()
yp_test = np.round(yp_test)
yp_test = np.array(yp_test, dtype = int)

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
plt.title('Neural Network - Heart')

plt.legend()

plt.show()


