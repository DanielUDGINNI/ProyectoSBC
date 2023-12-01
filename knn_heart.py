# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 13:18:57 2023

@author: DanielVazquez
"""

#Librerias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Carga del dataset
data =  pd.read_csv('heart.csv')

headers = data.columns
df = data.to_numpy()
#Definir X
x = df[:, :-1]
#Definir Y
y = df[:, -1]
#print("Pesos Iniciales: ",w)

#-------------------------------------------------------------------------------#
#Estandarizacion
scaler = StandardScaler()

#train y test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
yp_test = np.zeros(len(y_test))

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

k = 3

def etiquetar(yi, pos):
    orden = np.argsort(yi[:, 0])
    yiord = yi[orden]
    v1 = 0
    v0 = 0
    
    for q in range(k):
        if(yiord[k, 1] == 1):
            v1 = v1 + 1
        if(yiord[k, 1] == 0):
            v0 = v0 + 1
    
    if(v1 > v0):
       yp_test[pos] = 1
    if(v0 > v1):
       yp_test[pos] = 0
    


def eucl(x1, x2, pos):
    d = np.zeros((len(x2), 2))
    
    for j in range(len(x2)):    
        r = 0
        for k in range(len(x1)):
            r = r + ((x1[k] - x2[j, k])**2)
        d[j, 0] = np.sqrt(r)
        d[j, 1] = y_train[j] 
        
    #print(d)
    etiquetar(d, pos)


def test():
    for i in range(len(X_test)):
        eucl(X_test[i, :], X_train, i)


def accuracy(pred, etiq):
    correctos = 0
    #print(len(pred))
    for r in range(0, len(pred)):
        if(pred[r] == etiq[r]):
            correctos = correctos + 1
    #print(correctos)
    return correctos/len(pred)

def precision(pred, etiq):
    tp = 0
    fp = 0
    for r in range(0, len(pred)):
        if(etiq[r] == 1 and pred[r] == 1):
            tp = tp + 1
        if(etiq[r] == 0 and pred[r] == 1):
            fp = fp + 1
    #print(tp)
    #print(fp)
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


test()

yp_test = np.round(yp_test)
yp_test = np.array(yp_test, dtype = int)
y_test = np.round(y_test)
y_test = np.array(y_test, dtype = int)

for a in range(len(yp_test)):
  print(int(yp_test[a]), y_test[a])

ac = accuracy(yp_test, y_test)
pr = precision(yp_test, y_test)
sn = sensitivity(yp_test, y_test)
fs = f1_score(pr, sn)

print("Accuracy: ",ac * 100)
print("Precision: ",pr * 100)
print("Sensitivity: ",sn * 100)
print("F1 Score: ",fs * 100)


matriz_confusion(yp_test, y_test)
