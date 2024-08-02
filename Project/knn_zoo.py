# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 13:45:01 2023

@author: DanielVazquez
"""

#Librerias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

#Carga del dataset
data =  pd.read_csv('zoo.csv')

headers = data.columns
df = data.to_numpy()
#Definir X
x = df[:, :-1]
#Eliminar primera columna
x = x[:, 1:]
#Definir Y
y = df[:, -1]

#Estandarizacion
scaler = StandardScaler()

#train y test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
yp_test = np.zeros(len(y_test))

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

k = 3

def etiquetar(yi, pos):
    orden = np.argsort(yi[:, 0])
    yiord = yi[orden]
    v = np.zeros(7)
    #print(yiord)
    for q in range(k):
        if(yiord[k, 1] == 1):
            v[0] = v[0] + 1
        if(yiord[k, 1] == 2):
            v[1] = v[1] + 1
        if(yiord[k, 1] == 3):
            v[2] = v[2] + 1
        if(yiord[k, 1] == 4):
            v[3] = v[3] + 1
        if(yiord[k, 1] == 5):
            v[4] = v[4] + 1
        if(yiord[k, 1] == 6):
            v[5] = v[5] + 1
        if(yiord[k, 1] == 7):
            v[6] = v[6] + 1   
    
    indice = np.argmax(v)
   # print(v)
    yp_test[pos] = indice + 1

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
    for r in range(0, len(pred)):
        if(pred[r] == etiq[r]):
            correctos = correctos + 1
    return correctos/len(pred)

def precision(pred, etiq):
    tp = np.zeros(7)
    fp = np.zeros(7)
    prs = np.zeros(7)
    
    for j in range(7):
        
        for r in range(0, len(pred)):
            if(etiq[r] == j + 1 and pred[r] == j + 1):
                tp[j] = tp[j] + 1
                
            if(etiq[r] == j + 1 and pred[r] != j + 1):
                fp[j] = fp[j] + 1
                
    for i in range(7):
        if(tp[i] != 0):
            prs[i] = (tp[i]/(tp[i] + fp[i])) * 100
        else:
            prs[i] = 0    
    return prs

def sensitivity(pred, etiq):
    #Tp / (tp + fn)
    tp = np.zeros(7)
    fn = np.zeros(7)
    prs = np.zeros(7)
    
    for j in range(7):
        
        for r in range(0, len(pred)):
            if(etiq[r] == j + 1 and pred[r] == j + 1):
                tp[j] = tp[j] + 1
                
            if(etiq[r] != j + 1 and pred[r] == j + 1):
                fn[j] = fn[j] + 1
                
    for i in range(7):
        if(tp[i] != 0):
            prs[i] = (tp[i]/(tp[i] + fn[i])) * 100
        else:
            prs[i] = 0    
    return prs

def f1_score(p, s):
    
    prs = np.zeros(7)
    for j in range(len(prs)):
        if(p[j] != 0):
            prs[j] = (p[j] * s[j])/(p[j] + s[j])
            prs[j] = prs[j] * 2
        else:
            prs[j] = 0
            
    return prs

def matriz_confusion(pred, etiq):
    tp = np.zeros(7)
    fn = np.zeros(7)
    tn = np.zeros(7)
    fp = np.zeros(7)
    
    for j in range(7):
        for r in range(0, len(pred)):
            if(etiq[r] == j + 1 and pred[r] == j + 1):
                tp[j] = tp[j] + 1
            if(etiq[r] == j + 1 and pred[r] != j + 1):
                fn[j] = fn[j] + 1
            if(etiq[r] != j + 1 and pred[r] != j + 1):
                tn[j] = tn[j] + 1
            if(etiq[r] != j + 1 and pred[r] == j + 1):
                fp[j] = fp[j] + 1
        print("Verdaderos Positivos Categoria: ", j + 1, ": ",tp[j])
        print("Falsos Negativos Categoria: ", j + 1, ": ",fn[j])
        print("Verdaderos Negativos Categoria: ", j + 1, ": ",tn[j])
        print("Falsos Positivos Categoria: ", j + 1, ": ",fp[j])
        
    
test()
#print(yp_test)
#for a in range(len(yp_test)):
  #print(int(yp_test[a]), y_test[a])

ac = accuracy(yp_test, y_test)
pr = precision(yp_test, y_test)
sn = sensitivity(yp_test, y_test)
fs = f1_score(pr, sn)

print("Accuracy: ",ac * 100)
for elm in range(len(pr)):
    print("Categoria: ", elm + 1)
    print("Precision: ",pr[elm])
    print("Sensitivity: ",sn[elm])    
    print("F1 Score Cateogira: ", fs[elm])       
        
        
matriz_confusion(yp_test, y_test)   
        
        
        
        
        
        
        
        
        
        
        
        
