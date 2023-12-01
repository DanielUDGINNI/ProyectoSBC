    # -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 16:49:38 2023

@author: DanielVazquez
"""

#Librerias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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

#one hot encoder

encoder = OneHotEncoder(sparse=False)
y_reshaped = y.reshape(-1, 1)
y = encoder.fit_transform(y_reshaped)

#Inicilizar pesos de forma aleatoria, 7 clases y 16 caracteristicas mas el sesgo
w = np.random.uniform(0, 11, size=(7, 16))
alpha = 8
#print("Pesos Iniciales: \n", w)
#print(w[0, :])

#Estandarizacion
scaler = StandardScaler()

#train y test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

yp_train = np.zeros((len(y_train), 7))

def numerador(pesos, variables):
     z = 0
     for j in range(len(variables)):
        z = z + (variables[j] * pesos[j])
     return np.exp(z + pesos[0])

def denominador(pesos, variables):
     z = 0
     for k in range(0, 7):
        e = 0
        for h in range(0, len(variables)):
            e = e + (pesos[k, h] * variables[h])
        z = z + np.exp(e + pesos[k, 0])
     return z
 
    
def funError(pred, train):
     epsilon = 1e-15
     er = 0
     pred = np.clip(pred, epsilon, 1 - epsilon)
     for k in range(0, 7):
        er = er + (train[k] * (np.log(pred[k])))
        
     return -er

def train():
     for i in range(len(X_train)):
        for j in range(0, 7):
            n = numerador(w[j, :],X_train[i, :])
            d = denominador(w, X_train[i, :])
            #softmax
            yp_train[i, j] = (n/d)

sz = 16
bi = np.zeros((7, 16)) 
def gradiente():
     for m in range(len(X_train)):
        
        for n in range(7):
            gr = y_train[m, n] - yp_train[m, n]
            for o in range(16):
                bi[n, o] = bi[n, o] + X_train[m, o] * gr
    #final          
     for q in range(7):
            for s in range(16):
                bi[q, s] = -1/(len(X_train)) * bi[q, s]

def actualizar():
     for q in range(7):
            for s in range(16):
                w[q, s] = w[q, s] - (alpha * bi[q, s])
                
yp_test = np.zeros((len(y_test), 7))
def test():
     for i in range(len(X_test)):
        for j in range(0, 7):
            n = numerador(w[j, :],X_test[i, :])
            d = denominador(w, X_test[i, :])
            #softmax
            yp_test[i, j] = (n/d)


def accuracy():
     trus = 0
     for m in range(len(yp_test)):
        for n in range(7):
            if(yp_test[m,n] == 1 and y_test[m,n] == 1):
                trus = trus + 1
     return trus/len(yp_test)

def precision():
     tp = 0
     fp = 0
     for m in range(len(yp_test)):
        for n in range(7):
            if(yp_test[m,n] == 1 and y_test[m,n] == 1):
                tp = tp + 1
            if(yp_test[m,n] == 1 and y_test[m,n] == 0): 
                fp = fp + 1
     return tp/(tp + fp)

def sensitivity():
     tp = 0
     fn = 0
     for m in range(len(yp_test)):
        for n in range(7):
            if(yp_test[m,n] == 1 and y_test[m,n] == 1):
                tp = tp + 1
            if(yp_test[m,n] == 0 and y_test[m,n] == 1): 
                fn = fn + 1
     return tp/(tp + fn)
    
def f1_score(p,s):
     res = (p * s)/(p + s)
     return 2 * res

def matriz_confusion(pred, etiq):
     tp = 0
     fp = 0 
     tn = 0 
     fn = 0 
    
     for r in range(0, len(pred)):
        for q in range(6):
            if(etiq[r, q] == 1 and pred[r, q] == 1):
                tp = tp + 1  #Verdaderos positivos
            if(etiq[r, q] == 0 and pred[r, q] == 0):
                tn = tn + 1  #Verdaderos negativos
            if(etiq[r, q] == 0 and pred[r, q] == 1):
                fp = fp + 1  #Falsos Positivos
            if(etiq[r, q] == 1 and pred[r, q] == 0): 
                fn = fn + 1  #Falsos negativos
                   
     print("TP: ", tp)
     print("FP: ", fp)
     print("TN: ", tn)
     print("FN: ", fn)     
            
iterations = np.zeros(500)
arror = np.zeros(500)
iter = 0
while iter <  500:
     train()
     #print(yp_train)
     error = 0
     for p in range(len(y_train)):
        error = error + funError(yp_train[p, :], y_train[p, :])
       
     entropia = error/(len(y_train))
     arror[iter] = error
     iterations[iter] = iter
     #print("Error: ",entropia)
     gradiente()
     actualizar()
     iter = iter + 1

#print(w[0, :])

test()
yp_test = np.round(yp_test)
yp_test = np.array(yp_test, dtype = int)

y_test = np.round(y_test)
y_test = np.array(y_test, dtype = int)
#print("Pesos Finales: \n", w)
#print(y_test)
#print(yp_test)
ac = accuracy()
pr = precision()
sn = sensitivity()
fs = f1_score(pr, sn)

print("Accuracy: ", ac * 100)
print("Precision: ", pr * 100)
print("Sensitivity: ", sn * 100)
print("F1 Score: ", fs * 100)
matriz_confusion(yp_test, y_test)

#for f in range(len(yp_test)):
   # print(yp_test[f, :], y_test[f, :])



plt.plot(iterations, arror, label='Perdida')

plt.xlabel('iter')
plt.ylabel('Error')
plt.title('Logistic Regresion - Zoo')

plt.legend()

plt.show()

