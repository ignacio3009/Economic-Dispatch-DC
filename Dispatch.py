import pandas as pd
import numpy as np

#=========================================
# LOAD DATA 
#=========================================


dfLineas=pd.read_excel('Datos.xlsx','ParametrosLineas')
lineas=dfLineas.values
dfGeneradores=pd.read_excel('Datos.xlsx','ParametrosGeneradores')
generadores=dfGeneradores.values
dfCargas=pd.read_excel('Datos.xlsx','ParametrosCargas')
cargas=dfCargas.values
dfNodos=pd.read_excel('Datos.xlsx','ParametrosNodos')
nodos=dfNodos.values


from __future__ import print_function
from ortools.linear_solver import pywraplp

#Numero barras
NB=nodos[0,0]
#Numero generadores
NG=generadores.shape[0]

#En pu
X=createMatrixX(lineas,NB)
B=createMatrixB(X)


#Pmax, Pmin
Pmin,Pmax=_initPmaxmin(generadores)


#Flujos por lineas
Pf=np.zeros((NB,NB))

#Vector de potencia demandada
Pd=_fillPd(NB)

#Vector de potencia generada
Pg=np.zeros(NB)

#Costos
Alpha, Beta, Gamma=_initCosts(generadores)


#=========================================
# FUNCTIONS 
#=========================================

Sbase=100
def createMatrixX(lineas,NB):
    X=np.zeros((NB,NB))
    for i in range(len(lineas)):
        nodo_i=int(lineas[i,0])-1
        nodo_j=int(lineas[i,1])-1
        X[nodo_i,nodo_j]=lineas[i,3]
        X[nodo_j,nodo_i]=lineas[i,3]
    return X

def createMatrixB(X):
    B=X.copy()
    for i in range(len(B)):
        for j in range(len(B)):
            if i==j:
                B[i,i]=sum(1/k for k in X[i,:] if k!=0)
            else:
                B[i,j]=-1/X[i,j]
    return B

#Calculate power injeted and consumed at each bus
def fillPower(Pg,Pd,NB):
    #Pinj=np.zeros((NB,NB))
    #Pcon=np.zeros((NB,NB))
    DeltaP=np.zeros(NB)
    for i in range(NB):
        DeltaP[i]=Pg[i]-Pd[i]
    return DeltaP

def _fillPd(NB):
    Pd=[0]*NB
    for i in range(NB):
        ii=int(cargas[i,0])
        Pd[ii-1]=cargas[i,1]/Sbase
    return Pd

    
def _initCosts(generadores):
    NG=generadores.shape[0]
    Alpha = [0]*NG
    Beta = [0]*NG
    Gamma = [0]*NG
    for i in range(NG):
        Alpha[i]=generadores[i,3]
        Beta[i]=generadores[i,4]
        Gamma[i]=generadores[i,5]
    return Alpha, Beta, Gamma
    
def _initPmaxmin(generadores):
    NG=generadores.shape[0]
    Pmax=[0]*NG
    Pmin=[0]*NG
    for i in range(NG):
        ii=int(generadores[i,0])
        Pmin[ii-1]=generadores[i,6]/Sbase
        Pmax[ii-1]=generadores[i,7]/Sbase
    return Pmin,Pmax



#=========================================
# STARTING ALGORITHM 
#=========================================

from scipy.optimize import Bounds,LinearConstraint, minimize

#bounds = Bounds([Pmin[1],Pmin[2],-2*np.pi,-2*np.pi],[Pmax[1],Pmax[2],2*np.pi,2*np.pi])
#bounds = Bounds([Pmin[1],Pmin[2],-1000,-1000],[Pmax[1],Pmax[2],1000,1000])
bounds = Bounds([Pmin[0],Pmin[1],Pmin[2],-1,-1],[Pmax[0],Pmax[1],Pmax[2],1,1])


#linear_constraint=LinearConstraint([1,1,1,0,0],[sum(Pd)],[sum(Pd)])

Alpha, Beta, Gamma=_initCosts(generadores)

Bp=B[1:3,1:3]
Binv=np.linalg.inv(Bp)

def fo(y):
    #Pgen- Pcons
    DP = (y[0:3]-Pd)[1:]
    y[3:]=np.dot(Binv,DP)
    
    Angles=np.copy(y[3:])
    Angles=np.insert(Angles,0,0,axis=0)
    
    y[0:3]=np.dot(B,Angles)+Pd
    
    #return Pgen[0]*Beta[0] + Pgen[1]*Beta[1]  + Pgen[2]*Beta[2]
    return np.dot(y[0:3],Beta)

Cap=0.1
#from Ax=b, A matrix
linear_constraint = LinearConstraint([[1,0,0,-1/X[0,1],-1/X[0,2]],
                                      [0,1,0,1/X[1,0]+1/X[1,2],-1/X[1,2]],
                                      [0,0,1,-1/X[2,1],1/X[2,0]+1/X[2,1]], 
                                      [0,0,0,-1,0],
                                      [0,0,0,0,-1],
                                      [0,0,0,1,-1]],
                                     [Pd[0],Pd[1],Pd[2], -Cap*X[0,1], -Cap*X[0,2], -Cap*X[1,2]],        #Lower bounds constraints (bmin)
                                     [Pd[0],Pd[1],Pd[2], Cap*X[0,1], Cap*X[0,2], Cap*X[1,2]])           #Upper bounds constraints (bmax)

y0 = np.array([10,10,10,0,0])
res = minimize(fo, y0, method='trust-constr',jac=cost_der, hess=cost_hess, constraints=linear_constraint,
               #options={'maxiter':10000,'verbose': 1},bounds=bounds)
               options={'verbose': 1,'xtol':1e-8,'disp':True}, bounds=bounds)


#PRINT RESULTS
print('--------------------------')
print('Balance')
print('Generacion = ',sum(res.x[0:3]))
print('Demanda = ',sum(Pd))
theta0=0
theta1=res.x[3]
theta2=res.x[4]
P01=(theta0-theta1)/X[0,1]
P02=(theta0-theta2)/X[0,2]
P12=(theta1-theta2)/X[1,2]
print('--------------------------')
print('Detalle Generacion')
print('PG1 = ',round(res.x[0],2))
print('PG2 = ',round(res.x[1],2))
print('PG3 = ',round(res.x[2],2))
print('--------------------------')
print('Flujos')
print('P01 = ',round(P01,2))
print('P02 = ',round(P02,2))
print('P12 = ',round(P12,2))
print('--------------------------')
print('Angulos')
print('Theta0 = ',round(theta0,4))
print('Theta1 = ',round(theta1,4))
print('Theta2 = ',round(theta2,4))
print('--------------------------')
print('Costos del Sistema')
print('USD = ',round(np.dot(res.x[0:3],Beta),2))
