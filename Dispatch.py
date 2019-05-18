#======================================================
# MAIN FUNCTIONS
#======================================================
import pandas as pd
import numpy as np
Sbase=100
def createMatrixX(lines,NB):
    X=np.zeros((NB,NB))
    for i in range(len(lines)):
        nodo_i=int(lines[i,0])-1
        nodo_j=int(lines[i,1])-1
        X[nodo_i,nodo_j]=lines[i,3]
        X[nodo_j,nodo_i]=lines[i,3]
    return X

def createMatrixB(X):
    B=X.copy()
    for i in range(len(B)):
        for j in range(len(B)):
            if i==j:
                B[i,i]=sum(1/k for k in X[i,:] if k!=0)
            else:
                if X[i,j]!=0:
                    B[i,j]=-1/X[i,j]
                else:
                    B[i,j]=0
    return B

#Calculate power injeted and consumed at each bus
def fillPower(Pg,Pd,NB):
    #Pinj=np.zeros((NB,NB))
    #Pcon=np.zeros((NB,NB))
    DeltaP=np.zeros(NB)
    for i in range(NB):
        DeltaP[i]=Pg[i]-Pd[i]
    return DeltaP

def fillPd(loads,NB):
    Pd=[0]*NB
    for i in range(NB):
        ii=int(loads[i,0])
        Pd[ii-1]=loads[i,1]/Sbase
    return Pd

    
def initCosts(generators):
    NG=generators.shape[0]
    Alpha = [0]*NG
    Beta = [0]*NG
    Gamma = [0]*NG
    for i in range(NG):
        Alpha[i]=generators[i,3]
        Beta[i]=generators[i,4]
        Gamma[i]=generators[i,5]
    return Alpha, Beta, Gamma
    
def initPmaxmin(generators):
    NG=generators.shape[0]
    Pmax=[0]*NG
    Pmin=[0]*NG
    for i in range(NG):
        ii=int(generators[i,0])
        Pmin[ii-1]=generators[i,6]/Sbase
        Pmax[ii-1]=generators[i,7]/Sbase
    return Pmin,Pmax

def cost_der(x):
    g=np.zeros_like(x)
    n=len(Beta)
    for i in range(n):
        g[i]=Beta[i]
    g[n:]=0
    return g

def cost_hess(x):
    h=np.zeros((len(x),len(x)))
    return h

def makeConstraint(NB,NL,NG,slack,file,B,X,Pd,Cap):
    dflines=pd.read_excel(file,'ParametrosLineas')
    lines=dflines.values
    binf=np.zeros(NB+NL)
    bsup=np.zeros(NB+NL)
    A=np.zeros((NB+NL,NG+NB))
    for i in range(NB):
        A[i,i]=1
        for j in range(NG,NG+NB):
            A[i,j]=B[i,j-NG]
    #Constraints of flows (angles)
    bsup[0:NB]=np.copy(Pd)
    binf[0:NB]=np.copy(Pd)
    k=0
    for line in lines:
        i=int(line[0])-1
        j=int(line[1])-1
        A[NB+k,NB+i]=1
        A[NB+k,NB+j]=-1
        bsup[NB+k]=Cap*X[i,j]
        binf[NB+k]=-Cap*X[i,j]
        k=k+1
    #Take out slack angle (because theta0 = 0)
    aux=[i for i in range(NG+NB) if i!=(NG+slack)]
    return A[:,aux],binf,bsup
    
def readsystem(file):
    dflines=pd.read_excel(file,'ParametrosLineas')
    lines=dflines.values
    dfgenerators=pd.read_excel(file,'ParametrosGeneradores')
    generators=dfgenerators.values
    dfloads=pd.read_excel(file,'ParametrosCargas')
    loads=dfloads.values
    dfbus=pd.read_excel(file,'ParametrosNodos')
    bus=dfbus.values
    #Number of busbar
    NB=bus[0,0]
    #Number og generators
    NG=generators.shape[0]
    #Number of lines
    NL=len(lines)
    #En pu
    X=createMatrixX(lines,NB)
    B=createMatrixB(X)
    #Pmax, Pmin
    Pmin,Pmax=initPmaxmin(generators)
    #Vector de potencia demandada
    Pd=fillPd(loads,NB)
    #Costos
    Alpha, Beta, Gamma=initCosts(generators)
    return NB,NL,NG,Pd, Alpha, Beta, Gamma,X,B,Pmin,Pmax



#======================================================
# INIT VARIABLES
#======================================================
import pandas as pd
import numpy as np
from __future__ import print_function
from ortools.linear_solver import pywraplp

#Cargar datos
file = 'Data.xlsx'
NB,NL,NG,Pd, Alpha, Beta, Gamma,X,B,Pmin,Pmax = readsystem(file)






#======================================================
# START OPTIMIZATION VARIABLES
#======================================================
from scipy.optimize import Bounds,LinearConstraint, minimize

#bounds = Bounds([Pmin[1],Pmin[2],-2*np.pi,-2*np.pi],[Pmax[1],Pmax[2],2*np.pi,2*np.pi])
#bounds = Bounds([Pmin[1],Pmin[2],-1000,-1000],[Pmax[1],Pmax[2],1000,1000])
bounds = Bounds([Pmin[0],Pmin[1],Pmin[2],-1,-1],[Pmax[0],Pmax[1],Pmax[2],1,1])


slack=0
Cap=0.1
#from Ax=b, A matrix
A,binf,bsup=makeConstraint(NB,NL,NG,slack,file,B,X,Pd,Cap)
linear_constraint = LinearConstraint(A,binf,bsup)         


Bp=B[1:3,1:3]
Binv=np.linalg.inv(Bp)

def fo(y):
    return np.dot(y[0:3],Beta)


y0 = np.array([0.3,0.2,0.1,0,0])
res = minimize(fo, y0, method='trust-constr',jac=cost_der,
               hess=cost_hess, constraints=linear_constraint,
               options={'verbose': 1,'maxiter':1024}, bounds=bounds)
               #options={'maxiter':10000,'verbose': 1},bounds=bounds)


#PRINT RESULTS
print('--------------------------')
print('Balance')
print('Generacion = ',sum(res.x[0:3]))
print('Demanda = ',sum(Pd))
theta0=0
theta1=res.x[3]
theta2=res.x[4]
P01=(theta0-theta1)/X[0,1]
P12=(theta1-theta2)/X[1,2]
#P02=(theta0-theta2)/X[0,2]
print('--------------------------')
print('Detalle Generacion')
print('PG1 = ',round(res.x[0],2))
print('PG2 = ',round(res.x[1],2))
print('PG3 = ',round(res.x[2],2))
print('--------------------------')
print('Flujos')
print('P01 = ',round(P01,2))
#print('P02 = ',round(P02,2))
print('P12 = ',round(P12,2))
print('--------------------------')
print('Angulos')
print('Theta0 = ',round(theta0,4))
print('Theta1 = ',round(theta1,4))
print('Theta2 = ',round(theta2,4))
print('--------------------------')
print('Costos del Sistema')
print('USD = ',round(np.dot(res.x[0:3],Beta),2))
