# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 21:35:59 2024

@author: Souvignon
"""

import sys
import math
from math import sqrt
import numpy as np  
from scipy import special
from numpy import linalg as LA
import scipy as sp 
import time
import datetime 
from ncon import ncon
from matplotlib import pyplot as plt
D = 15  # We use D = 20 in paper. 
Niters = 30  # 2^30 x 2^30 square lattice 
h = 0.0 # Magentic field

L = np.zeros([D])
A = np.zeros([D])
startTime = time.time()
print ("STARTED: " , datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) 

def tensorsvd(input,left,right,D):
    '''Reshape an input tensor into a rectangular matrix with first index corresponding
    to left set of indices and second index corresponding to right set of indices. Do SVD
    and then reshape U and V to tensors [left,D] x [D,right]  
    '''
    T = input 
    left_index_list = []
    for i in range(len(left)):
        left_index_list.append(T.shape[i])
    xsize = np.prod(left_index_list) 
    right_index_list = []
    for i in range(len(left),len(left)+len(right)):
        right_index_list.append(T.shape[i])
    ysize = np.prod(right_index_list)
    T = np.reshape(T,(xsize,ysize))
    
    U, s, V = np.linalg.svd(T,full_matrices = False)
    if D < len(s):  # Truncate if length exceeds input D 
        s = np.diag(s[:D])
        U = U[:,:D]
        V = V[:D,:]
    else:
        D = len(s)
        s = np.diag(s)

    U = np.reshape(U,left_index_list+[D])
    V = np.reshape(V,[D]+right_index_list)   
    return U, s, V

 
def Z2d_Ising(beta, h):
    betah = beta*h 
    tau = np.exp(0.250000*betah) 
    a = np.sqrt(np.cosh(beta)) 
    b = np.sqrt(np.sinh(beta)) 
    W = np.array([[a*tau,b*tau],[(a/tau),-(b/tau)]]) 
    out = np.einsum("ia, ib, ic, id  -> abcd", W, W, W, W) 
    return out 


def Z2d_Ising_S(beta, h):
# Alternate definition 
    a = np.sqrt(np.exp(2*beta) + np.sqrt(np.exp(4*beta) - 1.0))  
    W = np.array([[a,(1.0/a)],[(1.0/a), a]]) 
    W /= np.sqrt(2.0*np.exp(beta))  
    out = np.einsum("ia, ib, ic, id  -> abcd", W, W, W, W) 
    return out 


def CG_step(matrix, in2):
    T = matrix  
    TI = in2 
    AAdag = ncon([T,T,T,T],[[-2,1,2,5],[-1,5,3,4],[-4,1,2,6],[-3,6,3,4]])
    U, s, V = tensorsvd(AAdag,[0,1],[2,3],D) 
    A = ncon([U,T,T,U],[[1,2,-1],[2,-2,4,3],[1,3,5,-4],[5,4,-3]])

    AAAAdag = ncon([A,A,A,A],[[1,-1,2,3],[2,-2,4,5],[1,-3,6,3],[6,-4,4,5]])
    U, s, V = tensorsvd(AAAAdag,[0,1],[2,3],D)  
    AA = ncon([U,A,A,U],[[1,2,-2],[-1,1,3,4],[3,2,-3,5],[4,5,-4]])

    maxAA = np.max(AA)
    AA = AA/maxAA # Normalize by the largest element of the tensor
        
    return AA, maxAA


if __name__ == "__main__":

    beta = np.arange(0.3, 0.6, 0.01).tolist()
    Nsteps = int(np.shape(beta)[0])
    f = np.zeros(Nsteps)
    internalE = np.zeros(Nsteps)

    for p in range (0, Nsteps):

      T = Z2d_Ising(beta[p], h)
      norm = LA.norm(T)
      T /= norm 
      Tim = T 
      Z = ncon([T,T,T,T],[[7,5,3,1],[3,6,7,2],[8,1,4,5],[4,2,8,6]])
      C = 0
      N = 1
      C = np.log(norm)
      Free = -(1.0/beta[p])*(np.log(Z)+4*C)/(4*N)

      for i in range (Niters):

          T, norm = CG_step(T, Tim)
          C = np.log(norm)+4*C
          N *= 4.0
          Free = -(1.0/beta[p])*(np.log(Z)+4*C)/(4*N) 
          if i == Niters-1:

            Z1 = ncon([T,T],[[1,-1,2,-2],[2,-3,1,-4]])
            Z = ncon([Z1,Z1],[[1,2,3,4],[2,1,4,3]])
            Free = -(1.0/beta[p])*(np.log(Z)+4*C)/(4*N)
            f[p] = Free 
            internalE[p] = Free*beta[p] 
            print (round(beta[p],5), Free)

    dx = beta[1]-beta[0] # Assuming equal spacing ...
    dfdx = np.gradient(internalE, dx)
    d2fdx2 = np.gradient(dfdx, dx)
    plt.plot(beta, f, marker="*", color = "r")
    plt.grid(True)
    plt.title('Free energy of Classical 2d Ising model', fontsize=20)
    plt.xlabel('Inverse Temperature, beta', fontsize=16)
    plt.ylabel('f', fontsize=16)
    plt.show()   
    plt.plot(beta, dfdx, marker="+", color = "b")
    plt.grid(True)
    plt.title('Internal energy - 2d Ising model', fontsize=20)
    plt.xlabel('Inverse Temperature, β', fontsize=16)
    plt.ylabel('E', fontsize=16)
    plt.show()
    plt.plot(beta, -d2fdx2*beta*beta, marker="*", color = "g")
    plt.grid(True)
    plt.title('Specific heat - 2d Ising model', fontsize=20)
    plt.xlabel('Inverse Temperature, β', fontsize=16)
    plt.ylabel('SH', fontsize=16)
    plt.show()
    print ("Specific heat peaks at β = ", beta[int(np.array(-d2fdx2*beta*beta).argmax())])
    print ("COMPLETED: " , datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))