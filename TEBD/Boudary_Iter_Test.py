# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 20:44:23 2024

@author: Souvignon
"""

import numpy as np
from doTEBD import doTEBD
from ncon import ncon
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.linalg import expm
from scipy.sparse.linalg import LinearOperator, eigs
from doTEBD import apply_gate_MPS

def Boundary_iter(A_up,B_up,sAB,sBA,A_down,B_down,sAB_,sBA_,gateAB,gateBA):
    
    chi = A_up.shape[0];d = gateAB.shape[0]
    
    sAB = np.diag(sAB);sBA = np.diag(sBA)
    sAB_ = np.diag(sAB_);sBA_ = np.diag(sBA_)
    
    Right_iter = np.random.rand(chi,d,chi)
    # Right_iter += Right_iter.transpose(2,1,0)
    Right_iter /= LA.norm(Right_iter)
    tol = 1e-10
    
    
    for i in range(800):
    
        tensors = [np.sqrt(sBA),np.sqrt(sBA_),A_up,A_down,sAB,
                   sAB_,B_up,B_down,np.sqrt(sBA),np.sqrt(sBA_),
                   gateAB,gateBA,Right_iter]
        labels = [[-1,1],[-3,2],[1,3,5],[2,4,6],[5,7],[6,8],[7,15,9],[8,14,10],[9,11],[10,12],
                  [-2,16,3,15],[4,14,16,13],[11,13,12]]    
    
        Right_iter_new = ncon(tensors,labels)
        
        Eig_Factor = LA.norm(Right_iter_new)
        
        Right_iter_new /= Eig_Factor
        
        
        # print(Eig_Factor)
        
        if LA.norm( Right_iter_new - Right_iter ) < tol :
            print('success!')
            print('current Eig_Factor:',Eig_Factor)
            break 
            # print('iter: %d, diff: %e' % (i, LA.norm(Right_iter_new - Right_iter)))
        
        Right_iter = Right_iter_new
        
        if i+1 == 800:
            print('current Eig_Factor:',Eig_Factor )
    
    
    return Eig_Factor


def Transfer_Tensor_Eig(A, B, sAB, sBA, gateAB, gateBA, ob = True):
    ''' 
      Final Version for z ~ < psi_R | T | psi_L > / < psi_R | psi_L >,
      With N -> +inf, Z = z^N, z for FreeE per sites... 
     
    '''
    chi = A.shape[0]; d = gateAB.shape[0]
    
    # sAB,sBA = np.diag(sAB),np.diag(sBA)
    if ob==True:
        # Inner_Product of < psi_R | T | psi_L > per_2sites
        A,sAB,B = apply_gate_MPS(gateAB, A, sAB, B, sBA, chi, normalize=False)
    if ob==False:
        # Inner_Product of < psi_R | psi_L > per_2sites
        A,sAB,B = A,sAB,B
    
    sAB,sBA = np.diag(sAB),np.diag(sBA)
    
    tensors = [np.sqrt(sBA),np.sqrt(sBA),
               A, sAB, B,
               A.conj(),sAB,B.conj(),
               np.sqrt(sBA),np.sqrt(sBA)]
    open_labels = [[-1,1],[-3,3],
                   [1,-2,2],[2,4],[4,5,6],
                   [3,5,7],[7,8],[8,-5,9],
                   [6,-4],[9,-6]]
    
    Transfer_matrix = (ncon(tensors,open_labels)).reshape(chi*d*chi,chi*d*chi)
    
    Eigvals,EigVect = np.linalg.eig(Transfer_matrix)
    
    return Eigvals[0]
 
   
# Init_Experimental_Environment

# set bond dimensions and simulation options
chi = 16  # bond dimension
# tau = 0.4  # timestep
numiter = 1000  # number of timesteps
evotype = "imag"  # real or imaginary time evolution
E0 = -4 / np.pi  # specify exact ground energy (if known)
# midsteps = int(1 / tau)  # timesteps between MPS re-orthogonalization
midsteps = 100

# define Hamiltonian (quantum XX model)
sX = np.array([[0, 1], [1, 0]])
sY = np.array([[0, -1j], [1j, 0]])
sZ = np.array([[1, 0], [0, -1]])

# define Atensor,Btensor,SABweight,SBAweight
# # Representing Ho
hamAB = (np.real(np.kron(sZ,sZ))).reshape(2, 2, 2, 2)
# # Representing He
hamBA = (np.real(np.kron(sZ,sZ))).reshape(2, 2, 2, 2)

# hamAB = (np.real(np.kron(sX, sX) + np.kron(sY, sY))).reshape(2, 2, 2, 2)
# hamBA = (np.real(np.kron(sX, sX) + np.kron(sY, sY))).reshape(2, 2, 2, 2)


# # Init Bulk Tensor
d = hamAB.shape[0]
sAB_init = np.ones(chi) / np.sqrt(chi)
sBA_init = np.ones(chi) / np.sqrt(chi)
a = np.random.rand(chi, d, chi)
A0 = a / LA.norm(a)
b = np.random.rand(chi, d, chi)
B0 = b / LA.norm(b)

# # Init Contraction gate
def Z2d_Ising(beta, h=0):
    ''' bata for tau'''
    betah = beta*h 
    tau = np.exp(0.250000*betah) 
    a = np.sqrt(np.cosh(beta)) 
    b = np.sqrt(np.sinh(beta)) 
    W = np.array([[a*tau,b*tau],[(a/tau),-(b/tau)]]) 
    out = np.einsum("ia, ib, ic, id  -> abcd", W, W, W, W) 
    return out 

# print('Heating Preparation steps for 4/20')

# for i in range(1,5):
    
#     tau = i/2500
    
#     A0, B0, sAB_init, sBA_init, rhoAB, rhoBA = \
#         doTEBD(hamAB, hamBA, A0, B0, sAB_init, sBA_init, chi,
#             tau, 
#             Func_type = 'evol',
#             evotype=evotype, 
#             numiter=numiter, 
#             midsteps=midsteps, 
#             E0=E0, Canonical_Option = True)


FreeE = []
midsteps = 1000

for i in range(30,60):
# for i in range(20,90):
    
    tau = 0.010 * i
    # tau = 1/10000
    gateAB = Z2d_Ising(tau, h=0)
    gateBA = Z2d_Ising(tau, h=0)

    A, B, sAB, sBA, rhoAB, rhoBA = \
        doTEBD(hamAB, hamBA, A0, B0, sAB_init, sBA_init, chi,
            tau, 
            Func_type = 'contract',
            # Func_type = 'evol',
            evotype=evotype, 
            numiter=numiter, 
            midsteps=midsteps, 
            E0=E0, Canonical_Option = False)


    # gateAB = expm(-tau * hamAB.reshape(d**2, d**2)).reshape(d, d, d, d)
    # gateBA = expm(-tau * hamBA.reshape(d**2, d**2)).reshape(d, d, d, d)
    
    Sites2_PartitionFunc_Evaluation = \
        np.real(Transfer_Tensor_Eig(A, B, sAB, sBA, gateAB, gateBA)/ \
                Transfer_Tensor_Eig(A, B, sAB, sBA, gateAB, gateBA,ob=False))


    FreeE_current = ((-1/(2 * tau)) * np.log(Sites2_PartitionFunc_Evaluation))
    FreeE.append(FreeE_current)
    print('with current EigFactors:',FreeE_current)
    
# plt.title('Free energy of Quantum 1d XX model With TransferMatrix_TEBD', fontsize=20)

plt.title('Free energy of Classical 2d Ising model With TransferMatrix_TEBD', fontsize=20)
plt.plot(np.linspace(0.3,0.6,len(FreeE)),FreeE)
plt.scatter(np.linspace(0.3,0.6,len(FreeE)),FreeE)

# plt.plot(np.linspace(2,9,len(FreeE)),FreeE)
# plt.scatter(np.linspace(2,9,len(FreeE)),FreeE)

# # For quantum Lattice \beta = numiter * \tau
# plt.grid(True)
# plt.show()