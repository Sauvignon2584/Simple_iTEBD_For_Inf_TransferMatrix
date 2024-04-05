# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 20:08:17 2024

@author: Souvignon
"""

import numpy as np
import matplotlib.pyplot as plt


# \tau ~ \beta
# for : exp(-\beta H)
# FreeE ~ -KT ln(Z) ~ (-1/\beta) ln(Z)
# FreeE ~ (-1/((i+1)*0.010)) ln(Z)

# InternalE ~ FreeE * ((i+1)*0.010)
# dfdx = np.gradient(InternalE,0.010)
# d2fdx2 = np.gradient(dfdx,0.010)    
TRG_FreeE = \
    [-2.635196902667435559,
-2.573435395314507268,
-2.516564103549669973,
-2.464176788748784652,
-2.415919147792111144,
-2.371482217944389959,
-2.330597099524018390,
-2.293030948933557411,
-2.258584156622415406,
-2.227088895342361319e+00,
-2.198409415165320624e+00,
-2.172445295873458360e+00,
-2.149140684483680452e+00,
-2.128509835002935890e+00,
-2.110745310872728098e+00,
-2.096407727822293232e+00,
-2.084502698975942536e+00,
-2.074378602978998565e+00,
-2.065675827867783809e+00,
-2.058141876438006168e+00,
-2.051585568185311903e+00,
-2.045856333440933650e+00,
-2.040832667723459615e+00,
-2.036414827240766723e+00,
-2.032519905995671117e+00,
-2.029078325009565642e+00,
-2.026031236320437579e+00,
-2.023328549088077910e+00,
-2.020927396212139904e+00,
-2.018790920986767823e+00,
]


Test_observation_FreeE = \
    [-2.6036772416132976, -2.57343539626973, -2.544415168794959, -2.5165641053366663, -2.489833227786511, -2.464176792016772, -2.4395520707411547, -2.4159191590232183, -2.393240800820996, -2.371482235316788, -2.350611062152508, -2.330597125094598, -2.311412414117446, -2.293030986461373, -2.2754289079537737, -2.258584216878177, -2.242476914096705, -2.2270889852500892, -2.2124044641598726, -2.198409551937361, -2.185092815522652, -2.1724455061131835, -2.1604620705258677, -2.1491409978464517, -2.138486309552488, -2.12851047053151, -2.1192412364599216, -2.110746544221862, -2.1032021664609064, -2.096408384661627, -2.0902039382992825, -2.0845030714528994, -2.0792441582777803, -2.074378815023721, -2.069867283018216, -2.0656759676390912, -2.0617759442164125, -2.0581419699671177, -2.054751789699359, -2.051585625390038, -2.04862578763372, -2.0458563716130116, -2.043263013853027, -2.0408326940364234, -2.03855357106061, -2.0364148456843973, -2.034406644193191, -2.0325199189455874, -2.0307463626621107, -2.029078334036171, -2.0275087927702033, -2.026031242529542, -2.0246396806077334, -2.023328553315446, -2.0220927162894977, -2.0209273990536696, -2.01982817326917, -2.0187909242144997, -2.017811825090174, -2.0168873138141286]
True_FreeE = \
    [-2.635196903170878, -2.57343539626973, -2.5165641053366663, -2.464176792016772, -2.4159191590232183, -2.371482235316789, -2.330597125094598, -2.2930309864613725, -2.258584216878178, -2.2270889852500897, -2.1984095519373623, -2.1724455061131844, -2.149140997844733, -2.128510213178903, -2.110318233007635, -2.0964082936411534, -2.0845030714507935, -2.074378815023981, -2.065675967639089, -2.0581419699669943, -2.0515856253899276, -2.045856371613164, -2.0408326940364443, -2.03641484568439, -2.0325199189455563, -2.0290783340361807, -2.0260312425296734, -2.0233285533154004, -2.0209273990536336, -2.0187909242144895]
FreeE = True_FreeE

FreeE = \
    [-2.6351969031708764, -2.573435396269732, -2.51656410533667, -2.4641767920167754, -2.4159191590232196, -2.3714822353167913, -2.3305971250945956, -2.2930309864613725, -2.258584216878176, -2.227088985250091, -2.198409551937366, -2.1724455061131853, -2.1491409978464517, -2.1285104705315114, -2.1107413140474858, -2.0964083846615633, -2.0845030714527764, -2.07437881502391, -2.065675967639155, -2.0581419699669965, -2.0515856253898352, -2.0458563716130547, -2.0408326940363306, -2.036414845684341, -2.0325199189454732, -2.0290783340361735, -2.026031242529615, -2.0233285533153715, -2.020927399053601, -2.0187909242144904]
FreeE = \
    [-2.635196903170875, -2.5734353962697307, -2.5165641053366676, -2.4641767920167736, -2.4159191590232214, -2.371482235316792, -2.3305971250945956, -2.2930309864613756, -2.2585842168781745, -2.22708898525009, -2.1984095519373668, -2.172445506113185, -2.1491409978464535, -2.128510470531511, -2.11074660386692, -2.096408384661559, -2.0845030714527795, -2.0743788150239113, -2.0656759676391547, -2.0581419699669983, -2.051585625389838, -2.0458563716130524, -2.040832694036331, -2.036414845684344, -2.032519918945475, -2.029078334036174, -2.026031242529617, -2.0233285533153698, -2.0209273990536025, -2.01879092421449]

beta = \
    np.linspace(0.3,0.6,len(FreeE))
    # np.linspace(0.30,0.60,len(FreeE))
       
      # np.linspace(0.30,0.60,len(FreeE))\
          

InternalE = []

for i in range(len(FreeE)):
    InternalE.append(
        (beta[i]) * FreeE[i]
        )

dfdx = np.gradient(InternalE,beta[1]-beta[0])
d2fdx2 = np.gradient(dfdx,beta[1]-beta[0]) 
   
Cv = []
for i in range(len(FreeE)):
    Cv.append(
        -(beta[i])*(beta[i])*d2fdx2[i]
        )

## 画图:能量
plt.plot(
    np.linspace(0.30,0.60,len(dfdx)),
    dfdx,
    marker="+",
    color ='r' 
    )
plt.grid(True)
# plt.scatter(
#     np.linspace(0.30,0.60,len(dfdx)),
#     dfdx)

plt.title('Internal energy - 2d Ising model With TransferMatrix_TEBD', fontsize=20)
plt.xlabel('Inverse Temperature, β', fontsize=16)
plt.ylabel('E', fontsize=16)
plt.show()

## 画图:热容
plt.plot(
    np.linspace(0.30,0.60,len(d2fdx2)),
    # - d2fdx2,
    Cv,
    marker = '*',
    color = 'g'
    )
plt.grid(True)
# plt.scatter(
#     np.linspace(0.30,0.60,len(d2fdx2)),
#     - d2fdx2
#     )
plt.title('Specific heat C - 2d Ising model With TransferMatrix_TEBD', fontsize=20)
plt.xlabel('Inverse Temperature, β', fontsize=16)
plt.ylabel('C', fontsize=16)

plt.show()