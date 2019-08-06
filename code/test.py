# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 14:40:35 2019

@author: Abhi Sarma
"""
import numpy as np
from numpy import array
import timeit

import math

import QKMC

import matplotlib.pyplot as plt

from qiskit.aqua.components.feature_maps.raw_feature_vector import RawFeatureVector

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import BasicAer, execute
from qiskit import Aer
from qiskit.tools.visualization import plot_state_city
from qiskit import IBMQ

import logging
from qiskit.aqua import set_qiskit_aqua_logging
set_qiskit_aqua_logging(False)

#X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32])
X = np.array([1,  1])
#Y=X*10
#X = np.array([312, 523])
Y = [2, 0]
#Y = np.array([ -0.35, 0.70])
#Y = np.array([1, 0])
#Y=1000*np.array([0.75, -0.5])

backend = IBMQ.get_provider().get_backend('ibmq_qasm_simulator')
#print(QKMC.QKMC.quantum_calculate_squared_distance(backend, X, Y))

classifications = {'0': [array([11.81649276, 24.54856634]), array([13.42459402, 24.10990129]), array([11.55199032, 23.35616928]), array([14.85233046, 21.50430736]), array([14.32223019, 21.71420362]), array([14.93111274, 23.33765323]), array([16.76214486, 27.19700542]), array([13.51492985, 24.80276145]), array([15.0103456 , 26.63683521]), array([15.29714333, 27.07302925]), array([13.86292148, 19.30636735]), array([13.32833298, 22.3789185 ]), array([13.88470985, 22.29855478]), array([12.65330286, 26.66149104]), array([15.4309023 , 26.15439088]), array([14.03974876, 26.49109187]), array([12.63121823, 21.54360233])], '1': [array([22.89750891, 20.64559515]), array([25.31739056, 18.1450818 ]), array([26.3054508 , 20.16370253]), array([22.78651727, 23.64446433]), array([27.67264928, 20.69607017]), array([23.49041929, 18.17306578]), array([23.73633997, 19.88208634]), array([22.24457921, 18.40077177]), array([22.07618066, 19.26461151]), array([24.81843404, 20.6220368 ]), array([18.79502435, 17.21022472]), array([22.53576089, 15.97393167]), array([22.25872545, 17.18469893]), array([25.57427542, 19.5125783 ]), array([21.32256107, 17.58447166])], '2': [array([23.27533514, 18.90026478]), array([19.15622682, 17.11472094]), array([20.68448347, 16.45266443]), array([19.40108305, 10.        ]), array([17.47444147, 14.32327414]), array([20.7508271 , 14.10558629]), array([18.2604467 , 15.13015395]), array([23.26617773, 15.99122675]), array([24.07378217, 13.97632705]), array([16.70443056, 19.30250409]), array([18.30338469, 19.23836884]), array([17.0087138 , 14.43424701]), array([17.79578683, 15.5432762 ]), array([18.59838382, 14.78733198]), array([21.03105278, 12.67621569]), array([17.9850352 , 14.31759193]), array([23.62191241, 14.37281409]), array([17.60012086, 13.93757481])]}

colors = ['b', 'r', 'g']
for i in range(len(classifications)):
    X = []
    Y = []
    for j in range(len(classifications[str(i)])):
        X.append(classifications[str(i)][j][0])
        Y.append(classifications[str(i)][j][1])
    plt.scatter(X, Y, c=colors[i])
plt.show()