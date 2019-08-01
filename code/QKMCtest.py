# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 14:41:24 2019

@author: Abhi Sarma
"""
from datasets import *
from QKMC import QKMC
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as ax3d

from qiskit import BasicAer
from qiskit import Aer

feature_dim=2 # we support feature_dim 2 or 3
sample_Total, training_input, test_input, class_labels = Wine(
    training_size=10, 
    test_size=50, 
    n=feature_dim, 
    #gap=0.3,
    PLOT_DATA=False
)

#print(test_input)
for key in test_input:
    for i in test_input[key]:
        i *= 10
        i += 20

backend = Aer.get_backend('qasm_simulator')

kmc = QKMC(feature_dim, False, backend, test_input)
qkmc = QKMC(feature_dim, True, backend, test_input)

class_class = kmc.run()
quant_class = qkmc.run()

fig, ax = plt.subplots(2)
ax[1].set_title('Quantum')
ax[0].set_title('Classical')
plt.title('Clusters on Wine')
colors = ['green', 'blue', 'red']
for i in range(len(class_class)):
    X = []
    Y = []
    X1 = []
    Y1 = []
    for j in range(len(class_class[str(i)])):
        X.append(class_class[str(i)][j][0])
        Y.append(class_class[str(i)][j][1])
    for j in range(len(quant_class[str(i)])):
        X1.append(quant_class[str(i)][j][0])
        Y1.append(quant_class[str(i)][j][1])
    ax[0].scatter(X, Y, c=colors[i])
    ax[1].scatter(X1, Y1, c=colors[i])
plt.show()