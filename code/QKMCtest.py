# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 14:41:24 2019

@author: Abhi Sarma
"""
from datasets import *
from QKMC import QKMC
from matplotlib import pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D as ax3d
import timeit

from qiskit import BasicAer
from qiskit import Aer

feature_dim=2 # we support feature_dim 2 or 3
sample_Total, training_input, test_input, class_labels = Iris(
    training_size=10, 
    test_size=30, 
    n=feature_dim, 
    #gap=0.3,
    PLOT_DATA=False
)
test_input = training_input
#print(test_input)
for key in test_input:
    for i in test_input[key]:
        i *= 10
        i += 20
print(test_input)
backend = Aer.get_backend('qasm_simulator')

kmc = QKMC(feature_dim, False, backend, test_input)
qkmc = QKMC(feature_dim, True, backend, test_input)

#now = timeit.default_timer()
#class_class = kmc.run()
#elapsed = timeit.default_timer()-now
#now = timeit.default_timer()
#class_class = qkmc.run()
#elapsed = timeit.default_timer()-now

#for key in class_class:
#    for i in class_class[key]:
#        i -= 20
#        i /= 10
#    for i in quant_class[key]:
#        i -= 20
#        i /= 10
        
new_class1 = {'A': class_class['0'], 'B': class_class['1'], 'C': class_class['2']}
new_class2 = {'A': class_class['0'], 'B': class_class['2'], 'C': class_class['1']}
new_class3 = {'A': class_class['1'], 'B': class_class['0'], 'C': class_class['2']}
new_class4 = {'A': class_class['1'], 'B': class_class['2'], 'C': class_class['0']}
new_class5 = {'A': class_class['2'], 'B': class_class['0'], 'C': class_class['1']}
new_class6 = {'A': class_class['2'], 'B': class_class['1'], 'C': class_class['0']}
        
#new_class1 = {'A': class_class['0'], 'B': class_class['1']}
#new_class2 = {'A': class_class['1'], 'B': class_class['0']}
num_wrong = 0
for i in test_input:
    for j in test_input[i]:
        is_wrong = True
        for k in new_class1[i]:
            if (k==j).all():
                is_wrong = False
        if is_wrong:
            num_wrong += 1
#print(num_wrong)
print('Accuracy: ', (1-num_wrong/30)*100, '%')
num_wrong = 0
for i in test_input:
    for j in test_input[i]:
        is_wrong = True
        for k in new_class2[i]:
            if (k==j).all():
                is_wrong = False
        if is_wrong:
            num_wrong += 1
print('Accuracy: ', (1-num_wrong/30)*100, '%')
num_wrong = 0
for i in test_input:
    for j in test_input[i]:
        is_wrong = True
        for k in new_class3[i]:
            if (k==j).all():
                is_wrong = False
        if is_wrong:
            num_wrong += 1
print('Accuracy: ', (1-num_wrong/30)*100, '%')
num_wrong = 0
for i in test_input:
    for j in test_input[i]:
        is_wrong = True
        for k in new_class4[i]:
            if (k==j).all():
                is_wrong = False
        if is_wrong:
            num_wrong += 1
print('Accuracy: ', (1-num_wrong/30)*100, '%')
num_wrong = 0
for i in test_input:
    for j in test_input[i]:
        is_wrong = True
        for k in new_class5[i]:
            if (k==j).all():
                is_wrong = False
        if is_wrong:
            num_wrong += 1
print('Accuracy: ', (1-num_wrong/30)*100, '%')
num_wrong = 0
for i in test_input:
    for j in test_input[i]:
        is_wrong = True
        for k in new_class6[i]:
            if (k==j).all():
                is_wrong = False
        if is_wrong:
            num_wrong += 1
print('Accuracy: ', (1-num_wrong/30)*100, '%')
#
print(elapsed)

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}
matplotlib.rc('font', **font)

fig, ax = plt.subplots(ncols=2)
ax[1].set_title('Quantum Clusters')
ax[0].set_title('True Boundaries')
#plt.title('Clusters on ad_hoc')
colors = ['green', 'blue', 'red']
fakestrings = {'0': 'A', '1': 'B', '2': 'C'}
for i in range(len(class_class)):
    X = []
    Y = []
    X1 = []
    Y1 = []
    for j in range(len(class_class[str(i)])):
        X.append(class_class[str(i)][j][0])
        Y.append(class_class[str(i)][j][1])
    for j in range(len(test_input[fakestrings[str(i)]])):
        X1.append(test_input[fakestrings[str(i)]][j][0])
        Y1.append(test_input[fakestrings[str(i)]][j][1])
    #plt.scatter(X, Y, c=colors[i])
    ax[1].scatter(X, Y, c=colors[i], s=250)
    ax[0].scatter(X1, Y1, c=colors[i], s=250)
plt.show()