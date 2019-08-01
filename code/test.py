# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 14:40:35 2019

@author: Abhi Sarma
"""
import numpy as np
import timeit

import math

import QKMC

from qiskit.aqua.components.feature_maps.raw_feature_vector import RawFeatureVector

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import BasicAer, execute
from qiskit import Aer

#X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32])
X = np.array([-0.35,  0.35])
#Y=X*10
#X = np.array([312, 523])
Y = X+0.1
#Y = np.array([ -0.35, 0.70])
#Y = np.array([1, 0])
#Y=1000*np.array([0.75, -0.5])

c0 = ClassicalRegister(1)
q0 = QuantumRegister(1)
zerocircuit = QuantumCircuit(q0)
zerocircuit.h(q0)

fvc = RawFeatureVector(len(X))
q1 = QuantumRegister(fvc.num_qubits)
q2 = QuantumRegister(fvc.num_qubits)
ketxcircuit = fvc.construct_circuit(X, qr=q1)
ketycircuit = fvc.construct_circuit(Y, qr=q2)

psicircuit = zerocircuit+ketxcircuit+ketycircuit
for i in range(fvc.num_qubits):
    psicircuit.cswap(q0, q1[i], q2[i])
    
psicircuit.barrier(q0, q1, q2)
psicircuit.reset(q2)

Z=0
for i in range(len(X)):
    Z += X[i]**2+Y[i]**2

fvc2 = RawFeatureVector(2)
p1 = np.linalg.norm(X)
p2 = -np.linalg.norm(Y)
phi = np.array([p1, p2])
phicircuit = fvc2.construct_circuit(phi, qr=q2)
#print(phicircuit)

q3 = QuantumRegister(1)
fcircuit = QuantumCircuit(q3, c0)
swapcircuit = fcircuit+psicircuit+phicircuit
swapcircuit.h(q3)

#for i in range(fvc.num_qubits+1):
#    if i==0:
#        swapcircuit.cswap(q3, q0, q2[0])
#    else:
#        swapcircuit.cswap(q3, q1[i-1], q2[i])
swapcircuit.cswap(q3, q0, q2[0])

swapcircuit.h(q3)
swapcircuit.measure(q3, c0)

#print(swapcircuit)
##num qubits = 2*roof(log_2(feature_dim))+3
simulator = Aer.get_backend('qasm_simulator')
#result = execute(swapcircuit, simulator, shots=4096).result()
#distance = Z*((4*result.get_counts()['0']/4096.0)-2)
#print(result.get_counts())
print('fake distance ', QKMC.QKMC.quantum_calculate_squared_distance(simulator, X, Y))
distance = 0
for i in range(len(X)):
    distance += (X[i]-Y[i])**2
print('real distance ', distance)