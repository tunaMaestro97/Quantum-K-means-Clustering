# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 13:39:12 2019

@author: Abhi Sarma
"""
# useful additional packages
import random
import math
from sympy.ntheory import isprime
# importing the QISKit
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute, compile
from qiskit.tools.visualization import plot_histogram, circuit_drawer
from qiskit import BasicAer

 #Finds the overlap of data1 and data2 as long as P = 2
def get_overlap(point1, point2):
    mag_of_data1 = math.sqrt((point1[0])**2 + (point1[1])**2) #calclates the magnitude of Xi
    mag_of_data2 = math.sqrt((point2[0])**2 + (point2[1])**2) #caluclates the magnitude of Xj
    theta1 = 2 * (math.acos((point1[0] / mag_of_data1))) #the angle at which we should rotate the second qubit
    theta2 = 2 * (math.acos((point2[0] / mag_of_data2))) #the angle at which we should rotate the fourth qubit
    qr = QuantumRegister(5) #creates a quantum register of length 5
    cr = ClassicalRegister(5) #creates a classical register of length 5
    find_overlap = QuantumCircuit(qr, cr) #Defining the circuit to take in the values of qr and cr
    find_overlap.u3(theta1, 0, 0, qr[1]) #rotate the second qubit
    find_overlap.x(qr[2]) #perform a not gate on the third qubit
    find_overlap.cx(qr[1], qr[2]) #perform a comntrolled not gate from the second to the third qubit
    find_overlap.u3(theta2, 0, 0, qr[3]) #rotate the fourth qubit
    find_overlap.x(qr[4]) #perform a not gate on the 5th qubit
    find_overlap.cx(qr[3], qr[4]) #perform a controlled not gate from the 4th to the 5th qubit
    find_overlap.h(qr[0]) #hadamard the first qubit
    find_overlap.cswap(qr[0], qr[1], qr[3]) #Swap the second and fourth qubits if the first is a 1
    find_overlap.cswap(qr[0], qr[2], qr[4]) #Swap the third and fifth qubits if the first is a 1
    find_overlap.h(qr[0]) #hadamard the first qubit
    find_overlap.measure(qr[0], cr[0]) #measure the 1st qubit
    return find_overlap

def overlap_value(point1, point2):
    backend_sim = BasicAer.get_backend('qasm_simulator')
    params = get_overlap(point1, point2)
    job = execute(params, backend_sim, shots = 1000)
    result = job.result()
    counts = result.get_counts(params)
    value = counts.get('00000')
    p_zero = value / 1000
    plot_histogram(result.get_counts())
    overlap = math.sqrt(np.linalg.norm((p_zero - (.5))/(0.5)))
    return overlap

print(overlap_value([1, 0], [0, 1]))

def get_euclidean(point1, point2):
    mag_of_data1 = math.sqrt((point1[0])**2 + (point1[1])**2) #calclates the magnitude of Xi
    mag_of_data2 = math.sqrt((point2[0])**2 + (point2[1])**2) #caluclates the magnitude of Xj
    z = (mag_of_data1)**2 + (mag_of_data2)**2
    theta1 = 2 * (math.acos((point1[0] / mag_of_data1))) #the angle at which we should rotate the third qubit
    theta2 = 2 * (math.acos((point2[0] / mag_of_data2))) #the angle at which we should rotate the fifth qubit
    theta3 = 2 * (math.acos((mag_of_data1 / math.sqrt(z)))) #the angle at which we should rotate the sixth qubit
    qr = QuantumRegister(7) #creates a quantum register of length 5
    cr = ClassicalRegister(1) #creates a classical register of length 5
    find_euclidean = QuantumCircuit(qr, cr) #Defining the circuit to take in the values of qr and cr
    find_euclidean.u3(theta1, 0, 0, qr[2]) #rotate the third qubit
    find_euclidean.x(qr[3]) #perform a not gate on the fourth qubit
    find_euclidean.cx(qr[2], qr[3]) #perform a comntrolled not gate from the third to the fourth qubit
    find_euclidean.u3(theta2, 0, 0, qr[4]) #rotate the fifith qubit
    find_euclidean.x(qr[5]) #perform a not gate on the 6th qubit
    find_euclidean.cx(qr[4], qr[5]) #perform a controlled not gate from the 5th to the 6th qubit
    find_euclidean.h(qr[1]) #perform a hadamard on the second qubit
    find_euclidean.x(qr[6]) #perform a not gate on the last qubit
    find_euclidean.u3(theta3, 0, 0, qr[6]) #rotate the last qubit by theta3
    find_euclidean.h(qr[0]) #hadamard the first qubit
    find_euclidean.cswap(qr[0], qr[5], qr[6]) #Swap the second and fourth qubits if the first is a 1
    find_euclidean.cswap(qr[0], qr[4], qr[5]) #Swap the third and fifth qubits if the first is a 1
    find_euclidean.cswap(qr[0], qr[3], qr[4]) #Swap the third and fifth qubits if the first is a 1
    find_euclidean.cswap(qr[0], qr[2], qr[3]) #Swap the third and fifth qubits if the first is a 1
    find_euclidean.h(qr[0]) #hadamard the first qubit
    find_euclidean.measure(qr[6], cr) #measure the 1st qubit
    return find_euclidean

def euclidean_value(point1, point2):
    backend_sim = BasicAer.get_backend('qasm_simulator')
    params = get_euclidean(point1, point2)
    job = execute(params, backend_sim, shots = 1000)
    result = job.result()
    counts = result.get_counts(params)
    value = counts.get('0')
    p_zero = value / 1000
    mag_of_data1 = math.sqrt((point1[0])**2 + (point1[1])**2) #calclates the magnitude of Xi
    mag_of_data2 = math.sqrt((point2[0])**2 + (point2[1])**2) #caluclates the magnitude of Xj
    z = (mag_of_data1)**2 + (mag_of_data2)**2
    euclidean = z * (4 * p_zero - 2)
    return euclidean

print(euclidean_value([500, 7], [300, 4]))

# Import necessary libraries
#from copy import deepcopy
#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#from matplotlib import pyplot as plt
## Set k centers for k clusters (Example k=3)
#center_1 = np.array([4,7])
#center_2 = np.array([2,6])
#center_3 = np.array([8,1])
## Generate random data and center it to the k centers (Example k=3)
#data_1 = np.random.randn(200, 2) + center_1
#data_2 = np.random.randn(200,2) + center_2
#data_3 = np.random.randn(200,2) + center_3
#data = np.concatenate((data_1, data_2, data_3), axis = 0)
##plt.scatter(data[:,0], data[:,1], s=7)
#
## Number of clusters
#k = 2
## Number of data points
#n = data.shape[0]
## Number of features in the data
#c = data.shape[1]
## Generate random centers, here we use sigma and mean to ensure it represent the whole data
#mean = np.mean(data, axis = 0)
#std = np.std(data, axis = 0)
#centers = np.random.randn(k,c)*std + mean
## Plot the data and the centers generated as random
#plt.scatter(data[:,0], data[:,1], s=7)
#plt.scatter(centers[:,0], centers[:,1], marker='*', c='g', s=150)
#
#centers_old = np.zeros(centers.shape) # to store old centers
#centers_new = deepcopy(centers) # Store new centers
#data.shape
#clusters = np.zeros(n)
#distances = np.zeros((n,k))
#error = np.linalg.norm(centers_new - centers_old)
## When, after an update, the estimate of that center stays the same, exit loop
#count = 0
#while (error != 0 and count<10):
#    count += 1
#    print(count)
##----------------------------------------------------------------------------
#    # Measure the distance to every center using Quantum Measuring technique
#    for i in range(k):
#        #np.linalg.norm(data - centers[i], axis=1) Old Distance Technique
#        old = euclidean_value(data[0],centers[i])
#        for j in range(len(data)):
#            new = euclidean_value(data[j],centers[i])
#            if old >= new:
#                distances[j] = new
#        #-----------------------------------------------------------------------------
#        # Assign all training data to closest center
#    clusters = np.argmin(distances, axis = 1)
#    centers_old = deepcopy(centers_new)
#    # Calculate mean for every cluster and update the center
#    for i in range(k):
#        centers_new[i] = np.mean(data[clusters == i], axis=0)
#    error = np.linalg.norm(centers_new - centers_old)
#    
#centers_new
#
## Plot the data and the centers generated as random
#plt.scatter(data[:,0], data[:,1], s=7)
#plt.scatter(centers_new[:,0], centers_new[:,1], marker='*', c='g', s=150)