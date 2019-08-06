# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 11:49:53 2019

@author: Abhi Sarma
"""

import logging

import numpy as np
import random

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.tools import parallel_map
from qiskit.tools.events import TextProgressBar
from qiskit.aqua.algorithms import QuantumAlgorithm
from qiskit.aqua.utils import split_dataset_to_data_and_labels
from qiskit.aqua.components.feature_maps.raw_feature_vector import RawFeatureVector

from qiskit import execute

logger = logging.getLogger(__name__)

class QKMC(QuantumAlgorithm):
    """
    Quantum K-means clustering algorithm.
    """
    
    CONFIGURATION = {
        'name': 'QKMC',
        'description': 'QKMC Algorithm',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'QKMC_schema',
            'type': 'object',
            'properties': {
            },
            'additionalProperties': False
        },
        'problems': ['classification'],
        'depends': [
        ],
    }

    BATCH_SIZE = 1000
    
    def __init__(self, feature_dim, is_quantum, backend, test_dataset):
        """
        K-means Clustering Classification Algorithm
        """
        super().__init__()

        self.test_dataset = None
        self.class_to_label = None
        self.label_to_class = None
        self.num_classes = None
        
        self.feature_dim = feature_dim
        self.is_quantum = is_quantum
        
        self.setup_test_data(test_dataset)
        
        self.backend = backend
    def setup_test_data(self, test_dataset):
        """Setup test data, if the data were there, they would be overwritten.
        Args:
            test_dataset (dict): test dataset.
        """
        if test_dataset is not None:
            if self.class_to_label is None:
                logger.warning("The mapping from the class name to the label is missed, "
                               "regenerate it but it might be mismatched to previous mapping.")
                self.test_dataset, self.class_to_label = split_dataset_to_data_and_labels(test_dataset)
            else:
                self.test_dataset = split_dataset_to_data_and_labels(test_dataset, self.class_to_label)
            self.num_clusters = len(list(self.class_to_label.keys()))
    
    def _run(self):
        """
        Classify with k-means clustering algorithm
        """
        cluster_assignments = self.get_initial_clusters()
        stop = False
        count = 1
        while(not stop):
            if (count>5):
                print('Algorithm failed to converge: run again')
                return cluster_assignments
            cluster_assignments, stop = self.iterate(cluster_assignments)
            print(count)
            #print(cluster_assignments)
            #print()
            count+=1
        return cluster_assignments
        
    def get_initial_clusters(self):
        """
        Randomly assign each datapoint to a cluster
        """
        cluster_assignments = {}
        cluster_arrays = []
        for i in range(self.num_clusters):
            cluster_arrays.append([])
        for i in self.test_dataset[0]:
            cluster = random.randint(0, self.num_clusters-1)
            cluster_arrays[cluster].append(i)
        for i in range(len(cluster_arrays)):
            cluster_assignments.update({str(i): cluster_arrays[i]})
        #print(cluster_assignments)
        #print()
        return cluster_assignments
    
    def iterate(self, cluster_assignments):
        stop = True
        new_cluster_assignments = {}
        centroids = []
        cluster_arrays = []
        for i in range(self.num_clusters):
            cluster_arrays.append([])
            centroids.append(QKMC.calculate_centroid(self.feature_dim, cluster_assignments[str(i)]))
        for i in self.test_dataset[0]:
            closest_cluster = self.closest_cluster(i, centroids)
            cluster_arrays[closest_cluster].append(i)
            if(stop):
                old_cluster = 1000
                for j in range(len(cluster_assignments)):
                    for k in cluster_assignments[str(j)]:
                        if (k==i).all():
                            old_cluster = j
                if(old_cluster != closest_cluster):
                    stop=False
                    
        for i in range(len(cluster_arrays)):
            new_cluster_assignments.update({str(i): cluster_arrays[i]})
        return (new_cluster_assignments, stop)
    
    @staticmethod
    def calculate_centroid(feature_dim, cluster_array):
        """
        Calculate centroid of a cluster
        """
        if (len(cluster_array)==0):
            return np.zeros(feature_dim)
        centroid = []
        for i in range(feature_dim):
            featuremean = 0
            for j in range(len(cluster_array)):
                featuremean += cluster_array[j][i]
            featuremean /= len(cluster_array)
            centroid.append(featuremean)
        return centroid
    
    def closest_cluster(self, x, centroids):
        if (self.is_quantum):
            quant = QKMC.closest_cluster_quantum(self.backend, x, centroids)
            #classic = QKMC.closest_cluster_classical(x, centroids)
            #if (quant != classic):
                #print('Wrong')
            return quant
        else:
            return QKMC.closest_cluster_classical(x, centroids)
        
    @staticmethod
    def closest_cluster_classical(x, centroids):
        """
        Calculate closest centroid from a data point
        """
        distances = []
        for i in range(len(centroids)):
            distances.append(QKMC.classical_calculate_squared_distance(x, centroids[i]))
        return distances.index(min(distances))
    
    @staticmethod
    def closest_cluster_quantum(backend, x, centroids):
        """
        Calculate closest centroid from a data point
        """
        distances = []
        for i in range(len(centroids)):
            distances.append(QKMC.quantum_calculate_squared_distance(backend, x, centroids[i]))
        return distances.index(min(distances))
    
    @staticmethod
    def classical_calculate_squared_distance(x, y):
        squares = 0
        if(len(x) != len(y)):
            raise ValueError("x and y must be of same length")
        for i in range(len(x)):
            squares += (x[i]-y[i])**2
        return squares
    
    @staticmethod
    def quantum_calculate_squared_distance(backend, X, Y):
        if(len(X) != len(Y)):
            raise ValueError("x and y must be of same length")
        #feature vector converter
        c0 = ClassicalRegister(1)
        q0 = QuantumRegister(1)
        zerocircuit = QuantumCircuit(q0)
        zerocircuit.h(q0)
        
        Y = np.array(Y)
        
        if(np.linalg.norm(Y)==0):
            return 1000000
        
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
        
        q3 = QuantumRegister(1)
        swapcircuit = psicircuit+phicircuit
        swapcircuit.add_register(q3)
        swapcircuit.add_register(c0)
        swapcircuit.h(q3)
        swapcircuit.cswap(q3, q0, q2[0])
        swapcircuit.h(q3)
        swapcircuit.measure(q3, c0)
        result = execute(swapcircuit, backend, shots=40000).result()
        squares = Z*((4*result.get_counts()['0']/40000.0)-2)
        #print('error ', abs(100*(squares - QKMC.classical_calculate_squared_distance(X, Y))/QKMC.classical_calculate_squared_distance(X, Y)), "%")
        return squares