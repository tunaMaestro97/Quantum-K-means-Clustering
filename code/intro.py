# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 16:09:39 2019

@author: Abhi Sarma
"""
#setup
import numpy as np
import matplotlib.pyplot as plt
from qiskit import(
        QuantumCircuit,
        execute,
        Aer)
from qiskit.visualization import plot_histogram

#circuit declaration
circuit = QuantumCircuit(2,2)

#create bell state, measure qubits
circuit.h(0)
circuit.cx(0, 1)
circuit.measure([0, 1], [0, 1])

#visualize circuit
circuit.draw(output='mpl').show()

#execute circuit with qasm simulator
simulator = Aer.get_backend('qasm_simulator')
job = execute(circuit, simulator, shots=1000)
result = job.result()
counts = result.get_counts(circuit)
print("\nTotal count for 00 and 11 are:", counts)

#visualize results
plot_histogram(counts).show()