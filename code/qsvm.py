# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 11:22:01 2019

@author: Abhi Sarma
"""
from datasets import *

from qiskit import BasicAer
from qiskit.aqua.utils import split_dataset_to_data_and_labels, map_label_to_class_name
from qiskit.aqua.input import ClassificationInput
from qiskit.aqua import run_algorithm, QuantumInstance
from qiskit.aqua.algorithms import QSVM
from qiskit.aqua.components.feature_maps import SecondOrderExpansion

# setup aqua logging
import logging
from qiskit.aqua import set_qiskit_aqua_logging
set_qiskit_aqua_logging(logging.DEBUG)  # choose INFO, DEBUG to see the log

from qiskit import IBMQ
IBMQ.load_account()

feature_dim=2 # we support feature_dim 2 or 3
sample_Total, training_input, test_input, class_labels = ad_hoc_data(
    training_size=4, 
    test_size=2, 
    n=feature_dim, 
    gap=0.3,
    PLOT_DATA=True
)

seed = 10598

feature_map = SecondOrderExpansion(feature_dimension=feature_dim, depth=2, entanglement='linear')
qsvm = QSVM(feature_map, training_input, test_input)

backend = BasicAer.get_backend('qasm_simulator')
device = IBMQ.get_provider().get_backend('ibmqx2')
quantum_instance = QuantumInstance(device, shots=1024, seed=seed, seed_transpiler=seed, skip_qobj_validation=False)

result = qsvm.run(quantum_instance)

print("testing success ratio: ", result['testing_accuracy'])
