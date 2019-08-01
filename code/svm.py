# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 11:53:52 2019

@author: Abhi Sarma
"""

from datasets import *

import matplotlib.pyplot as plt
import numpy as np

from qiskit import BasicAer
from qiskit import IBMQ
from qiskit.aqua.utils import split_dataset_to_data_and_labels, map_label_to_class_name
from qiskit.aqua.input import ClassificationInput
from qiskit.aqua import run_algorithm, QuantumInstance

from qiskit.aqua.algorithms import SVM_Classical
from qiskit.aqua.components.feature_maps import PauliExpansion
from qiskit.aqua.algorithms import QSVM

from qiskit.aqua.components.multiclass_extensions.one_against_rest import OneAgainstRest
from qiskit.aqua.algorithms.classical.svm._rbf_svc_estimator import _RBF_SVC_Estimator
from qiskit.aqua.algorithms.many_sample.qsvm._qsvm_estimator import _QSVM_Estimator

# setup aqua logging
import logging
from qiskit.aqua import set_qiskit_aqua_logging
set_qiskit_aqua_logging(logging.DEBUG)  # choose INFO, DEBUG to see the log

IBMQ.load_account()

feature_dim=3 # we support feature_dim 2 or 3
sample_Total, training_input, test_input, class_labels = Breast_cancer(
    training_size=50, 
    test_size=25, 
    n=feature_dim, 
    #gap=0.3,
    PLOT_DATA=False
)

seed = 1001

multi_ext_classical = OneAgainstRest(_RBF_SVC_Estimator)
svm = SVM_Classical(training_input, test_input)#, multiclass_extension=multi_ext_classical)

feature_map = PauliExpansion(feature_dimension=feature_dim, depth=3, paulis=['Z', 'ZZ', 'ZZZ'], entanglement='linear')
multi_ext_quantum = OneAgainstRest(_QSVM_Estimator, [feature_map])
qsvm = QSVM(feature_map, training_input, test_input)#, multiclass_extension=multi_ext_quantum)

#backend = BasicAer.get_backend('qasm_simulator')
device = IBMQ.get_provider().get_backend('ibmqx2')
#classical_instance = QuantumInstance(backend, shots=1024, seed=seed, seed_transpiler=seed)
quantum_instance = QuantumInstance(device, shots=1024, seed=seed, seed_transpiler=seed, circuit_caching=False, skip_qobj_validation=False)

#start_classical = time.time()
#result_classical = svm.run(classical_instance)
#elapsed_classical = time.time() - start_classical

result_quant = qsvm.run(quantum_instance)

#print("classical testing success ratio: ", result_classical['testing_accuracy'])
#print("classical runtime: ", elapsed_classical)
print("quantum testing success ratio: ", result_quant['testing_accuracy'])