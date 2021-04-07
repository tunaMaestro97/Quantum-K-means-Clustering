# Quantum-K-means-Clustering
Quantum K-means Clustering Algorithm, Coded by Abhijat Sarma. 
Based on the paper "Quantum Unsupervised and Supervised Learning on Superconducting Processors", A Sarma, R Chatterjee, K Gili, T Yu. Quantum Information and Computation 20 (7&amp;8), 541-552. 
Available at https://arxiv.org/abs/1909.04226. 
Feel free to use this code for nonprofit research, but please cite this paper if you do so.

To use:
Simply use the QKMC class in QKMC.py. For the quantum distance measure to work correctly, all data values must be positive, and not near zero. 
Positive data on the order of 10^1 should work well. This can be easily ensured by translation and scaling in preprocessing. 
