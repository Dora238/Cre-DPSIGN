# Cre-DPSIGN

## paper info

**Credit-based Differential Privacy Stochastic Model Aggregation Algorithm for Robust Federated Learning via Blockchain** is submitted in 52nd International Conference on Parallel Processing Proceedings.

## Document description
There are seven executive documents in our experiment, named CreFlip.py, CreGau.py with two differential privacy mechanisms, and baseline: RSA.py, RSA-flip.py, RSA-Gau.py, SGD.py, SIGNSGD.py. 

The options.py has all setting parameters 

## code executing
For executing CreFlip.py: python CreFlip.py --eps=0.4 --lr=0.04 --byzantinue_users=10 

## running time
In the MNIST dataset around 6 hours, and in the CIFAR dataset around 5 days. (Our code has the potential for further optimization in terms of time, which will be pursued in future work.)

## experiment results
The experiment results(.pkl) are automatically stored in different packages to distinguish different attacks, such as sign_flipping, same_value. We also have two evaluation documents named draw.py and draw_cre.py, representing accuracy and robustness respectively.

 
