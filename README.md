# Cre-DPSIGN

## document info

- options.py contains all setting parameters
- There are seven executive documents in our experiment, named CreFlip.py, CreGau.py with two differential privacy mechanisms, and baseline: RSA.py, RSA-flip.py, RSA-Gau.py, SGD.py, SIGNSGD.py.
- Attack.py contains the setting of attacks


## Execution command
terminal run: python CreFlip.py --eps=0.4 --lr=0.04 --byzantinue_users=10 

## Experiment results
The experiment results(.pkl) are automatically stored in different packages to distinguish different attacks, such as sign_flipping, same_value. 
We also have two evaluation documents named draw.py and draw_cre.py, representing accuracy and robustness respectively.

## Running time
In the MNIST dataset around 6 hours, in the CIFAR dataset around 5 days.[Our code has potential for further optimization in terms of time, which will be pursued in future work.]
