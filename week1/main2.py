# ---------------------------------------
# This file is just an attempt to use the numbers given in the 
# actual research paper, since I was wondering if that could be causing the 
# discrepancies in the numbers I am getting.
# ---------------------------------------
import numpy as np

W = np.matrix([ [0, -1, 2], [-1, 0, 2], [2, 2, 0] ])
visible_biases = np.matrix([-2, 1, 1]).T
hidden_biases = np.matrix([1, 1, -2]).T

def sigmoid(x):  
    return np.exp(-np.logaddexp(0, -x))

def energy(v: np.matrix, h: np.matrix):
    return -1 * (h.T @ W @ v + hidden_biases.T @ h + visible_biases.T @ v )[0,0]

def probHgivenV(v: np.matrix, h:np.matrix):
    product = 1
    for j in range(0, h.shape.count):
        product *= sigmoid(hidden_biases[0][j] + (W[j] @ v))
    return product 

def probVgivenH(v:np.matrix, h:np.matrix):
    product = 1
    for k in range(0, v.shape.count):
        product *= sigmoid(visible_biases[0][k] + (h.T @ (W.T)[k].T))
    return product

Z = 0

for i1 in range(0, 2):
    for i2 in range(0, 2):
        for i3 in range(0, 2):
            for j1 in range(0, 2):
                for j2 in range(0, 2):
                    for j3 in range(0, 2):
                        Z += np.exp(-1 * energy(np.matrix([i1, i2, i3]).T, np.matrix([j1, j2, j3]).T))

def softplus(x):
    return np.log(1 + np.exp(x))

def probV(v: np.matrix):
    inner_sum = 0
    for i in range(0, hidden_biases.T.shape[1]):
        inner_sum += softplus(hidden_biases.T[0,i] + (W[i] @ v)[0,0])
    return np.exp(visible_biases.T @ v + inner_sum)[0,0] / Z
print(probV(np.matrix([0,0,0]).T))
print(probV(np.matrix([0,0,1]).T))
print(probV(np.matrix([0,1,0]).T))
print(probV(np.matrix([0,1,1]).T))
print(probV(np.matrix([1,0,0]).T))
print(probV(np.matrix([1,0,1]).T))
print(probV(np.matrix([1,1,0]).T))
print(probV(np.matrix([1,1,1]).T))

# this was checking to make sure that it was normalizing correctly.
#mysum = 0
#for i1 in range(0, 2):
#    for i2 in range(0, 2):
#        for i3 in range(0, 2):
#            mysum += probV(np.matrix([i1, i2, i3]).T)
#print(mysum)