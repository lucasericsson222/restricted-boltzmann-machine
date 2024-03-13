import numpy as np

W = np.matrix([ [-9, -12, 4], [-9, 4, -12], [-1, -10, -10] ])
visible_biases = np.matrix([6, 6, 4]).T
hidden_biases = np.matrix([4, 6, 6]).T

def sigmoid(x):  
    return np.exp(-np.logaddexp(0, -x))

def energy(v: np.matrix, h: np.matrix):
    return -1 * (v.T @ W @ h + hidden_biases.T @ h + visible_biases.T @ v )[0,0]

def probHgivenV(v: np.matrix, h:np.matrix):
    product = 1
    for j in range(0, h.shape[0]):
        product *= sigmoid(hidden_biases[0][j] + (W[j] @ v))
    return product 

def probVgivenH(v:np.matrix, h:np.matrix):
    product = 1
    for k in range(0, v.shape[0]):
        product *= sigmoid(visible_biases[0][k] + (h.T @ (W.T)[k].T))
    return product

Z = 0
for i1 in range(0, 2):
    for i2 in range(0, 2):
        for i3 in range(0, 2):
            for j1 in range(0, 2):
                for j2 in range(0, 2):
                    for j3 in range(0, 2):
                        print("V: " + str(np.matrix([i1, i2, i3]).T))
                        print("H: " + str(np.matrix([j1, j2, j3]).T))
                        print(energy(np.matrix([i1, i2, i3]).T, np.matrix([j1, j2, j3]).T))
                        Z += np.exp(-1 * energy(np.matrix([i1, i2, i3]).T, np.matrix([j1, j2, j3]).T))

def softplus(x):
    return np.log(1 + np.exp(x))

# this is slightly more efficient, but still uses the partition function Z, which is intractable
def probV(v: np.matrix):
    inner_sum = 0
    for i in range(0, hidden_biases.T.shape[1]):
        inner_sum += softplus(hidden_biases.T[0,i] + (W[i] @ v)[0,0])
    return np.exp(visible_biases.T @ v + inner_sum)[0,0] / Z

# this is the brute force approach for calculating the probability
# prob_v = sum_h p(v,h) / sum_v,h p(v,h)
def probV2(v: np.matrix):
    inner_sum = 0
    for i1 in range(0, 2):
        for i2 in range(0, 2):
            for i3 in range(0, 2):
                inner_sum += np.exp(-1 * energy(v, np.matrix([i1, i2, i3]).T))
    return inner_sum / Z


print("----------------")
print("Brute Force Method")
print("000: " + str(probV2(np.matrix([0,0,0]).T)))
print("001: " + str(probV2(np.matrix([0,0,1]).T)))
print("010: " + str(probV2(np.matrix([0,1,0]).T)))
print("011: " + str(probV2(np.matrix([0,1,1]).T)))
print("100: " + str(probV2(np.matrix([1,0,0]).T)))
print("101: " + str(probV2(np.matrix([1,0,1]).T)))
print("110: " + str(probV2(np.matrix([1,1,0]).T)))
print("111: " + str(probV2(np.matrix([1,1,1]).T)))
print("----------------")
print("Softplus Method")
print("000: " + str(probV(np.matrix([0,0,0]).T)))
print("001: " + str(probV(np.matrix([0,0,1]).T)))
print("010: " + str(probV(np.matrix([0,1,0]).T)))
print("011: " + str(probV(np.matrix([0,1,1]).T)))
print("100: " + str(probV(np.matrix([1,0,0]).T)))
print("101: " + str(probV(np.matrix([1,0,1]).T)))
print("110: " + str(probV(np.matrix([1,1,0]).T)))
print("111: " + str(probV(np.matrix([1,1,1]).T)))

# this was checking to make sure that it was normalizing correctly.
#mysum = 0
#for i1 in range(0, 2):
#    for i2 in range(0, 2):
#        for i3 in range(0, 2):
#            mysum += probV(np.matrix([i1, i2, i3]).T)
#print(mysum)

def h_from_v(v: np.matrix):
    out = np.zeros((3, 1))
    for i in range(0, out.shape[0]):
        myrand = np.random.rand()
        cutoff = sigmoid(hidden_biases[i][0] + (W[i] @ v))[0,0]
        if myrand < cutoff:
            out[i] = 1
        else:
            out[i] = 0
    return out

def v_from_h(h: np.matrix):
    out = np.zeros((3,1))
    for i in range(0, out.shape[0]):
        myrand = np.random.rand()
        cutoff = sigmoid(visible_biases[i][0] + (W[i] @ h))[0,0]
        if myrand < cutoff:
            out[i] = 1
        else:
            out[i] = 0
    return out

def gibbs_sampling_prob_v():
    init_v = np.random.randint(0, 2, (3, 1))

    k = 5
    for i in range(0, k):
        h = h_from_v(init_v)
        init_v = v_from_h(h)
    
    return init_v

# this array is ordered as
# [000, 001, 010, ... ]
res = [0, 0, 0, 0, 0, 0, 0, 0]
count: float = 10000.0
for i in range(0, int(count)):
    val = gibbs_sampling_prob_v()
    res[int(4 * val[0,0] + 2*val[1, 0] + val[2, 0])] += 1 

res = [val / count for val in res]
print("----------------")
print("Gibbs Sampling")
for i in range(0, 8):
    binnum = f'{i:03b}'
    print(binnum + ": " + str(res[i]))

print("----------------")

# todo:
# * do everything inside a class
# * have later gibbs samples be based off of the previous ones
# * put in the training as well
