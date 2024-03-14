import numpy as np
import itertools as it

class RestrictedBoltzmannMachine:
    def __init__(self, visible_biases, hidden_biases, weights):
        self.visible_biases = visible_biases
        self.hidden_biases = hidden_biases
        self.weights = weights
        self.init_z()

    def init_z(self):
        self.Z = 0
        for v in it.product(range(2), repeat=3):
            for h in it.product(range(2), repeat=3):
                self.Z += np.exp(-1 * self.energy(np.matrix([v[0], v[1], v[2]]).T, np.matrix([h[0], h[1], h[2]]).T))
    
    def sigmoid(x):  
        return np.exp(-np.logaddexp(0, -x))
    
    def energy(self, v: np.matrix, h: np.matrix):
        return -1 * (v.T @ self.weights @ h + self.hidden_biases.T @ h + self.visible_biases.T @ v )[0,0]

    def prob_h_given_v(self, v: np.matrix, h:np.matrix):
        product = 1
        for j in range(0, h.shape[0]):
            product *= RestrictedBoltzmannMachine.sigmoid(self.hidden_biases[0][j] + (self.weights[j] @ v))
        return product 

    def prob_v_given_h(self, v:np.matrix, h:np.matrix):
        product = 1
        for k in range(0, v.shape[0]):
            product *= RestrictedBoltzmannMachine.sigmoid(self.visible_biases[0][k] + (h.T @ (self.weights.T)[k].T))
        return product

    def softplus(x):
        return np.log(1 + np.exp(x))

    # this is slightly more efficient, but still uses the partition function Z, which is intractable
    def prob_v_softplus(self, v: np.matrix):
        inner_sum = 0
        for i in range(0, self.hidden_biases.T.shape[1]):
            inner_sum += RestrictedBoltzmannMachine.softplus(self.hidden_biases.T[0,i] + (v.T @ (self.weights.T[i].T))[0,0])
        return np.exp(self.visible_biases.T @ v + inner_sum)[0,0] / self.Z

    # this is the brute force approach for calculating the probability
    # prob_v = sum_h p(v,h) / sum_v,h p(v,h)
    def prob_v_brute_force(self, v: np.matrix):
        inner_sum = 0
        for h in it.product(range(2), repeat=3):
            inner_sum += np.exp(-1 * self.energy(v, np.matrix([h[0], h[1], h[2]]).T))
        return inner_sum / self.Z

    def h_from_v(self, v: np.matrix):
        dist = RestrictedBoltzmannMachine.sigmoid(self.hidden_biases + (v.T @ self.weights).T)
        unif = np.random.rand(3,1)
        return unif <= dist

    def v_from_h(self, h: np.matrix):
        dist = RestrictedBoltzmannMachine.sigmoid(self.visible_biases + self.weights @ h)
        unif = np.random.rand(3,1)
        return unif <= dist

    def gibbs_sample_v(self, init_v: np.matrix):
        k = 5
        for _ in range(0, k):
            h = self.h_from_v(init_v)
            init_v = self.v_from_h(h)
        
        return init_v

    def prob_v_gibbs_sampling(self) -> list[float]:

        init_v_sample = np.random.randint(0, 2, (3, 1))
        res = [0, 0, 0, 0, 0, 0, 0, 0]
        val = init_v_sample
        count: float = 10000.0
        for i in range(0, int(count)):
            val = self.gibbs_sample_v(val) 
            res[int(4 * val[0,0] + 2*val[1, 0] + val[2, 0])] += 1 
        
        res = [val / count for val in res]
        return res