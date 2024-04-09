import numpy as np
import itertools as it
import wandb


class RestrictedBoltzmannMachine:
    Z = None
    def __init__(self, visible_count, hidden_count):
        self.visible_biases = np.zeros((visible_count, 1))
        self.hidden_biases = np.zeros((hidden_count, 1))
        self.weights = np.random.normal(0, 0.01, (visible_count, hidden_count))
        self.minibatchsize = 100
        self.learning_rate = 0.01

    def init_z(self):
        self.Z = 0
        for v in it.product(range(2), repeat=self.visible_biases.size):
            for h in it.product(range(2), repeat=self.hidden_biases.size):
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
        if self.Z == None:
            self.init_z()
        inner_sum = 0
        for i in range(0, self.hidden_biases.T.shape[1]):
            inner_sum += RestrictedBoltzmannMachine.softplus(self.hidden_biases.T[0,i] + (v.T @ (self.weights.T[i].T))[0,0])
        return np.exp(self.visible_biases.T @ v + inner_sum)[0,0] / self.Z

    # this is the brute force approach for calculating the probability
    # prob_v = sum_h p(v,h) / sum_v,h p(v,h)
    def prob_v_brute_force(self, v: np.matrix):
        if self.Z == None:
            self.init_z()
        inner_sum = 0
        for h in it.product(range(2), repeat=self.hidden_biases.size):
            inner_sum += np.exp(-1 * self.energy(v, np.matrix([h[0], h[1], h[2]]).T))
        return inner_sum / self.Z

    def h_from_v(self, v: np.matrix):
        dist = RestrictedBoltzmannMachine.sigmoid(self.hidden_biases + (v.T @ self.weights).T)
        unif = np.random.rand(dist.shape[0], dist.shape[1])

        return (unif <= dist).astype(int)

    def v_from_h(self, h: np.matrix):
        dist = RestrictedBoltzmannMachine.sigmoid(self.visible_biases + self.weights @ h)
        unif = np.random.rand(dist.shape[0], dist.shape[1])
        return (unif <= dist).astype(int)

    def gibbs_sample_v(self, init_v: np.matrix, k):
        for _ in range(0, k):
            h = self.h_from_v(init_v)
            init_v = self.v_from_h(h)
        
        return init_v

    def conditional_gibbs_sample(self, init_v: np.matrix, k):
        out = init_v
        for _ in range(0, k):
            h = self.h_from_v(out)
            out = self.v_from_h(h)
            out[:392] = init_v[:392]
        return out

    def prob_v_gibbs_sampling(self) -> list[float]:
        init_v_sample = np.random.randint(0, 2, (3, 1))
        res = [0, 0, 0, 0, 0, 0, 0, 0]
        val = init_v_sample
        count: float = 10000.0
        for i in range(0, int(count)):
            val = self.gibbs_sample_v(val, 5) 
            res[int(4 * val[0,0] + 2*val[1, 0] + val[2, 0])] += 1 
        
        res = [val / count for val in res]
        return res

    def constrastive_divergence_step(self, v_data: np.matrix):

        h_data =self.h_from_v(v_data)
        v_sample = self.v_from_h(h_data)

        for i in range(0,4):
            h_temp = self.h_from_v(v_sample)
            v_sample = self.v_from_h(h_temp)
        
        h_sample = self.h_from_v(v_sample)

        weights_dif = self.learning_rate * (v_data @ h_data.T - v_sample @ h_sample.T) / self.minibatchsize 
        self.weights += weights_dif
        self.hidden_biases += self.learning_rate * (h_data - h_sample).sum(axis=1) / self.minibatchsize
        self.visible_biases += self.learning_rate * (v_data - v_sample).sum(axis=1) / self.minibatchsize
    
    def contrastive_divergence(self, training_data: np.matrix, num_epochs=10):
        print(training_data.shape)
        w_diffs = [100] # temp value that doesn't ever get used, but prevents error on first while loop run
        epoch = 0
        while w_diffs[-1] > 10**(-3):
            w_old = self.weights.copy()
            print(f"Epoch: {epoch}")
            for i in range(0, 100):
                
                j = i * self.minibatchsize 

                self.constrastive_divergence_step(np.matrix(training_data[j:j+self.minibatchsize]).T)
            cur_diff = self.calculate_relative_mean_absolute(w_old)

            # log changes
            print(f"current_weight_diff: {cur_diff}")
            average_reconstruction_error = self.calculate_reconstruction_error(training_data)
            average_weight = self.calculate_average_weight()
            print(f"average_reconstruction_error: {average_reconstruction_error}")
            wandb.log({"weight_diff": cur_diff, "average_weight": average_weight, "average_reconstruction_error": average_reconstruction_error})

            w_diffs.append(cur_diff)
            if epoch == num_epochs - 1:
                break
            epoch += 1

    def calculate_reconstruction_error(self, data, num_gibbs_steps=100, num_trials=100): 
        myin = np.matrix(data[np.random.randint(0,data.shape[0])])
        inimg = myin.reshape(28,28).copy()
        inimg[14:, :] = 0

        out = self.gibbs_sample_v(np.matrix(inimg.reshape(1, 28*28)).T, num_gibbs_steps)
        for i in range(0, num_trials):
            out += self.gibbs_sample_v(np.matrix(inimg.reshape(1, 28*28)).T, num_gibbs_steps)
        out[0, 392:] = out[0, 392:] / (num_trials + 1)

        return np.sum(np.abs((myin.T) - out)) / myin.shape[1]
    
    def calculate_sampled_average_free_energy(data, num_samples=100) -> float:
        return

    def free_energy(self, v: np.matrix) -> float:
        def x_j(j: int) -> float:
            return self.hidden_biases[j] + np.sum(v.T @ self.weights)
        return - np.sum(v.T @ self.visible_biases) - np.sum(np.log(1 + np.exp(np.matrix(list(map(x_j, range(0, self.hidden_biases.shape[0])))))))

    def calculate_average_weight(self):
        cur_w_arr = self.weights.reshape((1, -1))
        return np.sum(np.abs(cur_w_arr)) / cur_w_arr.shape[1] 
            
    def calculate_relative_mean_absolute(self, old_w):
        old_w_arr = old_w.reshape((1, -1)) 
        cur_w_arr = self.weights.reshape((1, -1))
        return np.sum(np.abs((cur_w_arr - old_w_arr) / old_w_arr)) / cur_w_arr.shape[1]