# Restricted Boltzman Machine

This is a project created to be a learning experience for myself about
how Restricted Boltzmann Machines (RBMs) work.
It implements CD-k with numpy along with Gibb's Block Sampling.

The weights are formulated in this machine as
$$ v.T W h $$.

All vectors that are passed in should be column vectors.

## Resources

This RBM currently implements:

- Normal Gaussian Initial Weights
- Mean Squared Reconstruction Error Logging
- Mini Batch Size of 100
- Learning rate of 0.01
  
all advised from the following paper:
[https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf](A Practical Guide to Training
Restricted Boltzmann Machines)
