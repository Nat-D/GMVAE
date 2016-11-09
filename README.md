# Code for "Deep Unsupervised Clustering with Gaussian Mixture Variational Autoencoders"
(https://arxiv.org/abs/1611.02648)
By
Nat Dilokthanakul, Pedro A.M. Mediano, Marta Garnelo, Matthew C.H. Lee, Hugh Salimbeni, Kai Arulkumaran, Murray Shanahan

# Abstract
We study a variant of the variational autoencoder model with a Gaussian mixture as a prior distribution, with the goal of performing unsupervised clustering through deep generative models. We observe that the standard variational approach in these models is unsuited for unsupervised clustering, and mitigate this problem by leveraging a principled information-theoretic regularisation term known as consistency violation. Adding this term to the standard variational optimisation objective yields networks with both meaningful internal representations and well-defined clusters. We demonstrate the performance of this scheme on synthetic data, MNIST and SVHN, showing that the obtained clusters are distinct, interpretable and result in achieving higher performance on unsupervised clustering classification than previous approaches.

# Requirements
Luarocks packages:
- mnist ( torch-rocks install https://raw.github.com/andresy/mnist/master/rocks/mnist-scm-1.rockspec )

Python packages:
- torchfile (pip install torchfile)

# Instructions

To run a spiral experiment,

./run.sh spiral 

To turn off CV regularisation term (set alpha)

set flag 

	\- cvWeight 0.0

To turn off z-prior term (set eta)

	\- zPriorWeight 0.0

To run MNIST

./run.sh mnist

To use GPU 

set flag

	\- gpu 1

To run quick MNIST on fully-connected network

./run.sh mnist_fc


