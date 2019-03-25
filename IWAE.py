import numpy as np
import matplotlib.pyplot as plt
import torch
from pylab import *
import torch.nn as nn
import torch.nn.functional as F
from util import *
from model import *
device = torch.device("cuda")



def generate_x(no_sample, mu_true):
    z = np.random.multivariate_normal(mu_true, np.diag(np.ones(x_dim)))
    x = np.random.multivariate_normal(z, np.diag(np.ones(x_dim)),size = no_sample)
    return x


def train_IWAE(no_of_particles):
    '''k = no. of particles'''
    model.train()

    optimizer.zero_grad()
    z, mu_z_given_x, logvar_z_given_x = model(X, no_of_particles)
    # loss is negative of elbo
    loss = -compute_elbo(X, z, mu_z_given_x, logvar_z_given_x)
    loss.backward()
    optimizer.step()
    return loss.item()



if __name__ == '__main__':
    x_dim = 20
    n_sample = 1024
    k = 30
    learning_rate = .001
    nb_epoch = 3000

    # Generate true mu and and x
    mu_true = np.random.multivariate_normal(np.zeros(x_dim), np.diag(np.ones(x_dim)))
    X = generate_x(n_sample, mu_true)
    X = torch.Tensor(x)
    X = X.to(device)  # will use GPU for training


    model = IWAE()
    model = model.to(device)
    mu_z_prior = torch.randn(x_dim).cuda()
    mu_z_prior.requires_grad_(True)
    params = list(model.parameters()) + [mu_z_prior]
    optimizer = torch.optim.Adam(params, lr=learning_rate)