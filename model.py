import torch.nn as nn
import torch.nn.functional as F
import torch


class IWAE(nn.Module):

    def __init__(self):
        super(IWAE, self).__init__()
        self.fc = nn.Linear(20, 20)

    def reparametrize(self, mu, logvar, k=1):
        '''k = no. of particles, returns a tensor of size (k, batch_size, latent_space_dim)'''
        latent_size = mu.size()
        std = torch.exp(0.5 * logvar)
        eps = torch.randn((k,) + latent_size[1:])
        eps = eps.to(device)
        return eps.mul(std).add_(mu)

    def encode(self, x):
        z = self.fc(x)
        return z

    def forward(self, x, k=1):
        mu_z_given_x = self.encode(x)
        batch_size = x.size(0)
        logvar_z_given_x = torch.log((2 / 3) * torch.ones(batch_size, x_dim))  # Does not depend on x in this model
        logvar_z_given_x = logvar_z_given_x.to(device)
        mu_z_given_x, logvar_z_given_x = mu_z_given_x[None, ...], logvar_z_given_x[None, ...]
        z = self.reparametrize(mu_z_given_x, logvar_z_given_x, k=k)
        return z, mu_z_given_x, logvar_z_given_x





def compute_elbo(x, z, mu_z_given_x, logvar_z_given_x):
    zeros_prior = torch.zeros_like(z).cuda()
    zeros = torch.zeros(z.size()).cuda()

    logpx_given_z = normal_logpdf(x, z, zeros)
    logpz = normal_logpdf(z, mu_z_prior, zeros_prior)
    logqz = normal_logpdf(z, mu_z_given_x, logvar_z_given_x)

    logpx_given_z = torch.sum(logpx_given_z, dim=-1)
    logpz = torch.sum(logpz, dim=-1)
    logqz = torch.sum(logqz, dim=-1)

    logw = logpx_given_z + logpz - logqz
    assert len(logw.size()) == 2

    with torch.no_grad():
        reweight = torch.exp(logw - torch.logsumexp(logw, dim=0, keepdim=True))

    return torch.mean(torch.sum(reweight * logw, dim=0), dim=0)