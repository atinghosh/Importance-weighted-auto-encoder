import torch
import numpy as np

def normal_logpdf(x, mu, logvar):
    C = torch.Tensor([np.log(2. * np.pi)]).to(device)
    return (-.5 * (C + logvar) - (x - mu) ** 2. / (2. * torch.exp(logvar)))