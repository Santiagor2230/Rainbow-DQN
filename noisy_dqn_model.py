import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.nn.init import kaiming_uniform_, zeros_
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class NoisyLinear(nn.Module):
  def __init__(self, in_features, out_features, sigma):
    super(NoisyLinear, self).__init__()

    # we create the Linear regression: X*W + B
    self.w_mu = nn.Parameter(torch.empty((out_features, in_features)))
    self.w_sigma = nn.Parameter(torch.empty((out_features, in_features)))
    self.b_mu = nn.Parameter(torch.empty((out_features)))
    self.b_sigma = nn.Parameter(torch.empty((out_features)))

    kaiming_uniform_(self.w_mu, a = math.sqrt(5))
    kaiming_uniform_(self.w_sigma, a = math.sqrt(5))
    zeros_(self.b_mu)
    zeros_(self.b_sigma)

  def forward(self, x, sigma=0.5):
    if self.training:
      w_noise = torch.normal(0, sigma, size=self.w_mu.size()).to(device)
      b_noise = torch.normal(0, sigma, size=self.b_mu.size()).to(device)

      return F.linear(x, self.w_mu + self.w_sigma * w_noise, self.b_mu + self.b_sigma * b_noise)

    else:
      return F.linear(x,self.w_mu, self.b_mu)

class DQN(nn.Module):

  def __init__(self, hidden_size, obs_shape, n_actions, sigma=0.5, atoms=51):
    super().__init__()
    self.atoms = atoms
    self.n_actions = n_actions

    self.conv = nn.Sequential(
        nn.Conv2d(obs_shape[0], 64, kernel_size=3),
        nn.MaxPool2d(kernel_size=4),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3),
        nn.MaxPool2d(kernel_size=4),
        nn.ReLU(),
    )

    conv_out_size = self._get_conv_out(obs_shape)

    self.head = nn.Sequential(
        NoisyLinear(conv_out_size, hidden_size, sigma=sigma),
        nn.ReLU()
    )

    self.fc_adv = NoisyLinear(hidden_size, self.n_actions * self.atoms, sigma=sigma)
    self.fc_value = NoisyLinear(hidden_size, self.atoms, sigma=sigma)

  def _get_conv_out(self, shape):
    conv_out = self.conv(torch.zeros(1, *shape))
    return int(np.prod(conv_out.size()))

  def forward(self, x):
    x = self.conv(x.float()).view(x.size()[0], -1)
    x = self.head(x)
    ''' [-10, -9, -8,...8,9,10] we getting the probabilities
     (B,A, N), B= size of batch-rows,
     A = number of action that can be take-columns
     N = Atoms -depth dimension
     we have right now (B,A*N) so to get to (B,A,N)
     we use the view method to reshape (-1, number of actions, depth of atoms)'''
    adv = self.fc_adv(x).view(-1, self.n_actions, self.atoms)

    value = self.fc_value(x).view(-1, 1, self.atoms) #(B,N) -> (B,1,N) with view()

    q_logits = value + adv - adv.mean(dim=1, keepdim=True) #(B, A, N) linear
    q_probs = F.softmax(q_logits, dim=-1) #(B,A,N) probabilities
    return q_probs