import enum
from turtle import forward
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.beta import Beta
from typing import List

class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        """
        Network that represents the actor 
        args:
            obs_dim: dimension of the observation in input to the newtork
            act_dim: number of actuators in output
        """
        super().__init__()
        self.pi_encoder = ObservationEncoder(obs_dim=obs_dim)
        self.mean_layer = nn.Linear(64, act_dim)
        #nn.init.orthogonal_(self.mean_layer.weight.data, gain=1)
        #nn.init.constant_(self.mean_layer.bias.data, 0)
        self.std_layer = nn.Parameter(torch.zeros(act_dim))
        self.tanh = nn.Tanh()
        #self.std_layer = self.std_layer - 1
        #self.softplus = torch.nn.Softplus()
        #self.alpha_layer = nn.Linear(64, act_dim)
        #nn.init.constant_(self.alpha_layer.weight.data, 0)
        #self.beta_layer = nn.Linear(64, act_dim)
        #nn.init.constant_(self.beta_layer.weight.data, 0)

    def forward(self, obs):
        pi_obs = self.pi_encoder(obs)
        mean = self.mean_layer(pi_obs)
        std = torch.exp(self.std_layer)
        action_prob = Normal(mean, std)
        action = action_prob.sample()
        action = self.tanh(action)
        #alpha = self.softplus(self.alpha_layer(pi_obs)) + 1
        #beta = self.softplus(self.beta_layer(pi_obs)) + 1
        #action_prob = Beta(alpha, beta)
        return action_prob

class Critic(nn.Module):
    def __init__(self, obs_dim: int):
        """
        Network that represents the Critic
        args:
            obs_dim: dimension of the observation in input to the newtork
        """
        super().__init__()
        self.val_encoder = ObservationEncoder(obs_dim=obs_dim)
        self.critic_layer = nn.Linear(64,1)
        #nn.init.constant_(self.critic_layer.weight.data, 0)
        #nn.init.normal_(self.critic_layer.bias.data)
        #self.softplus = torch.nn.Softplus()

    def forward(self, obs):
        val_obs = self.val_encoder(obs)
        return self.critic_layer(val_obs)

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: List[int], act_dim: int):
        """
        args:
            obs_dim: dimension of the observation in input to the newtork
            act_dim: number of actuators in output
        """
        super().__init__()
        self.actor = Actor(obs_dim=obs_dim, act_dim=act_dim)
        self.critic = Critic(obs_dim=obs_dim)
    
    def forward(self, obs, action=None):
        action_prob = self.actor(obs)
        value = self.critic(obs)
        if action == None:
            return action_prob, value, None, None
        else:
            logp = action_prob.log_prob(action).sum(-1)
            entropy = action_prob.entropy().mean()
            return action_prob, value, logp, entropy

class SingleObservationEncoder(nn.Module):
    def __init__(self, obs_dim: int):
        """
        Encoder for a single type of observation
        args:
            obs_dim: dimension of the observation in input to the newtork
        """
        super().__init__()
        dims = [obs_dim] + [64,64]
        
        self.encoder = nn.Sequential()
        for n, (dim_in, dim_out) in enumerate(zip(dims[:-1], dims[1:])):
            self.encoder.add_module(name=f"single_observation_encoder_{n}", module=nn.Linear(in_features=dim_in, out_features=dim_out))
            self.encoder.add_module(name=f'tanh_{n}', module=nn.Tanh())

    def forward(self, obs):
        return self.encoder(obs)

class ObservationEncoder(nn.Module):
    def __init__(self, obs_dim: List[int]):
        """
        Full encoder: concatenate encoded observations and produce a 64-dim vector
        args:
            obs_dim: a list of the dimensions of all the observations to encode
        """
        super().__init__()
        self.encoders = torch.nn.ModuleList()
        self.obs_dim = obs_dim
        for obs_d in obs_dim:
            self.encoders.append(SingleObservationEncoder(obs_d))

        dims = [len(obs_dim) * 64] + [64,64]
        
        self.final_encoder = nn.Sequential()
        for n, (dim_in, dim_out) in enumerate(zip(dims[:-1], dims[1:])):
            self.final_encoder.add_module(name=f"final_observation_encoder_{n}", module=nn.Linear(in_features=dim_in, out_features=dim_out))
            self.final_encoder.add_module(name=f'tanh_{n}', module=nn.Tanh())

    def forward(self, observations):
        if len(observations[0].shape) > 1:
            encoded_observations = torch.zeros(observations[0].shape[0], len(observations) * 64)
            for i, obs in enumerate(observations):
                encoded_observations[:,i * 64: i * 64 + 64] = self.encoders[i](obs)
        else:
            encoded_observations = torch.zeros(len(observations) * 64)
            for i, obs in enumerate(observations):
                encoded_observations[i * 64: i * 64 + 64] = self.encoders[i](obs)

        return self.final_encoder(encoded_observations)