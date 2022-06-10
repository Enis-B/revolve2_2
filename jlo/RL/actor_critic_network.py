from turtle import forward
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
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
        self.mean_layer = nn.Linear(obs_dim, act_dim)
        #self.std_layer = nn.Parameter(torch.zeros(1, act_dim))
        self.std_layer = nn.Linear(obs_dim, act_dim)

    def forward(self, obs):
        mean = self.mean_layer(obs)
        #std = torch.exp(self.std_layer)
        std = torch.exp(self.std_layer(obs))
        action_prob = Normal(mean, std)
        action = action_prob.sample()
        logp = action_prob.log_prob(action).sum(-1, keepdim=True)
        return action, logp

class Critic(nn.Module):
    def __init__(self, obs_dim: int):
        """
        Network that represents the Critic
        args:
            obs_dim: dimension of the observation in input to the newtork
        """
        super().__init__()
        self.critic_layer = nn.Linear(obs_dim,1)

    def forward(self, obs):
        return self.critic_layer(obs)

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: List[int], act_dim: int):
        """
        args:
            obs_dim: dimension of the observation in input to the newtork
            act_dim: number of actuators in output
        """
        super().__init__()
        self.val_encoder = ObservationEncoder(obs_dim=obs_dim)
        self.pi_encoder = ObservationEncoder(obs_dim=obs_dim)
        self.actor = Actor(obs_dim=64, act_dim=act_dim)
        self.critic = Critic(obs_dim=64)
    
    def forward(self, obs):
        val_obs = self.val_encoder(obs)
        pi_obs = self.pi_encoder(obs)
        action, logp = self.actor(pi_obs)
        value = self.critic(val_obs)
        return value, action, logp

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
            self.encoder.add_module(name='tanh', module=nn.Tanh())

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
        for obs_d in obs_dim:
            self.encoders.append(SingleObservationEncoder(obs_d))

        dims = [len(obs_dim) * 64] + [64,64]
        
        self.final_encoder = nn.Sequential()
        for n, (dim_in, dim_out) in enumerate(zip(dims[:-1], dims[1:])):
            self.final_encoder.add_module(name=f"final_observation_encoder_{n}", module=nn.Linear(in_features=dim_in, out_features=dim_out))
            self.final_encoder.add_module(name='tanh', module=nn.Tanh())

    def forward(self, observations):
        if len(observations.shape) > 2:
            encoded_observations = torch.zeros(observations.shape[1], observations.shape[0] * 64)
            for i, obs in enumerate(observations):
                encoded_observations[:,i * 64: i * 64 + 64] = self.encoders[i](obs)
        else:
            encoded_observations = torch.zeros(observations.shape[0] * 64)
            for i, obs in enumerate(observations):
                encoded_observations[i * 64: i * 64 + 64] = self.encoders[i](obs)

        return self.final_encoder(encoded_observations)

