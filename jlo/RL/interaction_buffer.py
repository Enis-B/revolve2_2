import torch
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.sampler import SubsetRandomSampler


class Buffer(object):
    """
    Replay buffer
    It stores observations, actions, values and rewards for each step of the simulation
    Used to create batches of data to train the controller 
    """
    def __init__(self, obs_dim, act_dim, num_agents):
        self.observations = torch.zeros(obs_dim[0],128, num_agents, obs_dim[1])
        self.actions = torch.zeros(128, num_agents, act_dim)
        self.values = torch.zeros(128, num_agents)
        self.rewards = torch.zeros(128, num_agents)
        self.logps = torch.zeros(128, num_agents)
        self.advantages = torch.zeros(128, num_agents)
        self.returns = torch.zeros(128, num_agents)
        self.next_state_value = torch.zeros(num_agents)

        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.step = 0

    def insert(self, obs, act, logp, val, rew):
        """
        Insert a new step in the replay buffer
        args:
            obs: observation at the current state
            act: action performed at the state
            logp: log probability of the action performed according to the current policy
            val: value of the state
            rew: reward received for performing the action
        """
        self.observations[:,self.step] = obs
        self.actions[self.step] = act
        self.values[self.step] = val
        self.rewards[self.step] = rew
        self.logps[self.step] = logp

        self.step += 1

    def set_next_state_value(self, next_state_value):
        """
        Insert the value of the last state reached
        """
        self.next_state_value = next_state_value

    def _compute_advantages(self):
        """
        Compute the advantage function and the returns used to compute the loss
        """
        gamma = 0.99
        lam = 0.95
        adv = 0
        vals = torch.cat((self.values, self.next_state_value.unsqueeze(0)), dim=0)
        for t in range(127, -1, -1):
            delta = self.rewards[t] + gamma * vals[t+1] - vals[t]
            adv = delta + (lam * gamma) * adv
            self.advantages[t] = adv
            self.returns[t] = adv + vals[t]

    def _normalize_rewards(self):
        """
        Normalize the rewards obtained
        """
        rewards = self.rewards.view(-1)
        mean = rewards.mean(dim=0)
        std = rewards.std()
        self.rewards = (self.rewards - mean) / std

    def get_sampler(self,):
        """
        Create a BatchSampler that divides the data in the buffer in batches 
        """
        dset_size = 128 * self.num_agents
        batch_size = 1024

        assert dset_size >= batch_size

        #self._normalize_rewards()
        #self._compute_advantages()

        sampler = BatchSampler(
            SubsetRandomSampler(range(dset_size)),
            batch_size,
            drop_last=True,
        )

        for idxs in sampler:
            obs = self.observations.view(self.obs_dim[0], -1, self.obs_dim[1])[:,idxs]
            val = self.values.view(-1, 1)[idxs]
            act = self.actions.view(-1, self.act_dim)[idxs]
            logp_old = self.logps.view(-1, 1)[idxs]
            rew = self.rewards.view(-1, 1)[idxs]
            adv = self.advantages.view(-1, 1)[idxs]
            ret = self.returns.view(-1, 1)[idxs]
            yield obs, val, act, logp_old, rew, adv, ret
