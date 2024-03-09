import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import random

from torch import distributions as pyd
from torch.distributions.utils import _standard_normal


# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=[30, 30]):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim[0])
        self.fc2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.fc3 = nn.Linear(hidden_dim[1], action_dim)
        for l in [self.fc1, self.fc2, self.fc3]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))
        return action 

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=[30, 30, 30]):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim[0])
        self.fc2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.fc3 = nn.Linear(hidden_dim[1], hidden_dim[2])
        self.fc4 = nn.Linear(hidden_dim[2], 1)
        for l in [self.fc1, self.fc2, self.fc3, self.fc4]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
            
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.fc4(x)
        return q_value

# Noise exploration
class ExplorationNoise:
    def __init__(self, action_dim, max_sigma=0.5, min_sigma=0.1, decay_period=100000):
        self.action_dim = action_dim
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.current_step = 0

    def get_action(self, action, t=0):
        sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return action + np.random.normal(0, sigma, size=self.action_dim)

#the truncated normal distribution
class TruncatedNormal(pyd.Normal):
	"""Utility class implementing the truncated normal distribution."""
	def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
		super().__init__(loc, scale, validate_args=False)
		self.low = low
		self.high = high
		self.eps = eps

	def _clamp(self, x):
		clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
		x = x - x.detach() + clamped_x.detach()
		return x

	def sample(self, clip=None, sample_shape=torch.Size()):
		shape = self._extended_shape(sample_shape)
		eps = _standard_normal(shape,
							   dtype=self.loc.dtype,
							   device=self.loc.device)
		eps *= self.scale
		if clip is not None:
			eps = torch.clamp(eps, -clip, clip)
		x = self.loc + eps
		return self._clamp(x)

# Algorithme TD3
class TD3:
    def __init__(self, state_dim, action_dim, actor_hidden_dim=[200, 200], critic_hidden_dim=[350, 350, 350], lr_actor=1e-3, lr_critic=1e-4, gamma=0.99, tau=0.001, policy_noise=0.1, noise_clip=0.5, batch_size=64, buffer_capacity=1000000):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_dim, action_dim, actor_hidden_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, actor_hidden_dim).to(self.device)
        self.critic1 = Critic(state_dim, action_dim, critic_hidden_dim).to(self.device)
        self.critic1_target = Critic(state_dim, action_dim, critic_hidden_dim).to(self.device)
        self.critic2 = Critic(state_dim, action_dim, critic_hidden_dim).to(self.device)
        self.critic2_target = Critic(state_dim, action_dim, critic_hidden_dim).to(self.device)
        
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr_critic)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr_critic)
        
        self.criterion = nn.MSELoss()
        
        self.replay_buffer = []
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.exploration_noise = ExplorationNoise(action_dim=action_dim)

        self.update_step = 0

    def sample_pi_actions(self, state, std=0.05):

        state = state.clone().detach().unsqueeze(0)

        action = self.actor(state)
        if std > 0:
            std = torch.ones_like(action) * std
            return TruncatedNormal(action, std).sample(clip=0.3)
        return self.exploration_noise.get_action(action)

    def select_action_TD(self, state):
        state = state.clone().detach().unsqueeze(0)
        return self.actor(state).cpu().squeeze(0).detach().numpy()

    def select_action_pi(self, state):

        state = torch.tensor(state, dtype=torch.float).to(self.device)
        state = state.unsqueeze(0)
        state = state.clone().detach().unsqueeze(0)
        action = self.actor(state).cpu().squeeze(0).detach().numpy()

        return action[0]

    
    def add_to_buffer(self, state, action, reward, next_state, terminated, done):
        data = (state, action, reward, next_state, terminated, done)
        if len(self.replay_buffer) < self.buffer_capacity:
            self.replay_buffer.append(data)
        else:
            self.replay_buffer.pop(0)
            self.replay_buffer.append(data)


    def update(self):
        if len(self.replay_buffer) < 10000:
            return

        # Sample a batch from the replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, terminated_batch, done_batch = zip(*batch)

        state_batch_np = np.array(state_batch, dtype=np.float32)
        state_batch = torch.tensor(state_batch_np, dtype=torch.float32, device=self.device)

        action_batch_np = np.array(action_batch, dtype=np.float32)
        action_batch = torch.tensor(action_batch_np, dtype=torch.float32, device=self.device)

        reward_batch_np = np.array(reward_batch, dtype=np.float32)
        reward_batch = torch.tensor(reward_batch_np, dtype=torch.float32, device=self.device).view(-1, 1)
        
        next_state_batch_np = np.array(next_state_batch, dtype=np.float32)
        next_state_batch = torch.tensor(next_state_batch_np, dtype=torch.float32, device=self.device)


        done_batch_np = np.array(done_batch, dtype=np.float32)
        done_batch = torch.tensor(done_batch_np, dtype=torch.float32, device=self.device).view(-1, 1)

        terminated_batch_np = np.array(terminated_batch, dtype=np.float32)
        terminated_batch = torch.tensor(terminated_batch_np, dtype=torch.float32, device=self.device).view(-1, 1)



        # Compute target Q value using the target actor and target critics
        with torch.no_grad():

            next_action = self.actor_target(next_state_batch)

            noise = torch.clamp(torch.randn_like(next_action) * self.policy_noise, -self.noise_clip, self.noise_clip)

            next_action = torch.clamp(next_action + noise, -1.0, 1.0)
            target_q1 = self.critic1_target(next_state_batch, next_action)
            target_q2 = self.critic2_target(next_state_batch, next_action)
            target_q = torch.min(target_q1, target_q2)
            
            target_q = reward_batch + (self.gamma * (1 - terminated_batch) * target_q).detach()

        # Update the critics
        q1 = self.critic1(state_batch, action_batch)
        q2 = self.critic2(state_batch, action_batch)

        

        critic1_loss = self.criterion(q1, target_q)
        critic2_loss = self.criterion(q2, target_q)
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic1_loss.backward()
        critic2_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        # Delayed update for the actor and target networks
        if self.update_step % 2 == 0:
            actor_loss = -self.critic1(state_batch, self.actor(state_batch)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update for target networks
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.update_step += 1

    def save(self, filename):
         
        torch.save(self.critic1.state_dict(), filename + "_critic1")
        #torch.save(self.critic1_target.state_dict(), filename + "_critic1_target")
        #torch.save(self.critic1_optimizer.state_dict(), filename + "_critic1_optimizer")

        torch.save(self.critic2.state_dict(), filename + "_critic2")
        #torch.save(self.critic2_target.state_dict(), filename + "_critic2_target")
        #torch.save(self.critic2_optimizer.state_dict(), filename + "_critic2_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        #torch.save(self.actor_target.state_dict(), filename + "_actor_target")
        #torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
         
        self.critic1.load_state_dict(torch.load(filename + "_critic1"))
        #self.critic1_target.load_state_dict(torch.load(filename + "_critic1_target"))
        #self.critic1_optimizer.load_state_dict(torch.load(filename + "_critic1_optimizer"))

        self.critic2.load_state_dict(torch.load(filename + "_critic2"))
        #self.critic2_target.load_state_dict(torch.load(filename + "_critic2_target"))
        #self.critic2_optimizer.load_state_dict(torch.load(filename + "_critic2_optimizer"))

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        #self.actor_target.load_state_dict(torch.load(filename + "_actor_target"))
        #self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))




