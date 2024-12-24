# eps_greedy_agent.py

import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
from agent.baseAgent import BaseAgent
from utils.experience_replay import TorchBuffer as ExperienceReplay
import agent.nonlinear.nn_utils as nn_utils
from scipy.optimize import minimize

# Import the CriticOptimizer from the new file
from agent.nonlinear.policy.MLP import ScipyOptimizer as CriticOptimizer


class EpsGreedyAgent(BaseAgent):
    def __init__(
        self, 
        env,
        baseline_actions,
        reparameterized,
        clip_actions,
        policy,
        target_update_interval,
        uniform_exploration_steps,
        steps_before_learning,
        gamma=0.99, 
        tau=0.005, 
        epsilon=1.0, 
        epsilon_decay=0.995, 
        epsilon_min=0.1, 
        batch_size=64,
        replay_capacity=1e6, 
        critic_lr=3e-4, 
        actor_lr_scale=1, 
        actor_hidden_dim=256, 
        critic_hidden_dim=256, 
        seed=42, 
        device=None,
        init=None,
        activation="relu",
        cuda=False,
    ):
        super().__init__()
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.activation = activation
        self.init = init

        self.device = torch.device("cuda:0" if cuda and torch.cuda.is_available() else "cpu")

        self.replay_buffer = ExperienceReplay(
            capacity=int(replay_capacity),
            seed=seed,
            state_size=env.observation_space.shape,
            action_size=env.action_space.shape[0],
            device=self.device
        )
        actor_lr = actor_lr_scale * critic_lr
        self._steps_before_learning = steps_before_learning

        # Initialize networks
        self.actor = self._init_actor(actor_hidden_dim).to(self.device)
        self.critic = self._init_critic(critic_hidden_dim).to(self.device)
        self.target_critic = self._init_critic(critic_hidden_dim).to(self.device)
        nn_utils.hard_update(self.target_critic, self.critic)

        # Set an attribute for action_dim if needed (used by CriticOptimizer)
        self.critic.action_dim = env.action_space.shape[0]

        # Optimizers
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)

        # Our newly added parallel-optimizer for exploitation
        self.critic_optimizer_parallel = CriticOptimizer(self.critic, self.device)

        # Other parameters
        self.step_count = 0
        self._is_training = True
        self._t = 0

    def _init_actor(self, hidden_dim):
        from agent.nonlinear.policy.MLP import DeterministicAction
        return DeterministicAction(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            hidden_dim,
            self.activation,
            self.env.action_space,
            self.init,
        )

    def _init_critic(self, hidden_dim):
        from agent.nonlinear.value_function.MLP import DoubleQ
        return DoubleQ(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            hidden_dim,
            init=self.init,
            activation=self.activation,
        )

    def sample_action(self, state):
        """
        Epsilon-greedy action selection:
        With probability epsilon, pick random action;
        otherwise, pick the best action according to minimize_single.
        """
        if self._is_training:
            self._t += 1

        if self._is_training and np.random.rand() < self.epsilon:
            # Random action
            action = torch.tensor(self.env.action_space.sample(), dtype=torch.float32)
        else:
            # Exploit with Critic-based optimization
            best_act = self.critic_optimizer_parallel.minimize_single(
                state_np=state,
                num_guesses=2  # only 2 guesses
            )
            action = torch.tensor(best_act, dtype=torch.float32)

        return action.detach().cpu().numpy()

    def update(self, state, action, reward, next_state, done):
        """
        Add experience to replay buffer and update networks.
        """
        action_array = np.array([action]) if np.isscalar(action) else action
        self.replay_buffer.push(state, action_array, reward, next_state, done)

        if self._t < self._steps_before_learning:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Update critic
        self.update_critic(states, actions, rewards, next_states, dones)
        # Update actor
        self.update_actor(states)
        # Soft update target critic
        self.soft_update_target()

        # Optionally decay epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def soft_update_target(self):
        nn_utils.soft_update(self.target_critic, self.critic, self.tau)

    def update_critic(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            next_actions = self.actor.get_action(next_states)
            q1_next, q2_next = self.target_critic(next_states, next_actions)
            q_target = rewards + (1 - dones) * self.gamma * torch.min(q1_next, q2_next)

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor(self, states):
        actions = self.actor.get_action(states)
        q1, q2 = self.critic(states, actions)
        q_min = torch.min(q1, q2)
        actor_loss = -q_min.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def reset(self):
        pass

    def eval(self):
        self._is_training = False

    def train(self):
        self._is_training = True

    def save(self, filename):
        pass

    def load(self, filename):
        pass

    def get_parameters(self):
        pass
