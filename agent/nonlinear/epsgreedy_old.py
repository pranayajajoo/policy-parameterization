# eps_greedy_agent.py

import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
from agent.baseAgent import BaseAgent
from utils.experience_replay import TorchBuffer as ExperienceReplay
import agent.nonlinear.nn_utils as nn_utils

class EpsGreedyAgent(BaseAgent):
    def __init__(
        self,
        env,
        baseline_actions,     # not used
        reparameterized,      # not used
        clip_actions,         # not used
        policy,               # not used
        target_update_interval, # not used
        uniform_exploration_steps, # not used
        steps_before_learning, # not used
        gamma=0.99,
        tau=0.005,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.1,
        batch_size=64,
        replay_capacity=1e6,
        critic_lr=3e-4,
        actor_lr_scale = 1,
        actor_hidden_dim=256,
        critic_hidden_dim=256,
        seed=42,
        device=None,
        init=None,
        activation = "relu",
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
        self.device = torch.device(
            "cuda:0" if cuda and torch.cuda.is_available() else "cpu"
        )
        self.replay_buffer = ExperienceReplay(
            capacity=int(replay_capacity),
            seed=seed,
            state_size=env.observation_space.shape,
            action_size=env.action_space.shape[0],
            device=self.device
        )
        actor_lr = actor_lr_scale * critic_lr
        self._steps_before_learning = steps_before_learning

        # Actor/Critic init
        self.actor = self._init_actor(actor_hidden_dim).to(self.device)
        self.critic = self._init_critic(critic_hidden_dim).to(self.device)
        self.target_critic = self._init_critic(critic_hidden_dim).to(self.device)
        nn_utils.hard_update(self.target_critic, self.critic)

        # Optimizers
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)

        # Keep track of best actions from last iteration
        # Key: state index (or a unique ID), Value: best action as a np.array
        self.last_best_actions = {}

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
            self.init
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
        Epsilon-greedy action selection for interaction with the environment.
        """
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if self._is_training:
            self._t += 1
        if self._is_training and np.random.rand() < self.epsilon:
            # Random exploration
            action = torch.Tensor(self.env.action_space.sample())
        else:
            # Exploit
            action = self.actor.get_action(state_tensor)
        return action.detach().cpu().numpy()

    def update(self, state, action, reward, next_state, done):
        """
        Called each step to store experience and (if enough data) update networks.
        """
        action_array = np.array([action]) if np.isscalar(action) else action
        self.replay_buffer.push(state, action_array, reward, next_state, done)

        if self._t < self._steps_before_learning:
            return

        # Sample a batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        # Update critic & actor
        self.update_critic(states, actions, rewards, next_states, dones)
        self.update_actor(states)
        self.soft_update_target()

        # Decay epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def update_critic(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            next_actions = self.actor.get_actions(next_states)
            q1_next, q2_next = self.target_critic(next_states, next_actions)
            q_target = rewards + (1 - dones) * self.gamma * torch.min(q1_next, q2_next)

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor(self, states):
        # Standard actor update: maximize Q
        predicted_actions = self.actor.get_actions(states)
        q1, q2 = self.critic(states, predicted_actions)
        actor_loss = -torch.min(q1, q2).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def soft_update_target(self):
        nn_utils.soft_update(self.target_critic, self.critic, self.tau)

    # -------------------------------------------------------------------------
    #  NEW / MODIFIED: Parallel "best-action" search with only two guesses
    # -------------------------------------------------------------------------
    def parallel_best_action(self, states, state_ids):
        """
        For each state in the batch:
          1) Use the last best action if we have it; else random
          2) Generate a fresh random guess (or some second guess).
        Evaluate both in parallel on the critic, pick the one with higher Q-value.

        Parameters
        ----------
        states : torch.Tensor, shape [batch_size, state_dim]
        state_ids : list/array of unique IDs for each state
                    (so we can store/retrieve last_best_actions)

        Returns
        -------
        best_actions : torch.Tensor, shape [batch_size, action_dim]
                       The best actions chosen in parallel
        """
        batch_size = states.shape[0]
        action_dim = self.env.action_space.shape[0]

        # Gather "last best" guess or random if we don't have one
        guess1 = []
        for sid in state_ids:
            if sid in self.last_best_actions:
                guess1.append(self.last_best_actions[sid])
            else:
                guess1.append(self.env.action_space.sample())
        guess1 = torch.FloatTensor(guess1)

        # Second guess: random (or something else you prefer)
        guess2 = []
        for _ in range(batch_size):
            guess2.append(self.env.action_space.sample())
        guess2 = torch.FloatTensor(guess2)

        # Combine into shape [2 * batch_size, action_dim]
        combined_guesses = torch.cat([guess1, guess2], dim=0)

        # Repeat states accordingly: [batch_size, state_dim] -> [2*batch_size, state_dim]
        repeated_states = torch.repeat_interleave(states, 2, dim=0)

        # Evaluate Q-values (DoubleQ -> two critics)
        q1, q2 = self.critic(repeated_states, combined_guesses)
        # Use conservative Q estimate
        q_min = torch.min(q1, q2).squeeze(-1)  # shape [2 * batch_size]

        # Reshape to [batch_size, 2]
        q_min = q_min.view(batch_size, 2)

        # For each row, pick the best of guess1 or guess2
        best_indices = q_min.argmax(dim=1)  # shape [batch_size]
        combined_guesses = combined_guesses.view(batch_size, 2, action_dim)

        # Gather the best actions
        best_actions = []
        for i in range(batch_size):
            best_actions.append(combined_guesses[i, best_indices[i], :])
        best_actions = torch.stack(best_actions, dim=0)  # [batch_size, action_dim]

        # Store these best actions for next iteration
        for i, sid in enumerate(state_ids):
            self.last_best_actions[sid] = best_actions[i].detach().cpu().numpy()

        return best_actions

    # -------------------------------------------------------------------------
    #  Example usage: you might call parallel_best_action(...)
    #  inside update(), or a custom method, or anywhere you want to
    #  find the "best action" for multiple states in one shot.
    # -------------------------------------------------------------------------

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
