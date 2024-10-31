# Import modules
from env.CGW import GridWorld
from gym.spaces import Box, Discrete
import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from agent.baseAgent import BaseAgent
from utils.experience_replay import TorchBuffer as ExperienceReplay
from agent.nonlinear.value_function.MLP import Q, DoubleQ
from agent.nonlinear.policy.MLP import SquashedGaussian, Gaussian, Softmax
import agent.nonlinear.nn_utils as nn_utils
import inspect


class GreedyAC(BaseAgent):
    def __init__(self, num_inputs, action_space, gamma, tau, alpha, policy,
                 target_update_interval, critic_lr, actor_lr_scale,
                 actor_hidden_dim, critic_hidden_dim, replay_capacity, seed,
                 batch_size, rho, num_samples, betas, env, cuda=False,
                 clip_stddev=1000, init=None, entropy_from_single_sample=True,
                 activation="relu", uniform_exploration_steps=0,
                 steps_before_learning=0, soft_q=False, double_q=False):
        super().__init__()

        self.batch = True
        self._t = 0
        self._uniform_exploration_steps = uniform_exploration_steps
        self._steps_before_learning = steps_before_learning

        self.soft_q = soft_q

        # Ensure batch size < replay capacity
        if batch_size > replay_capacity:
            raise ValueError("cannot have a batch larger than replay " +
                             "buffer capacity")

        # Set the seed for all random number generators, this includes
        # everything used by PyTorch, including setting the initial weights
        # of networks. PyTorch prefers seeds with many non-zero binary units
        self.torch_rng = torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)

        self.is_training = True
        self.entropy_from_single_sample = entropy_from_single_sample
        self.gamma = gamma
        self.tau = tau  # Polyak average
        self.alpha = alpha  # Entropy scale
        self.state_dims = num_inputs
        self.discrete_action = isinstance(action_space, Discrete)
        self.action_space = action_space
        self._obs_space = env.observation_space

        self.device = torch.device("cuda:0" if cuda and
                                   torch.cuda.is_available() else "cpu")

        if isinstance(action_space, Box):
            self.action_dims = len(action_space.high)

            # Keep a replay buffer
            self.replay = ExperienceReplay(replay_capacity, seed,
                                           env.observation_space.shape,
                                           action_space.shape[0], self.device)
        elif isinstance(action_space, Discrete):
            self.action_dims = 1
            # Keep a replay buffer
            self.replay = ExperienceReplay(replay_capacity, seed,
                                           env.observation_space.shape,
                                           1, self.device)
        self.batch_size = batch_size

        # Set the interval between timesteps when the target network should be
        # updated and keep a running total of update number
        self.target_update_interval = target_update_interval
        self.update_number = 0

        # For GreedyAC update
        self.rho = rho
        self.num_samples = num_samples

        # Create the critic Q function
        if isinstance(action_space, Box):
            action_shape = action_space.shape[0]
        elif isinstance(action_space, Discrete):
            action_shape = 1

        obs_space = env.observation_space
        if len(obs_space.shape) != 1:
            raise ValueError("GreedyAC only supports vector observations")
        self.double_q = double_q
        self._create_critic(
            obs_space,
            action_space,
            critic_hidden_dim,
            init,
            activation,
            critic_lr,
            betas,
        )

        self._create_policies(policy, num_inputs, action_space,
                              actor_hidden_dim, clip_stddev, init, activation)

        actor_lr = actor_lr_scale * critic_lr
        self.policy_optim = Adam(self.policy.parameters(), lr=actor_lr,
                                 betas=betas)
        self.sampler_optim = Adam(self.sampler.parameters(), lr=actor_lr,
                                  betas=betas)
        nn_utils.hard_update(self.sampler, self.policy)

        self.is_training = True

        source = inspect.getsource(inspect.getmodule(inspect.currentframe()))
        self.info["source"] = source

        self._env = env
        if isinstance(env, GridWorld):
            self.info["entropy"] = []

    def update(self, state, action, reward, next_state, done_mask):
        # Adjust action shape to ensure it fits in replay buffer properly
        if self.discrete_action:
            action = np.array([action])

        # Keep transition in replay buffer
        self.replay.push(state, action, reward, next_state, done_mask)

        if self._t < self._steps_before_learning:
            return

        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, \
            mask_batch = self.replay.sample(batch_size=self.batch_size)

        if state_batch is None:
            # Too few samples in the buffer to sample
            return

        self._update_critic(state_batch, action_batch, reward_batch,
                            next_state_batch, mask_batch)

        # # When updating Q functions, we don't want to backprop through the
        # # policy and target network parameters
        # with torch.no_grad():
        #     next_state_action, next_state_log_pi = self.policy.sample(
        #         next_state_batch
        #     )[:2]

        #     if len(next_state_log_pi.shape) == 1:
        #         next_state_log_pi = next_state_log_pi.unsqueeze(-1)

        #     next_q = self.critic_target(next_state_batch, next_state_action)

        #     if self.soft_q:
        #         next_q -= self.alpha * next_state_log_pi

        #     target_q_value = reward_batch + mask_batch * self.gamma * next_q

        # q_value = self.critic(state_batch, action_batch)

        # # Calculate the loss on the critic
        # # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        # q_loss = F.mse_loss(target_q_value, q_value)

        # # Update the critic
        # self.critic_optim.zero_grad()
        # q_loss.backward()
        # self.critic_optim.step()

        # # Update target networks
        # self.update_number += 1
        # if self.update_number % self.target_update_interval == 0:
        #     self.update_number = 0
        #     nn_utils.soft_update(self.critic_target, self.critic, self.tau)

        # Sample actions from the sampler to determine which to update
        # with
        action_batch = self.sampler.sample(state_batch, self.num_samples)[0]
        action_batch = action_batch.permute(1, 0, 2)
        action_batch = action_batch.reshape(self.batch_size * self.num_samples,
                                            self.action_dims)
        stacked_s_batch = state_batch.repeat_interleave(self.num_samples,
                                                        dim=0)

        # Get the values of the sampled actions and find the best
        # ϱ * num_samples actions
        with torch.no_grad():
            q_values = self.critic(stacked_s_batch, action_batch)

            if self.double_q:
                q_values = torch.min(q_values[0], q_values[1])

        q_values = q_values.reshape(self.batch_size, self.num_samples,
                                    1)
        sorted_q = torch.argsort(q_values, dim=1, descending=True)
        best_ind = sorted_q[:, :int(self.rho * self.num_samples)]
        best_ind = best_ind.repeat_interleave(self.action_dims, -1)

        action_batch = action_batch.reshape(self.batch_size, self.num_samples,
                                            self.action_dims)
        best_actions = torch.gather(action_batch, 1, best_ind)

        # Reshape samples for calculating the loss
        samples = int(self.rho * self.num_samples)
        stacked_s_batch = state_batch.repeat_interleave(samples, dim=0)
        best_actions = torch.reshape(best_actions, (-1, self.action_dims))

        # Actor loss
        # print(stacked_s_batch.shape, best_actions.shape)
        # print("Computing actor loss")
        policy_loss = self.policy.log_prob(stacked_s_batch, best_actions)
        policy_loss = -policy_loss.mean()

        # Update actor
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # Calculate sampler entropy
        # This is horrible! We calculate the log prob for a bunch of actions
        # then only use some of them in the regularization! This is fixed in
        # PyRL.
        stacked_s_batch = state_batch.repeat_interleave(self.num_samples,
                                                        dim=0)
        stacked_s_batch = stacked_s_batch.reshape(-1, self.state_dims)
        action_batch = action_batch.reshape(-1, self.action_dims)

        sampler_entropy = self.sampler.log_prob(stacked_s_batch, action_batch)
        with torch.no_grad():
            sampler_entropy *= sampler_entropy

        sampler_entropy = sampler_entropy.reshape(self.batch_size,
                                                  self.num_samples, 1)
        if self.entropy_from_single_sample:
            sampler_entropy = -sampler_entropy[:, 0, :]
        else:
            sampler_entropy = -sampler_entropy.mean(axis=1)

        # Calculate sampler loss
        stacked_s_batch = state_batch.repeat_interleave(samples, dim=0)
        # print("Computing sampler loss")
        sampler_loss = self.sampler.log_prob(stacked_s_batch, best_actions)
        sampler_loss = sampler_loss.reshape(self.batch_size, samples, 1)
        sampler_loss = sampler_loss.mean(axis=1)
        sampler_loss = sampler_loss + (sampler_entropy * self.alpha)
        sampler_loss = -sampler_loss.mean()

        # Update the sampler
        self.sampler_optim.zero_grad()
        sampler_loss.backward()
        self.sampler_optim.step()

    def _update_critic(self, state_batch, action_batch, reward_batch,
                       next_state_batch, mask_batch):
        """
        Update the critic(s) given a batch of transitions sampled from a replay
        buffer.
        """
        if self.double_q:
            self._update_double_critic(state_batch, action_batch, reward_batch,
                                       next_state_batch, mask_batch)

        else:
            self._update_single_critic(state_batch, action_batch, reward_batch,
                                       next_state_batch, mask_batch)

        # Increment the running total of updates and update the critic target
        # if needed
        self.update_number += 1
        if self.update_number % self.target_update_interval == 0:
            self.update_number = 0
            nn_utils.soft_update(self.critic_target, self.critic, self.tau)

    def _update_double_critic(self, state_batch, action_batch, reward_batch,
                              next_state_batch, mask_batch):
        # When updating Q functions, we don't want to backprop through the
        # policy and target network parameters
        with torch.no_grad():
            next_state_action, next_state_log_pi = self.policy.sample(
                next_state_batch
            )[:2]

            if len(next_state_log_pi.shape) == 1:
                next_state_log_pi = next_state_log_pi.unsqueeze(-1)

            next_q1, next_q2 = self.critic_target(next_state_batch,
                                                  next_state_action)

            # Double Q: target uses the minimum of the two computed action
            # values
            min_next_q = torch.min(next_q1, next_q2)

            if self.soft_q:
                min_next_q -= self.alpha * next_state_log_pi

            # Calculate the target for the action value function update
            q_target = reward_batch + mask_batch * self.gamma * min_next_q

        # Calculate the two Q values of each action in each respective state
        q1, q2 = self.critic(state_batch, action_batch)

        # Calculate the losses on each critic
        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        q1_loss = F.mse_loss(q1, q_target)

        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        q2_loss = F.mse_loss(q2, q_target)
        q_loss = q1_loss + q2_loss

        # Update the critic
        self.critic_optim.zero_grad()
        q_loss.backward()
        self.critic_optim.step()

    def _update_single_critic(self, state_batch, action_batch, reward_batch,
                              next_state_batch, mask_batch):
        # When updating Q functions, we don't want to backprop through the
        # policy and target network parameters
        with torch.no_grad():
            next_state_action, next_state_log_pi = self.policy.sample(
                next_state_batch
            )[:2]

            if len(next_state_log_pi.shape) == 1:
                next_state_log_pi = next_state_log_pi.unsqueeze(-1)

            next_q = self.critic_target(next_state_batch, next_state_action)

            if self.soft_q:
                next_q -= self.alpha * next_state_log_pi

            target_q_value = reward_batch + mask_batch * self.gamma * next_q

        q_value = self.critic(state_batch, action_batch)

        # Calculate the loss on the critic
        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        q_loss = F.mse_loss(target_q_value, q_value)

        # Update the critic
        self.critic_optim.zero_grad()
        q_loss.backward()
        self.critic_optim.step()

    def sample_action(self, state):
        """
        Samples an action from the agent

        Parameters
        ----------
        state : np.array
            The state feature vector

        Returns
        -------
        array_like of float
            The action to take
        """
        if self.is_training:
            self._t += 1
            if self._t - 1 < self._uniform_exploration_steps:
                return self._env.action_space.sample()

        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if self.is_training:
            action = self.policy.sample(state)[0]
        else:
            _, _, action = self.policy.sample(state)[:3]

        act = action.detach().cpu().numpy()[0]

        if not self.discrete_action:
            return act
        else:
            return int(act[0])

    def reset(self):
        pass

    def eval(self):
        """
        Sets the agent into offline evaluation mode, where the agent will not
        explore
        """
        self.is_training = False

    def train(self):
        """
        Sets the agent to online training mode, where the agent will explore
        """
        self.is_training = True

    def _create_critic(self, obs_space, action_space, critic_hidden_dim, init,
                       activation, critic_lr, betas):
        """
        Initializes the critic
        """
        num_inputs = obs_space.shape[0]

        if self.double_q:
            critic_type = DoubleQ
        else:
            critic_type = Q

        self.critic = critic_type(
            num_inputs,
            action_space.shape[0],
            critic_hidden_dim,
            init,
            activation,
        ).to(device=self.device)

        self.critic_target = critic_type(
            num_inputs,
            action_space.shape[0],
            critic_hidden_dim,
            init,
            activation,
        ).to(self.device)

        # Ensure critic and target critic share the same parameters at the
        # beginning of training
        nn_utils.hard_update(self.critic_target, self.critic)

        self.critic_optim = Adam(
            self.critic.parameters(),
            lr=critic_lr,
            betas=betas,
        )

    def _create_policies(self, policy, num_inputs, action_space,
                         actor_hidden_dim, clip_stddev, init, activation):
        self.policy_type = policy.lower()
        if self.policy_type == "gaussian":
            self.policy = Gaussian(num_inputs, action_space.shape[0],
                                   actor_hidden_dim, activation,
                                   action_space, clip_stddev,
                                   init).to(self.device)

            self.sampler = Gaussian(num_inputs, action_space.shape[0],
                                    actor_hidden_dim, activation,
                                    action_space, clip_stddev,
                                    init).to(self.device)

        elif self.policy_type == "squashedgaussian":
            self.policy = SquashedGaussian(num_inputs, action_space.shape[0],
                                           actor_hidden_dim, activation,
                                           action_space, clip_stddev,
                                           init).to(self.device)

            self.sampler = SquashedGaussian(num_inputs, action_space.shape[0],
                                            actor_hidden_dim, activation,
                                            action_space, clip_stddev,
                                            init).to(self.device)

        elif self.policy_type == "softmax":
            num_actions = action_space.n
            self.policy = Softmax(num_inputs, num_actions,
                                  actor_hidden_dim, activation,
                                  action_space, init).to(self.device)

            self.sampler = Softmax(num_inputs, num_actions,
                                   actor_hidden_dim, activation,
                                   action_space, init).to(self.device)

        else:
            raise NotImplementedError

    def get_parameters(self):
        pass

    def save_model(self, env_name, suffix="", actor_path=None,
                   critic_path=None):
        """
        Saves the models so that after training, they can be used.

        Parameters
        ----------
        env_name : str
            The name of the environment that was used to train the models
        suffix : str, optional
            The suffix to the filename, by default ""
        actor_path : str, optional
            The path to the file to save the actor network as, by default None
        critic_path : str, optional
            The path to the file to save the critic network as, by default None
        """
        pass

    def load_model(self, actor_path, critic_path):
        """
        Loads in a pre-trained actor and a pre-trained critic to resume
        training.

        Parameters
        ----------
        actor_path : str
            The path to the file which contains the actor
        critic_path : str
            The path to the file which contains the critic
        """
        pass
