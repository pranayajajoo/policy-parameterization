# Import modules
import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
from agent.baseAgent import BaseAgent
import agent.nonlinear.nn_utils as nn_utils
from agent.nonlinear.policy.MLP import (SquashedGaussian, Gaussian, BetaPolicy,
                                        GaussianMixture, BetaPolicyV2,
                                        SquashedGaussianMixture, BetaMixture)
from agent.nonlinear.value_function.MLP import DoubleQ, Q
from utils.experience_replay import TorchBuffer as ExperienceReplay


class SAC(BaseAgent):
    """
    SAC implements the Soft Actor-Critic algorithm for continuous action spaces
    as found in the paper https://arxiv.org/pdf/1812.05905.pdf.
    """
    def __init__(
        self,
        gamma,
        tau,
        alpha,
        policy,
        target_update_interval,
        critic_lr,
        actor_lr_scale,
        alpha_lr,
        actor_hidden_dim,
        critic_hidden_dim,
        replay_capacity,
        seed,
        batch_size,
        betas,
        env,
        clip_actions,
        baseline_actions=-1,
        reparameterized=True,
        soft_q=True,
        double_q=True,
        num_samples=1,
        automatic_entropy_tuning=False,
        cuda=False,
        clip_stddev=1000,
        clip_min=1e-6,
        clip_max=1e6,
        epsilon=1.0,
        num_components=1,
        share_std=False,
        temperature=0.1,
        hard=False,
        impl='default',
        eps=1e-20,
        latent_dim=2,
        lmbda=-1,
        eta=1.0,
        repulsive_coef=0.0,
        init=None,
        activation="relu",
        uniform_exploration_steps=0,
        steps_before_learning=0,
        use_true_q=False,
        log_actions_every=10000000,
        n_actions_logged=1000,
        record_current_state=False,
        record_grad_norm=False,
        record_entropy=False,
        record_params=False,
        record_values=False,
        record_mixture_stat=False,
        record_eval_state=None,
        n_states_logged=1,
        state_path=None,
    ):
        """
        Constructor

        Parameters
        ----------
        gamma : float
            The discount factor
        tau : float
            The weight of the weighted average, which performs the soft update
            to the target critic network's parameters toward the critic
            network's parameters, that is: target_parameters =
            ((1 - τ) * target_parameters) + (τ * source_parameters)
        alpha : float
            The entropy regularization temperature. See equation (1) in paper.
        policy : str
            The type of policy, currently, only support "gaussian"
        target_update_interval : int
            The number of updates to perform before the target critic network
            is updated toward the critic network
        critic_lr : float
            The critic learning rate
        actor_lr : float
            The actor learning rate
        alpha_lr : float
            The learning rate for the entropy parameter, if using an automatic
            entropy tuning algorithm (see automatic_entropy_tuning) parameter
            below
        actor_hidden_dim : int
            The number of hidden units in the actor's neural network
        critic_hidden_dim : int
            The number of hidden units in the critic's neural network
        replay_capacity : int
            The number of transitions stored in the replay buffer
        seed : int
            The random seed so that random samples of batches are repeatable
        batch_size : int
            The number of elements in a batch for the batch update
        automatic_entropy_tuning : bool, optional
            Whether the agent should automatically tune its entropy
            hyperparmeter alpha, by default False
        cuda : bool, optional
            Whether or not cuda should be used for training, by default False.
            Note that if True, cuda is only utilized if available.
        clip_stddev : float, optional
            The value at which the standard deviation is clipped in order to
            prevent numerical overflow, by default 1000. If <= 0, then
            no clipping is done.
        clip_min : float, optional
            The minimum value of the alpha and beta, by default 1e-6. If
            <= 0, then no clipping is done.
        clip_max : float, optional
            The maximum value of the alpha and beta, by default 1e3. If
            <= 0, then no clipping is done.
        init : str
            The initialization scheme to use for the weights, one of
            'xavier_uniform', 'xavier_normal', 'uniform', 'normal',
            'orthogonal', by default None. If None, leaves the default
            PyTorch initialization.
        soft_q : bool
            Whether or not to learn soft Q functions, by default True. The
            original SAC uses soft Q functions since we learn an
            entropy-regularized policy. When learning an entropy regularized
            policy, guaranteed policy improvement (in the ideal case) only
            exists with respect to soft action values.
        reparameterized : bool
            Whether to use the reparameterization trick to learn the policy or
            to use the log-likelihood trick. The original SAC uses the
            reparameterization trick.
        double_q : bool
            Whether or not to use a double Q critic, by default True
        num_samples : int
            The number of samples to use to estimate the gradient when using a
            likelihood-based SAC (i.e. `reparameterized == False`), by default
            1.

        Raises
        ------
        ValueError
            If the batch size is larger than the replay buffer
        """
        super().__init__()
        self._t = 0
        self._env = env

        self.use_true_q = use_true_q
        self.log_actions_every = log_actions_every
        self.n_actions_logged = n_actions_logged
        self.record_current_state = record_current_state
        self.record_grad_norm = record_grad_norm
        self.record_entropy = record_entropy
        self.record_params = record_params
        self.record_values = record_values
        self.record_mixture_stat = record_mixture_stat
        self.n_states_logged = n_states_logged

        if record_eval_state is not None:
            self.record_eval_state = True
            self.eval_ep_num = record_eval_state.get("episode", 10)
            self.eval_ep_length = record_eval_state.get("episode_length", 1000)
            # NOTE: if not None, override the eval_ep_num and eval_ep_length
            self.eval_steps = record_eval_state.get("eval_steps", None)
            self.eval_state_ep_buffer = []
            self.eval_state_buffer = []
        else:
            self.record_eval_state = False

        self._uniform_exploration_steps = uniform_exploration_steps
        self._steps_before_learning = steps_before_learning

        # Ensure batch size < replay capacity
        if batch_size > replay_capacity:
            raise ValueError("cannot have a batch larger than replay " +
                             "buffer capacity")

        if reparameterized and num_samples != 1:
            raise ValueError

        action_space = env.action_space
        self._action_space = action_space
        obs_space = env.observation_space
        self._obs_space = obs_space
        if len(obs_space.shape) != 1:
            raise ValueError("SAC only supports vector observations")

        self._baseline_actions = baseline_actions

        # Set the seed for all random number generators, this includes
        # everything used by PyTorch, including setting the initial weights
        # of networks. PyTorch prefers seeds with many non-zero binary units
        self._torch_rng = torch.manual_seed(seed)
        self._rng = np.random.default_rng(seed)

        # Random hypers and fields
        self._is_training = True  # Whether in training or evaluation mode
        self._gamma = gamma  # Discount factor
        self._tau = tau  # Polyak averaging constant for target networks
        self._reparameterized = reparameterized  # Whether to use reparam trick
        self._soft_q = soft_q  # Whether to use soft Q functions or nor
        self._double_q = double_q  # Whether or not to use a double Q critic
        if num_samples < 1:
            raise ValueError("cannot have num_samples < 1")
        self._num_samples = num_samples  # Sample for likelihood-based gradient

        self._device = torch.device("cuda:0" if cuda and
                                    torch.cuda.is_available() else "cpu")

        # Experience replay buffer
        self._batch_size = batch_size
        self._replay = ExperienceReplay(replay_capacity, seed, obs_space.shape,
                                        action_space.shape[0], self._device)

        # Set the interval between timesteps when the target network should be
        # updated and keep a running total of update number
        self._target_update_interval = target_update_interval
        self._update_number = 0

        # Automatic entropy tuning
        self._automatic_entropy_tuning = automatic_entropy_tuning
        self._alpha_lr = alpha_lr
        # assert not self._automatic_entropy_tuning
        if self._automatic_entropy_tuning and self._alpha_lr == 0:
            raise ValueError("should not use entropy lr == 0")

        # Set up the critic and target critic
        if not self.use_true_q:
            self._init_critic(
                obs_space,
                action_space,
                critic_hidden_dim,
                init,
                activation,
                critic_lr,
                betas,
            )
        else:        
            self.true_q = lambda _, action: self._env.env.reward(action)

        # Set up the policy
        self._policy_type = policy.lower()
        actor_lr = actor_lr_scale * critic_lr
        self._init_policy(
            obs_space,
            action_space,
            actor_hidden_dim,
            init,
            activation,
            actor_lr,
            betas,
            clip_stddev,
            clip_actions,
            clip_min,
            clip_max,
            num_components,
            latent_dim,
            epsilon,
            temperature,
            hard,
            share_std,
            lmbda,
            eta,
            repulsive_coef,
            impl,
            eps,
        )

        # Set up auto entropy tuning
        if self._automatic_entropy_tuning:
            self._target_entropy = -torch.prod(
                torch.Tensor(action_space.shape).to(self._device)
            ).item()
            self._log_alpha = torch.zeros(
                1,
                requires_grad=True,
                device=self._device,
            )
            self._alpha = self._log_alpha.exp().detach()
            self._alpha_optim = Adam([self._log_alpha], lr=self._alpha_lr)
        else:
            self._alpha = alpha  # Entropy scale

        self.info = {
            "actions_sampled": [],
        }
        if self.record_grad_norm:
            self.info["grad_norm"] = []
        if self.record_entropy:
            # self.info["log_probs"] = []
            self.info["entropy"] = []
        if self.record_params:
            self.info["params"] = []
        if self.record_values:
            self.info["states"] = []
            self.info["sampled_actions"] = []
            self.info["sampled_values"] = []
            self.info["uniform_actions"] = []
            self.info["uniform_values"] = []
            self.info["action_params"] = []

            if state_path is not None:
                if state_path == "":
                    self.record_state = None
                elif state_path == "mc_init":
                    self.record_state = torch.FloatTensor(
                        np.array([
                            [-0.6, 0.0],
                            [-0.55, 0.0],
                            [-0.5, 0.0],
                            [-0.45, 0.0],
                            [-0.4, 0.0],
                        ])
                    ).to(self._device)
                else:
                    self._load_target_states(state_path)

            # create a uniform action space
            actions = [np.linspace(self._action_space.low[i],
                                   self._action_space.high[i], self.n_actions_logged) \
                        for i in range(self._action_space.shape[0])]
            self.uniform_actions = torch.FloatTensor(
                np.array(np.meshgrid(*actions)).T.reshape(-1, self._action_space.shape[0])
            ).to(self._device)

        if self.record_mixture_stat:
            self.info["mixture_mixing_min"] = []
            self.info["mixture_mixing_max"] = []
            self.info["mixture_mixing_std"] = []
            self.info["mixture_mean_min"] = []
            self.info["mixture_mean_max"] = []
            self.info["mixture_mean_mean"] = []
            self.info["mixture_mean_std"] = []
            self.info["mixture_std_min"] = []
            self.info["mixture_std_max"] = []
            self.info["mixture_std_mean"] = []
            self.info["mixture_std_std"] = []

        if self.record_eval_state:
            self.info["eval_state"] = []

        if self._automatic_entropy_tuning:
            self.info["alpha"] = []

    def _load_target_states(self, state_path):
        """
        Load target states from a file

        Parameters
        ----------
        state_path : str
            The path to the file containing states to evaluate
        """
        if not os.path.exists(state_path):
            raise FileNotFoundError(f"state file {state_path} not found")
        self.target_states = torch.FloatTensor(
            np.load(state_path, allow_pickle=True)
        ).to(self._device)

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
        state = torch.FloatTensor(state).to(self._device).unsqueeze(0)

        if hasattr(self, "record_state") and self.record_state is None:
            # record the initial state
            self.record_state = state

        if self._is_training:
            self._t += 1
            if self._t - 1 < self._uniform_exploration_steps:
                return self._env.action_space.sample()

        if self._is_training:
            action = self._policy.sample(state)[0]
        else:
            if self.record_eval_state:
                action = self._policy.sample(state)[0]
                if self.eval_steps is None:
                    self.eval_state_ep_buffer.append(state.detach().cpu().numpy())
                    if len(self.eval_state_ep_buffer) == self.eval_ep_length:
                        self.eval_state_buffer.append(self.eval_state_ep_buffer)
                        self.eval_state_ep_buffer = []
                    if len(self.eval_state_buffer) == self.eval_ep_num:
                        self.info["eval_state"].append(self.eval_state_buffer)
                        self.eval_state_buffer = []
                        print('put eval state in info')
                else:
                    # the actual recording of the state is done in the
                    # environment
                    pass
            else:
                action = self._policy.eval_sample(state)

        return action.detach().cpu().numpy()[0]  # size (1, action_dims)

    def update(self, state, action, reward, next_state, done_mask):
        """
        Takes a single update step, which may be a number of offline
        batch updates

        Parameters
        ----------
        state : np.array or array_like of np.array
            The state feature vector
        action : np.array of float or array_like of np.array
            The action taken
        reward : float or array_like of float
            The reward seen by the agent after taking the action
        next_state : np.array or array_like of np.array
            The feature vector of the next state transitioned to after the
            agent took the argument action
        done_mask : bool or array_like of bool
            False if the agent reached the goal, True if the agent did not
            reach the goal yet the episode ended (e.g. Max number of steps
            reached)
        """
        # Keep transition in replay buffer
        self._replay.push(state, action, reward, next_state, done_mask)
        # if self._t==2000:
            # import ipdb;ipdb.set_trace()
        if self._t < self._steps_before_learning:
            return

        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, \
            mask_batch = self._replay.sample(batch_size=self._batch_size)

        if state_batch is None:
            return
        # import ipdb;ipdb.set_trace()
        if not self.use_true_q:
            self._update_critic(state_batch, action_batch, reward_batch,
                                next_state_batch, mask_batch)

        self._update_actor(state_batch, action_batch, reward_batch,
                           next_state_batch, mask_batch)

        # DEBUG: print the alpha value, delete later
        if self._t % 1000 == 0 and self._automatic_entropy_tuning:
            print(f"Step {self._t}, alpha: {self._alpha.item()}")

        # Log the actions sampled
        if self._t % self.log_actions_every == 0:
            print(f"Logging actions at step {self._t}")
            if self.record_current_state:
                state = torch.FloatTensor(state).to(self._device).unsqueeze(0)
                actions, log_probs = [], []
                for _ in range(self.n_actions_logged):
                    action, log_prob = self._policy.sample(state)[:2]
                    actions.append(action.item())
                    if self.record_entropy:
                        log_probs.append(log_prob.item())
                self.info["actions_sampled"].append(np.array(actions))
                if self.record_entropy:
                    # self.info["log_probs"].append(np.array(log_probs))
                    # estimate entropy
                    entropy = -np.mean(log_probs)
                    self.info["entropy"].append(entropy)
            if self.record_grad_norm:
                grad_norm = nn_utils.get_grad_norm(self._policy)
                self.info["grad_norm"].append(
                    grad_norm.detach().cpu().numpy()
                )
            if self.record_params:
                assert self._policy_type == "gaussianmixture"
                mean, log_std, mixing = self._policy(state)
                self.info["params"].append([
                    mean.detach().cpu().numpy(),
                    log_std.detach().cpu().numpy(),
                    mixing.detach().cpu().numpy()
                ])
            if self.record_values:
                # get #n_states_logged states from state_batch
                if hasattr(self, "target_states"):
                    state_batch = self.target_states
                if hasattr(self, "record_state"):
                    if self.record_state.shape[0] < self.n_states_logged:
                        self.record_state = torch.cat(
                            [self.record_state,
                             state_batch[self.record_state.shape[0]:]],
                            dim=0
                        )
                    state_batch = self.record_state
                states = state_batch[:self.n_states_logged]
                self.info["states"].append(states.detach().cpu().numpy())

                # sample actions
                actions = self._policy.sample(states, num_samples=int(
                    self.n_actions_logged * 1))[0].transpose(0, 1).reshape(-1, self._action_space.shape[0])
                # get Q values for each action
                states_repeated = states.unsqueeze(1).repeat(1, int(
                    self.n_actions_logged * 1), 1).reshape(-1, self._obs_space.shape[0])
                q_values = self._get_q(states_repeated, actions)
                self.info["sampled_actions"].append(actions.detach().cpu().numpy())
                self.info["sampled_values"].append(q_values.detach().cpu().numpy())

                # repeat the uniform actions for each state
                actions = self.uniform_actions.unsqueeze(0).repeat(
                    self.n_states_logged, 1, 1).reshape(-1, self._action_space.shape[0])
                states_repeated = states.unsqueeze(1).repeat(
                    1, self.uniform_actions.shape[0], 1).reshape(-1, self._obs_space.shape[0])
                q_values = self._get_q(states_repeated, actions)
                self.info["uniform_actions"].append(actions.detach().cpu().numpy())
                self.info["uniform_values"].append(q_values.detach().cpu().numpy())

                # get the parameters of the policy
                actions_params = self._policy.sample_stat(states)
                self.info["action_params"].append([
                    param.detach().cpu().numpy() for param in actions_params
                ])
            if self.record_mixture_stat:
                if "mixture" in self._policy_type:
                    # each has a shape of (#batch, #action_dim, #components)
                    mean, std, mixing = self._policy.sample_stat(state_batch)
                    mean, std, mixing = mean.detach().cpu().numpy(), std.detach().cpu().numpy(), mixing.detach().cpu().numpy()
                    self.info["mixture_mixing_min"].append(np.min(mixing, axis=2).mean())
                    self.info["mixture_mixing_max"].append(np.max(mixing, axis=2).mean())
                    self.info["mixture_mixing_std"].append(np.std(mixing, axis=2).mean())
                    self.info["mixture_mean_min"].append(np.min(mean, axis=2).mean())
                    self.info["mixture_mean_max"].append(np.max(mean, axis=2).mean())
                    self.info["mixture_mean_mean"].append(np.mean(mean, axis=2).mean())
                    self.info["mixture_mean_std"].append(np.std(mean, axis=2).mean())
                    self.info["mixture_std_min"].append(np.min(std, axis=2).mean())
                    self.info["mixture_std_max"].append(np.max(std, axis=2).mean())
                    self.info["mixture_std_mean"].append(np.mean(std, axis=2).mean())
                    self.info["mixture_std_std"].append(np.std(std, axis=2).mean())
                elif self._policy_type == "squashedgaussian":
                    mean, std = self._policy.sample_stat(state_batch)
                    mean, std = mean.detach().cpu().numpy(), std.detach().cpu().numpy()
                    self.info["mixture_mean_mean"].append(mean.mean())
                    self.info["mixture_std_mean"].append(std.mean())
                else:
                    raise NotImplementedError

            if self._automatic_entropy_tuning:
                self.info["alpha"].append(self._alpha.item())

    def _update_actor(self, state_batch, action_batch, reward_batch,
                      next_state_batch, mask_batch):
        """
        Update the actor given a batch of transitions sampled from a replay
        buffer.
        """
        # Calculate the actor loss
        if self._reparameterized == "mixed":
            assert "mixture" in self._policy_type
            baseline = 0
            if self._baseline_actions > 0:
                with torch.no_grad():
                    pi = self._policy.sample(
                        state_batch,
                        num_samples=self._baseline_actions,
                    )[0]
                    pi = pi.transpose(0, 1).reshape(
                        -1,
                        self._action_space.high.shape[0],
                    )
                    s_state_batch = state_batch.repeat_interleave(
                        self._baseline_actions,
                        dim=0,
                    )
                    q = self._get_q(s_state_batch, pi)
                    q = q.reshape(
                        self._batch_size,
                        self._baseline_actions,
                        -1,
                    )
                    baseline = q[:, 1:].mean(axis=1)

            pi, log_pi, log_pi_mix = self._policy.mix_sample(state_batch)

            if self._num_samples > 1:
                pi = pi.reshape(self._num_samples * self._batch_size, -1)
                state_batch = state_batch.repeat(self._num_samples, 1)

            # reshape works when num_samples is larger than 1, may have
            # unintended consequences in that case
            log_pi = log_pi.reshape(self._num_samples * self._batch_size, -1)
            q = self._get_q(state_batch, pi)

            with torch.no_grad():
                scale = self._alpha * log_pi - (q - baseline)

            mix_loss = log_pi_mix * scale
            comp_loss = self._alpha * log_pi - q
            policy_loss = (mix_loss + comp_loss).mean()

        elif self._reparameterized:
            # import ipdb;ipdb.set_trace()
            # Reparameterization trick [Official loss]
            if self._baseline_actions > 0:
                pi, log_pi = self._policy.rsample(
                    state_batch,
                    num_samples=self._baseline_actions+1,
                )[:2]
                pi = pi.transpose(0, 1).reshape(
                    -1,
                    self._action_space.high.shape[0],
                )
                s_state_batch = state_batch.repeat_interleave(
                    self._baseline_actions + 1,
                    dim=0,
                )
                q = self._get_q(s_state_batch, pi)
                q = q.reshape(self._batch_size, self._baseline_actions + 1, -1)

                # Don't backprop through the approximate state-value baseline
                baseline = q[:, 1:].mean(axis=1).squeeze().detach()

                log_pi = log_pi[0, :, 0]
                q = q[:, 0, 0]
                q -= baseline
            else:
                pi, log_pi = self._policy.rsample(state_batch)[:2]
                q = self._get_q(state_batch, pi)

            policy_loss = ((self._alpha * log_pi) - q).mean()
            # print("Policy loss: {}".format(policy_loss.mean()))
            if hasattr(self._policy, "repulsive_loss") and \
                    self._policy.repulsive_coef > 0:
                # Compute the repulsive loss
                repulsive_loss = self._policy.repulsive_loss()
                policy_loss += repulsive_loss

        else:
            # Log likelihood trick
            baseline = 0
            if self._baseline_actions > 0:
                with torch.no_grad():
                    pi = self._policy.sample(
                        state_batch,
                        num_samples=self._baseline_actions,
                    )[0]
                    pi = pi.transpose(0, 1).reshape(
                        -1,
                        self._action_space.high.shape[0],
                    )
                    s_state_batch = state_batch.repeat_interleave(
                        self._baseline_actions,
                        dim=0,
                    )
                    q = self._get_q(s_state_batch, pi)
                    q = q.reshape(
                        self._batch_size,
                        self._baseline_actions,
                        -1,
                    )
                    baseline = q[:, 1:].mean(axis=1)

            sample = self._policy.sample(
                state_batch,
                self._num_samples,
            )
            pi, log_pi = sample[:2]  # log_pi is differentiable

            if self._num_samples > 1:
                pi = pi.reshape(self._num_samples * self._batch_size, -1)
                state_batch = state_batch.repeat(self._num_samples, 1)

            with torch.no_grad():
                # Context manager ensures that we don't backprop through the q
                # function when minimizing the policy loss
                q = self._get_q(state_batch, pi)
                q -= baseline

            # Compute the policy loss
            log_pi = log_pi.reshape(self._num_samples * self._batch_size, -1)

            if self._policy_type == "gaussianmixture":
                log_pi_entropy = sample[3]
                log_pi_entropy = log_pi_entropy.reshape(
                    self._num_samples * self._batch_size, -1)
            else:
                log_pi_entropy = log_pi

            with torch.no_grad():
                scale = self._alpha * log_pi_entropy - q
            policy_loss = log_pi * scale
            policy_loss = policy_loss.mean()

            if hasattr(self._policy, "repulsive_loss") and \
                    self._policy.repulsive_coef > 0:
                # Compute the repulsive loss
                repulsive_loss = self._policy.repulsive_loss()
                policy_loss += repulsive_loss

        # Update the actor
        self._policy_optim.zero_grad()
        policy_loss.backward()
        self._policy_optim.step()

        # Tune the entropy if appropriate
        if self._automatic_entropy_tuning:
            # In [SAC: alg and applic], α = ln(β) is the dual variable, and
            # β is the entropy scale.
            alpha_loss = -(self._log_alpha *
                           (log_pi + self._target_entropy).detach()).mean()

            self._alpha_optim.zero_grad()
            alpha_loss.backward()
            self._alpha_optim.step()

            self._alpha = self._log_alpha.exp().detach()

    def reset(self):
        """
        Resets the agent between episodes
        """
        pass

    def eval(self):
        """
        Sets the agent into offline evaluation mode, where the agent will not
        explore
        """
        self._is_training = False

    def train(self):
        """
        Sets the agent to online training mode, where the agent will explore
        """
        self._is_training = True

    # Save model parameters
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

    # Load model parameters
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

    def get_parameters(self):
        """
        Gets all learned agent parameters such that training can be resumed.

        Gets all parameters of the agent such that, if given the
        hyperparameters of the agent, training is resumable from this exact
        point. This include the learned average reward, the learned entropy,
        and other such learned values if applicable. This does not only apply
        to the weights of the agent, but *all* values that have been learned
        or calculated during training such that, given these values, training
        can be resumed from this exact point.

        For example, in the LinearAC class, we must save not only the actor
        and critic weights, but also the accumulated eligibility traces.

        Returns
        -------
        dict of str to float, torch.Tensor
            The agent's weights
        """
        pass

    def _init_critic(self, obs_space, action_space, critic_hidden_dim, init,
                     activation, critic_lr, betas):
        """
        Initializes the critic
        """
        num_inputs = obs_space.shape[0]

        if self._double_q:
            critic_type = DoubleQ
        else:
            critic_type = Q

        self._critic = critic_type(
            num_inputs,
            action_space.shape[0],
            critic_hidden_dim,
            init,
            activation,
        ).to(device=self._device)

        self._critic_target = critic_type(
            num_inputs,
            action_space.shape[0],
            critic_hidden_dim,
            init,
            activation,
        ).to(self._device)

        # Ensure critic and target critic share the same parameters at the
        # beginning of training
        nn_utils.hard_update(self._critic_target, self._critic)

        self._critic_optim = Adam(
            self._critic.parameters(),
            lr=critic_lr,
            betas=betas,
        )

    def _init_policy(self, obs_space, action_space, actor_hidden_dim, init,
                     activation,  actor_lr, betas, clip_stddev, clip_actions,
                     clip_min, clip_max, num_components, latent_dim, epsilon,
                     temperature, hard, share_std, lmbda, eta, repulsive_coef,
                     impl, eps):
        """
        Initializes the policy
        """
        num_inputs = obs_space.shape[0]

        if self._policy_type == "squashedgaussian":
            self._policy = SquashedGaussian(num_inputs, action_space.shape[0],
                                            actor_hidden_dim, activation,
                                            action_space, clip_stddev,
                                            init).to(self._device)

        elif self._policy_type == "gaussian":
            self._policy = Gaussian(num_inputs, action_space.shape[0],
                                    actor_hidden_dim, activation, action_space,
                                    clip_stddev, init,
                                    clip_actions=clip_actions).to(self._device)

        elif self._policy_type == "beta":
            self._policy = BetaPolicy(num_inputs, action_space.shape[0],
                               actor_hidden_dim, activation, action_space,
                               init, clip_min, clip_max, epsilon).to(self._device)

        elif self._policy_type == "betav2":
            self._policy = BetaPolicyV2(num_inputs, action_space.shape[0],
                                 actor_hidden_dim, activation, action_space,
                                 init, clip_min, clip_max).to(self._device)

        elif self._policy_type == "gaussianmixture":
            self._policy = GaussianMixture(num_inputs, action_space.shape[0],
                                actor_hidden_dim, activation, action_space,
                                init, num_components=num_components,
                                temperature=temperature, hard=hard,
                                share_std=share_std, lmbda=lmbda, eta=eta,
                                repulsive_coef=repulsive_coef,
                                impl=impl, eps=eps).to(self._device)

        elif self._policy_type == "squashedgaussianmixture":
            self._policy = SquashedGaussianMixture(num_inputs, action_space.shape[0],
                                actor_hidden_dim, activation, action_space,
                                init, num_components=num_components,
                                temperature=temperature, hard=hard, clip_stddev=clip_stddev,
                                share_std=share_std,
                                repulsive_coef=repulsive_coef,
                                impl=impl, eps=eps).to(self._device)

        elif self._policy_type == "betamixture":
            self._policy = BetaMixture(num_inputs, action_space.shape[0],
                                actor_hidden_dim, activation, action_space,
                                init, clip_min=clip_min, clip_max=clip_max,
                                epsilon=epsilon, num_components=num_components,
                                temperature=temperature, hard=hard,
                                impl=impl, eps=eps).to(self._device)

        else:
            raise NotImplementedError(f"policy {self._policy_type} unknown")

        self._policy_optim = Adam(
            self._policy.parameters(),
            lr=actor_lr,
            betas=betas,
        )

    def _get_q(self, state_batch, action_batch):
        """
        Gets the Q values for `action_batch` actions in `state_batch` states
        from the critic, rather than the target critic.

        Parameters
        ----------
        state_batch : torch.Tensor
            The batch of states to calculate the action values in. Of the form
            (batch_size, state_dims).
        action_batch : torch.Tensor
            The batch of actions to calculate the action values of in each
            state. Of the form (batch_size, action_dims).
        """
        if self.use_true_q:
            q = self.true_q(state_batch, action_batch)
            return q

        if self._double_q:
            q1, q2 = self._critic(state_batch, action_batch)
            return torch.min(q1, q2)
        else:
            return self._critic(state_batch, action_batch)

    def _update_critic(self, state_batch, action_batch, reward_batch,
                       next_state_batch, mask_batch):
        """
        Update the critic(s) given a batch of transitions sampled from a replay
        buffer.
        """
        if self._double_q:
            self._update_double_critic(state_batch, action_batch, reward_batch,
                                       next_state_batch, mask_batch)

        else:
            self._update_single_critic(state_batch, action_batch, reward_batch,
                                       next_state_batch, mask_batch)

        # Increment the running total of updates and update the critic target
        # if needed
        self._update_number += 1
        if self._update_number % self._target_update_interval == 0:
            self._update_number = 0
            nn_utils.soft_update(self._critic_target, self._critic, self._tau)

    def _update_single_critic(self, state_batch, action_batch, reward_batch,
                              next_state_batch, mask_batch):
        """
        Update the critic using a batch of transitions when using a single Q
        critic.
        """
        if self._double_q:
            raise ValueError("cannot call _update_single_critic when using " +
                             "a double Q critic")

        # When updating Q functions, we don't want to backprop through the
        # policy and target network parameters
        with torch.no_grad():
            # Sample an action in the next state for the SARSA update
            if self._policy_type == "gaussianmixture":
                next_state_action, _, _, next_state_log_pi = \
                    self._policy.sample(next_state_batch, num_samples=1)[:4]
            else:
                next_state_action, next_state_log_pi = \
                    self._policy.sample(next_state_batch)[:2]

            if len(next_state_log_pi.shape) == 1:
                next_state_log_pi = next_state_log_pi.unsqueeze(-1)

            # Calculate the Q value of the next action in the next state
            q_next = self._critic_target(next_state_batch,
                                         next_state_action)

            if self._soft_q:
                q_next -= self._alpha * next_state_log_pi

            # Calculate the target for the SARSA update
            q_target = reward_batch + mask_batch * self._gamma * q_next

        # Calculate the Q value of each action in each respective state
        q = self._critic(state_batch, action_batch)

        # Calculate the loss between the target and estimate Q values
        q_loss = F.mse_loss(q, q_target)

        # Update the critic
        self._critic_optim.zero_grad()
        q_loss.backward()
        self._critic_optim.step()

    def _update_double_critic(self, state_batch, action_batch, reward_batch,
                              next_state_batch, mask_batch):
        """
        Update the critic using a batch of transitions when using a double Q
        critic.
        """
        if not self._double_q:
            raise ValueError("cannot call _update_double_critic when using " +
                             "a single Q critic")

        # When updating Q functions, we don't want to backprop through the
        # policy and target network parameters
        with torch.no_grad():
            # Sample an action in the next state for the SARSA update
            if self._policy_type == "gaussianmixture":
                next_state_action, _, _, next_state_log_pi = \
                    self._policy.sample(next_state_batch, num_samples=1)[:4]
            else:
                next_state_action, next_state_log_pi = \
                    self._policy.sample(next_state_batch)[:2]

            # Calculate the action values for the next state
            next_q1, next_q2 = self._critic_target(next_state_batch,
                                                   next_state_action)

            # Double Q: target uses the minimum of the two computed action
            # values
            min_next_q = torch.min(next_q1, next_q2)

            # If using soft action value functions, then adjust the target
            if self._soft_q:
                min_next_q -= self._alpha * next_state_log_pi

            # Calculate the target for the action value function update
            q_target = reward_batch + mask_batch * self._gamma * min_next_q

        # Calculate the two Q values of each action in each respective state
        q1, q2 = self._critic(state_batch, action_batch)

        # Calculate the losses on each critic
        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        q1_loss = F.mse_loss(q1, q_target)
        # print("Q mean: {}".format(q1.mean()))
        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        q2_loss = F.mse_loss(q2, q_target)
        q_loss = q1_loss + q2_loss

        # Update the critic
        self._critic_optim.zero_grad()
        q_loss.backward()
        self._critic_optim.step()
