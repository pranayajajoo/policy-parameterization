# Import modules
import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
from agent.baseAgent import BaseAgent
from utils.experience_replay import TorchBuffer as ExperienceReplay
import agent.nonlinear.nn_utils as nn_utils
from scipy.optimize import minimize


class EpsGreedyAgent(BaseAgent):
    def __init__(
        self, 
        env,
        baseline_actions,  # TODO: not using this currently
        reparameterized,  # TODO: not using this currently
        clip_actions,  # TODO: not using this currently
        policy,  # TODO: not using this currently
        target_update_interval, # TODO: not using this currently
        uniform_exploration_steps, # TODO: not using this currently
        steps_before_learning, # TODO: not using this currently
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
        cuda = False,
    ):
        super().__init__()

        # tracking q values every episode
        self.q1_mean_per_episode = []

        self.env = env
        self.gamma = gamma
        self.tau = tau
        # TODO print epsilon
        self.epsilon = epsilon
        print(f'self.epsilon: {self.epsilon}')
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
        self.max_action = float(self.env.action_space.high[0])


        # Initialize networks
        self.actor = self._init_actor(actor_hidden_dim).to(self.device)
        self.critic = self._init_critic(critic_hidden_dim).to(self.device)
        self.target_critic = self._init_critic(critic_hidden_dim).to(self.device)
        # Ensure critic and target critic share the same parameters at the
        # beginning of training 
        nn_utils.hard_update(self.target_critic, self.critic)

        # Optimizers
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)

        # Other parameters
        self.step_count = 0
        self._is_training = True
        self._t = 0

    def _init_actor(self, hidden_dim):
        """Initialize deterministic actor network."""
        from agent.nonlinear.policy.MLP import DeterministicAction as deterministic_action
        return deterministic_action(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            hidden_dim,
            self.activation,
            self.env.action_space,
            self.init,  # Weight initialization scheme
        )

    def _init_critic(self, hidden_dim):
        """Initialize DoubleQ critic network."""
        from agent.nonlinear.value_function.MLP import DoubleQ
        return DoubleQ(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            hidden_dim,
            init=self.init,
            activation=self.activation,  # Activation function
        )

    def sample_action(self, state):
        """Select action using epsilon-greedy policy."""
        
        ### TODO: check if we want to do some random exploration before beginning eps greedy
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if self._is_training:
            self._t += 1
        # if self._is_training and np.random.rand() < self.epsilon:
        #     action = torch.Tensor(self.env.action_space.sample())  # Random action
        # else:
            # action = torch.Tensor(self.actor.get_action(state = state_tensor))  # Exploit
            
        # TD3
        action_tensor = self.actor.get_action(state=state_tensor).detach().numpy()
        action = (
                    torch.Tensor(action_tensor) 
                    +
                    np.random.normal(0, 
                                     self.max_action * 0.1,
                                     size=self.env.action_space.shape  # Match the expected output shape
                                    )
                ).clip(-self.max_action, self.max_action)

        return action.detach().cpu().numpy()[0]  # size (1, action_dims)
    
    ### TODO
    def batch_sample_action(self, states):
        """Select action using epsilon-greedy policy."""
        
        ### TODO: check if we want to do some random exploration before beginning eps greedy
        # states should already be a tensor
        states_tensor = torch.FloatTensor(states).to(self.device).unsqueeze(0)
        # states_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if self._is_training:
            self._t += 1
        if self._is_training:
            batch_size = states_tensor.size(1)
            random_action = torch.FloatTensor([
            self.env.action_space.sample() for _ in range(batch_size)
            ]).to(self.device) # random action
            greedy_action = torch.Tensor(self.actor.get_action(state = states_tensor))
            random_idx = np.random.uniform(size=(batch_size,))
            action = (random_idx<self.epsilon)*random_action + (random_idx>=self.epsilon)*greedy_action

        return action.squeeze(0).detach().cpu()  # size (1, action_dims)


    def update(self, state, action, reward, next_state, done):
        """Add experience to replay buffer and update networks."""
        action_array = np.array([action]) if np.isscalar(action) else action
        self.replay_buffer.push(state, action_array, reward, next_state, done)

        ### TODO: ADD IF WE're DOING EXPLORATION BEFORE TRAINING STARTS
        # if self._t < self._steps_before_learning:
        #     return 

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)


        # import ipdb; ipdb.set_trace()

        if states is None:
            # print('reached here')
            return
        # if states is not None:
        #     import ipdb; ipdb.set_trace()

        # import ipdb; ipdb.set_trace()


        ### TODO: create update functions for actor and critic to monitor the # of pupdates for actor and critic

        # Update critic
        self.update_critic(states, actions, rewards, next_states, dones)

        # Update actor
        self.update_actor(states)

        # Soft update target critic
        self.soft_update_target()

        # TODO: CHeck if we need to decay epsilon for exploration
        # self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def soft_update_target(self):
        """
        Perform a soft update of the target critic networks.
        """
        nn_utils.soft_update(self.target_critic, self.critic, self.tau)

    # Jiamin's optimizer suggestion
    # def get_potential_actions(self, states, q, action_min=-1, action_max=1, num_starting_points=30, lr=0.01, num_gd_steps=100):
    #     uniform_actions = [np.linspace(action_min, action_max, num_starting_points)]
    #     actions = torch.FloatTensor(uniform_actions)
        
    #     # create an optimizer
    #     optim = SGD(actions.parameters(), lr=lr)
    #     for i in range(num_gd_steps):
    #         optim.zero_grad()
    #         loss = - q(s, actions)
    #         loss.backward()
    #         optim.step()
    #     return actions

    def update_critic(self, states, actions, rewards, next_states, dones):
        """
        Update the DoubleQ critic using the Bellman equation.
        """
        with torch.no_grad():
            # Compute next actions using the actor
            next_actions = self.actor.get_action(next_states)

            # TESTING CRITIC EXPLLORATION + EXPLOITATION
            # next_actions = self.batch_sample_action(next_states)

            # import ipdb;ipdb.set_trace()

            # Compute target Q-values using target critic
            q1_next, q2_next = self.target_critic(next_states, next_actions)

            # TODO check dim of rewards, dones, q1_next should be same
            q_target = rewards + (1 - dones) * self.gamma * torch.min(q1_next, q2_next)
            # print(f'rewards dims: {rewards.size()}')
            # print(f'dones dims: {dones.size()}')
            # print(f'q1_next dims: {q1_next.size()}')
            # print(f'q_target dims: {q_target.size()}')

        # Compute current Q-values using the critic
        q1, q2 = self.critic(states, actions)

        # storing q1 values for plotting and debugging
        q1_mean = q1.mean().item()
        self.q1_mean_per_episode.append(q1_mean)
        # print(f"[update_critic] Average q1: {q1.mean().item()}")

        # Compute critic loss
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        # Update critic networks
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
    
    def get_q1_means(self):
        return self.q1_mean_per_episode
    

    def update_actor(self, states):
        """
        Update the actor by maximizing the minimum Q-value from the DoubleQ critic.
        """
        # Compute actions predicted by the actor
        actions = self.actor.get_action(states)

        # Evaluate the Q-values of the predicted actions
        q1, q2 = self.critic(states, actions)
        q_min = torch.min(q1, q2)  # Conservative Q-value estimate
        # print(f"[update_actor] q_min: {q_min}")

        # Compute the actor loss (negative Q-value to maximize it)
        actor_loss = -q_min.mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


    def reset(self):
        """Reset between episodes."""
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

    def save(self, filename):
        """Save the agent's model."""
        # torch.save({
        #     "actor": self.actor.state_dict(),
        #     "critic": self.critic.state_dict(),
        #     "target_critic": self.target_critic.state_dict(),
        #     "actor_optimizer": self.actor_optimizer.state_dict(),
        #     "critic_optimizer": self.critic_optimizer.state_dict(),
        # }, filename)
        pass

    def load(self, filename):
        """Load the agent's model."""
        # checkpoint = torch.load(filename)
        # self.actor.load_state_dict(checkpoint["actor"])
        # self.critic.load_state_dict(checkpoint["critic"])
        # self.target_critic.load_state_dict(checkpoint["target_critic"])
        # self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        # self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
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