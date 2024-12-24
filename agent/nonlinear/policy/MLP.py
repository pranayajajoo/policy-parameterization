# Import modules
import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, Beta
from agent.nonlinear.nn_utils import weights_init_
from utils.TruncatedNormal import TruncatedNormal
from utils.gumbel_softmax import GumbelSoftmaxCategorical, MixtureModel


# Global variables
EPSILON = 1e-6
FLOAT32_EPS = 10 * \
              np.finfo(np.float32).eps # differences of this size are 
                                       # representable up to ~ 15
LOG_STD_MAX = 2
LOG_STD_MIN = -5

# Class definitions
class DeterministicAction(nn.Module):
    """
    DeterministicAction implements a deterministic policy for epsilon-greedy exploration.
    The network is an MLP with two shared hidden layers and an output layer for actions.
    """
    def __init__(self, num_inputs, num_actions, hidden_dim, activation, action_space=None, init=None):
        """
        Constructor

        Parameters
        ----------
        num_inputs : int
            The number of elements in the state feature vector
        num_actions : int
            The dimensionality of the action vector
        hidden_dim : int
            The number of units in each hidden layer of the network
        activation : str
            The activation function to use, one of 'relu', 'tanh'
        action_space : gym.spaces.Space, optional
            The action space of the environment, by default None. This argument
            is used to ensure that the actions are within the correct scale.
        clip_stddev : float, optional
            The value at which the standard deviation is clipped in order to
            prevent numerical overflow, by default 1000. If <= 0, then
            no clipping is done.
        init : str
            The initialization scheme to use for the weights, one of
            'xavier_uniform', 'xavier_normal', 'uniform', 'normal',
            'orthogonal', by default None. If None, leaves the default
            PyTorch initialization.
        """
        super(DeterministicAction, self).__init__()

        self.num_actions = num_actions

        # Set up the layers
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, num_actions)

        # Initialize weights
        self.apply(lambda module: weights_init_(module, init, activation))

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

        if activation == "relu":
            self.act = F.relu
        elif activation == "tanh":
            self.act = torch.tanh
        else:
            raise ValueError(f"unknown activation function {activation}")

    def forward(self, state):
        """
        Performs the forward pass through the network, predicting the mean
        and the log standard deviation.

        Parameters
        ----------
        state : torch.Tensor of float
             The input state to predict the policy in

        Returns
        -------
        2-tuple of torch.Tensor of float
            Predicted action means for the input state.
        """
        x = self.act(self.linear1(state))
        x = self.act(self.linear2(x))

        mean = self.mean_linear(x)
        
        return mean

    def get_action(self, state, num_samples = 1):
        """
        Get deterministic action for a given state.
        Applies the tanh squashing function and rescales the action to the environment's action space.
        Parameters
        ----------
        state : torch.Tensor
            Input state to the network.
        Returns
        -------
        torch.Tensor
            Deterministic action within the environment's action space.
        """
        mean = self.forward(state)
        ### TODO check if we need action_scale and action_bias
        # mean = torch.tanh(mean) * self.action_scale + self.action_bias
        mean = torch.tanh(mean)
        return mean

    def eval_sample(self, state):
        with torch.no_grad():
            return self.get_action(state)

    def to(self, device):
        """
        Moves the network to a device

        Parameters
        ----------
        device : torch.device
            The device to move the network to

        Returns
        -------
        nn.Module
            The current network, moved to a new device
        """
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(DeterministicAction, self).to(device)

from scipy.optimize import minimize
class ScipyOptimizer:
    """
    Optimizer that, given a (Double) Critic, finds the action that maximizes Q(state, action).
    Because scipy can only minimize, we minimize the negative Q-value.
    """
    def __init__(self, critic, device):
        """
        Parameters
        ----------
        critic : nn.Module
            A DoubleQ critic that returns (Q1, Q2).
        device : torch.device
            Torch device (cpu/gpu).
        """
        self.critic = critic
        self.device = device
        self.last_best_actions = {}  
        # Dictionary: state_key -> np.array(action_dim,)
        # e.g. { (0.12, 0.98): np.array([0.55]), ... }

    def _state_to_key(self, state_np):
        """
        Convert state numpy array to a hashable key (tuple of floats).
        For multi-dimensional states, this will be a tuple with multiple elements.
        """
        return tuple(np.round(state_np, 5))  # or any rounding you prefer

    def objective_minimize(self, state_tensor, action):
        """
        Objective to minimize = - min(Q1, Q2).
        So effectively we are maximizing min(Q1, Q2).
        """
        action_tensor = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
        # shape: [1, action_dim]

        with torch.no_grad():
            q1, q2 = self.critic(state_tensor, action_tensor)
            q_min = torch.min(q1, q2).item()
        return -q_min  # Minimizing -q_min => maximizing q_min

    def minimize_single(self, state_np, num_guesses=2):
        """
        Minimizes the negative Q for a single (1D or multi-D) state.
        Uses two guesses: (1) the last best action (if any) for this state, (2) a new random guess.
        Returns the best action found.
        """
        # Convert state to PyTorch
        state_tensor = torch.tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        # shape: [1, state_dim]

        action_dim = self.critic.action_dim  # set in the agent
        bounds = [(-1, 1)] * action_dim       # Assume action in [-1,1]^action_dim

        # Build guesses
        guesses = []
        
        # 1) If we have a stored best action for this state, use it
        state_key = self._state_to_key(state_np)
        if state_key in self.last_best_actions:
            guesses.append(self.last_best_actions[state_key])
        else:
            # If no stored guess yet, add a random guess
            guesses.append(np.random.uniform(-1, 1, size=(action_dim,)))

        # 2) Always add one new random guess
        guesses.append(np.random.uniform(-1, 1, size=(action_dim,)))

        best_action = None
        best_val = float('inf')

        # Evaluate each guess with trust-constr
        for guess in guesses:
            res = minimize(
                lambda act: self.objective_minimize(state_tensor, act),
                guess,
                method='trust-constr',
                jac='2-point',
                bounds=bounds
            )
            if res.fun < best_val:
                best_val = res.fun
                best_action = res.x

        # Store the best action for next time
        self.last_best_actions[state_key] = best_action

        return best_action  # np.array(shape=(action_dim,))

class SquashedGaussian(nn.Module):
    """
    Class SquashedGaussian implements a policy following a squashed
    Gaussian distribution in each state, parameterized by an MLP.

    The MLP architecture is implemented
    as two shared hidden layers, followed by two separate output layers:
    one to predict the mean, and the other to predict the log standard
    deviation.

    For the the version that SAC used for the submission to ICML, see
    commit f66e4bf666da8c4142ff5acd33aed91dc25f4110.
    Basically there was a bug where the first and last layers
    used xavier_uniform while the second layer used kaiming_uniform
    """
    def __init__(self, num_inputs, num_actions, hidden_dim, activation,
                 action_space=None, clip_stddev=1000, init=None):
        """
        Constructor

        Parameters
        ----------
        num_inputs : int
            The number of elements in the state feature vector
        num_actions : int
            The dimensionality of the action vector
        hidden_dim : int
            The number of units in each hidden layer of the network
        activation : str
            The activation function to use, one of 'relu', 'tanh'
        action_space : gym.spaces.Space, optional
            The action space of the environment, by default None. This argument
            is used to ensure that the actions are within the correct scale.
        clip_stddev : float, optional
            The value at which the standard deviation is clipped in order to
            prevent numerical overflow, by default 1000. If <= 0, then
            no clipping is done.
        init : str
            The initialization scheme to use for the weights, one of
            'xavier_uniform', 'xavier_normal', 'uniform', 'normal',
            'orthogonal', by default None. If None, leaves the default
            PyTorch initialization.
        """
        super(SquashedGaussian, self).__init__()

        self.num_actions = num_actions

        # Determine standard deviation clipping
        self.clip_stddev = clip_stddev > 0
        self.clip_std_max = np.log(clip_stddev)
        self.clip_std_min = -np.log(clip_stddev)
        if clip_stddev == -1:
            self.clip_stddev = True
            self.clip_std_max = LOG_STD_MAX
            self.clip_std_min = LOG_STD_MIN

        # Set up the layers
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        # Initialize weights
        self.apply(lambda module: weights_init_(module, init, activation))

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

        if activation == "relu":
            self.act = F.relu
        elif activation == "tanh":
            self.act = torch.tanh
        else:
            raise ValueError(f"unknown activation function {activation}")

    def forward(self, state):
        """
        Performs the forward pass through the network, predicting the mean
        and the log standard deviation.

        Parameters
        ----------
        state : torch.Tensor of float
             The input state to predict the policy in

        Returns
        -------
        2-tuple of torch.Tensor of float
            The mean and log standard deviation of the Gaussian policy in the
            argument state
        """
        x = self.act(self.linear1(state))
        x = self.act(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)

        if self.clip_stddev:
            log_std = torch.clamp(log_std, min=self.clip_std_min, max=self.clip_std_max)
        return mean, log_std

    def sample(self, state, num_samples=1):
        """
        Samples the policy for an action in the argument state

        Parameters
        ----------
        state : torch.Tensor of float
             The input state to predict the policy in

        Returns
        -------
        torch.Tensor of float
            A sampled action
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        if self.num_actions > 1:
            normal = Independent(normal, 1)

        x_t = normal.sample((num_samples,))
        if num_samples == 1:
            x_t = x_t.squeeze(0)
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)

        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) +
                              EPSILON).sum(axis=-1).reshape(log_prob.shape)
        if self.num_actions > 1:
            log_prob = log_prob.unsqueeze(-1)

        if state.shape[0] == 1 or state.shape[0] == 32:
            if not hasattr(self, 'count'):
                self.count = 0
            if self.count % 5000 == 0:
                print(f'mean: {torch.tanh(mean)[0].cpu().detach().numpy().round(2).flatten()}, \n' \
                        f'std: {std[0].cpu().detach().numpy().round(2).flatten()}, \n')
            self.count += 1

        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean, x_t

    def sample_stat(self, state):
        with torch.no_grad():
            mean, log_std = self.forward(state)
            std = log_std.exp()
            return mean, std

    def rsample(self, state, num_samples=1):
        """
        Samples the policy for an action in the argument state using
        the reparameterization trick

        Parameters
        ----------
        state : torch.Tensor of float
             The input state to predict the policy in

        Returns
        -------
        torch.Tensor of float
            A sampled action
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        if self.num_actions > 1:
            normal = Independent(normal, 1)

        # For re-parameterization trick (mean + std * N(0,1))
        # rsample() implements the re-parameterization trick
        x_t = normal.rsample((num_samples,))
        if num_samples == 1:
            x_t = x_t.squeeze(0)

        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x_t)

        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) +
                              EPSILON).sum(axis=-1).reshape(log_prob.shape)
        if self.num_actions > 1:
            log_prob = log_prob.unsqueeze(-1)

        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean, x_t

    def log_prob_from_x_t(self, state_batch, x_t_batch):
        """
        Calculates the log probability of taking the action generated
        from x_t, where x_t is returned from sample or rsample. The
        log probability is returned for each action dimension separately.
        """
        mean, log_std = self.forward(state_batch)
        std = log_std.exp()
        normal = Normal(mean, std)

        if self.num_actions > 1:
            normal = Independent(normal, 1)

        y_t = torch.tanh(x_t_batch)
        log_prob = normal.log_prob(x_t_batch)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) +
                              EPSILON).sum(axis=-1).reshape(log_prob.shape)
        if self.num_actions > 1:
            log_prob = log_prob.unsqueeze(-1)

        print("logprob from x_t:", log_prob)

        return log_prob

    def log_prob(self, state_batch, action_batch):
        mean, log_std = self.forward(state_batch)
        std = log_std.exp()
        normal = Normal(mean, std)

        if self.num_actions > 1:
            normal = Independent(normal, 1)

        action_batch -= self.action_bias
        action_batch /= self.action_scale
        if (
            torch.max(action_batch) > 1.0 or
            torch.min(action_batch) < -1.0
        ):
            raise ValueError("cannot have action ∉ (-1, 1)")

        gaussian_actions = torch.atanh(
            torch.clamp(
                action_batch, -1.0 + EPSILON, 1.0 - EPSILON
            )
        )
        y_t = action_batch

        logprob = normal.log_prob(gaussian_actions)

        shift = torch.log(self.action_scale * (1 - y_t.pow(2)) +
                          EPSILON).sum(axis=-1).reshape(logprob.shape)
        logprob -= shift

        if self.num_actions > 1:
            logprob = logprob.unsqueeze(-1)

        return logprob

    def eval_sample(self, state):
        with torch.no_grad():
            mean, _ = self.forward(state)
            mean = torch.tanh(mean) * self.action_scale + self.action_bias
            return mean

    def to(self, device):
        """
        Moves the network to a device

        Parameters
        ----------
        device : torch.device
            The device to move the network to

        Returns
        -------
        nn.Module
            The current network, moved to a new device
        """
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(SquashedGaussian, self).to(device)


class Softmax(nn.Module):
    """
    Softmax implements a softmax policy in each state, parameterized
    using an MLP to predict logits.
    """
    def __init__(self, num_inputs, num_actions, hidden_dim, activation,
                 init=None):
        super(Softmax, self).__init__()

        self.num_actions = num_actions

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_actions)

        # self.apply(weights_init_)
        self.apply(lambda module: weights_init_(module, init, activation))

        if activation == "relu":
            self.act = F.relu
        elif activation == "tanh":
            self.act = torch.tanh
        else:
            raise ValueError(f"unknown activation {activation}")

    def forward(self, state):
        x = self.act(self.linear1(state))
        x = self.act(self.linear2(x))
        return self.linear3(x)

    def sample(self, state, num_samples=1):
        logits = self.forward(state)

        if len(logits.shape) != 1 and (len(logits.shape) != 2 and 1 not in
           logits.shape):
            shape = logits.shape
            raise ValueError(f"expected a vector of logits, got shape {shape}")

        probs = F.softmax(logits, dim=1)

        policy = torch.distributions.Categorical(probs)
        actions = policy.sample((num_samples,))

        log_prob = F.log_softmax(logits, dim=1)

        log_prob = torch.gather(log_prob, dim=1, index=actions)
        if num_samples == 1:
            actions = actions.squeeze(0)
            log_prob = log_prob.squeeze(0)

        actions = actions.unsqueeze(-1)
        log_prob = log_prob.unsqueeze(-1)

        # return actions.float(), log_prob, None
        return actions.int(), log_prob, logits.argmax(dim=-1)

    def all_log_prob(self, states):
        logits = self.forward(states)
        log_probs = F.log_softmax(logits, dim=1)

        return log_probs

    def log_prob(self, states, actions):
        """TODO: Docstring for log_prob.

        Parameters
        ----------
        states : TODO
        actions : TODO

        Returns
        -------
        TODO

        """
        logits = self.forward(states)
        log_probs = F.log_softmax(logits, dim=1)
        log_probs = torch.gather(log_probs, dim=1, index=actions.long())

        return log_probs

    def eval_sample(self, state):
        raise NotImplementedError

    def to(self, device):
        """
        Moves the network to a device

        Parameters
        ----------
        device : torch.device
            The device to move the network to

        Returns
        -------
        nn.Module
            The current network, moved to a new device
        """
        return super(Softmax, self).to(device)


class Gaussian(nn.Module):
    """
    Class Gaussian implements a policy following Gaussian distribution
    in each state, parameterized as an MLP. The predicted mean is scaled to be
    within `(action_min, action_max)`.

    The MLP architecture is implemented as two shared hidden layers,
    followed by two separate output layers: one to predict the mean, and the
    other to predict the log standard deviation.
    """
    def __init__(self, num_inputs, num_actions, hidden_dim, activation,
                 action_space, clip_stddev=1000, init=None, clip_actions=True):
        """
        Constructor

        Parameters
        ----------
        num_inputs : int
            The number of elements in the state feature vector
        num_actions : int
            The dimensionality of the action vector
        hidden_dim : int
            The number of units in each hidden layer of the network
        action_space : gym.spaces.Space
            The action space of the environment
        clip_stddev : float, optional
            The value at which the standard deviation is clipped in order to
            prevent numerical overflow, by default 1000. If <= 0, then
            no clipping is done.
        init : str
            The initialization scheme to use for the weights, one of
            'xavier_uniform', 'xavier_normal', 'uniform', 'normal',
            'orthogonal', by default None. If None, leaves the default
            PyTorch initialization.
        """
        super(Gaussian, self).__init__()

        self.num_actions = num_actions
        self.clip_action = clip_actions

        # Determine standard deviation clipping
        self.clip_stddev = clip_stddev > 0
        self.clip_std_threshold = np.log(clip_stddev)

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        # Initialize weights
        self.apply(lambda module: weights_init_(module, init, activation))

        # Action rescaling
        self.action_max = torch.FloatTensor(action_space.high)
        self.action_min = torch.FloatTensor(action_space.low)

        if activation == "relu":
            self.act = F.relu
        elif activation == "tanh":
            self.act = torch.tanh
        else:
            raise ValueError(f"unknown activation {activation}")

    def forward(self, state):
        """
        Performs the forward pass through the network, predicting the mean
        and the log standard deviation.

        Parameters
        ----------
        state : torch.Tensor of float
             The input state to predict the policy in

        Returns
        -------
        2-tuple of torch.Tensor of float
            The mean and log standard deviation of the Gaussian policy in the
            argument state
        """
        x = self.act(self.linear1(state))
        x = self.act(self.linear2(x))

        mean = torch.tanh(self.mean_linear(x))
        mean = ((mean + 1) / 2) * (self.action_max - self.action_min) + \
            self.action_min  # ∈ [action_min, action_max]
        log_std = self.log_std_linear(x)

        # Works better with std dev clipping to ±1000
        if self.clip_stddev:
            log_std = torch.clamp(log_std, min=-self.clip_std_threshold,
                                  max=self.clip_std_threshold)
        return mean, log_std

    def rsample(self, state, num_samples=1):
        """
        Samples the policy for an action in the argument state

        Parameters
        ----------
        state : torch.Tensor of float
             The input state to predict the policy in

        Returns
        -------
        torch.Tensor of float
            A sampled action
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        if self.num_actions > 1:
            normal = Independent(normal, 1)

        # For re-parameterization trick (mean + std * N(0,1))
        # rsample() implements the re-parameterization trick
        action = normal.rsample((num_samples,))

        if self.clip_action:
            action = torch.clamp(action, self.action_min, self.action_max)

        if num_samples == 1:
            action = action.squeeze(0)

        log_prob = normal.log_prob(action)
        if self.num_actions > 1:
            log_prob = log_prob.unsqueeze(-1)

        return action, log_prob, mean

    def sample(self, state, num_samples=1):
        """
        Samples the policy for an action in the argument state

        Parameters
        ----------
        state : torch.Tensor of float
             The input state to predict the policy in
        num_samples : int
            The number of actions to sample

        Returns
        -------
        torch.Tensor of float
            A sampled action
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        if self.num_actions > 1:
            normal = Independent(normal, 1)

        # Non-differentiable
        action = normal.sample((num_samples,))
        if self.clip_action:
            action = torch.clamp(action, self.action_min, self.action_max)

        if num_samples == 1:
            action = action.squeeze(0)

        log_prob = normal.log_prob(action)
        if self.num_actions > 1:
            log_prob = log_prob.unsqueeze(-1)

        # print(action.shape)

        return action, log_prob, mean

    def log_prob(self, states, actions, show=False):
        """
        Returns the log probability of taking actions in states. The
        log probability is returned for each action dimension
        separately, and should be added together to get the final
        log probability
        """
        mean, log_std = self.forward(states)
        std = log_std.exp()
        normal = Normal(mean, std)
        if self.num_actions > 1:
            normal = Independent(normal, 1)

        log_prob = normal.log_prob(actions)
        if self.num_actions > 1:
            log_prob = log_prob.unsqueeze(-1)

        if show:
            print(torch.cat([mean, std], axis=1)[0])

        return log_prob

    def eval_sample(self, state):
        with torch.no_grad():
            mean, _ = self.forward(state)
            return mean

    def to(self, device):
        """
        Moves the network to a device

        Parameters
        ----------
        device : torch.device
            The device to move the network to

        Returns
        -------
        nn.Module
            The current network, moved to a new device
        """
        self.action_max = self.action_max.to(device)
        self.action_min = self.action_min.to(device)
        return super(Gaussian, self).to(device)


class TruncatedGaussian(nn.Module):
    """
    Class TruncatedGaussian implements a policy following
    a truncated Gaussian distribution in each state.

    The MLP architecture is implemented
    as two shared hidden layers, followed by two separate output layers:
    one to predict the mean, and the other to predict the log standard
    deviation. The mean is scaled to be within (action_min, `action_max)`.
    """
    def __init__(self, num_inputs, num_actions, hidden_dim, activation,
                 action_space, clip_stddev=1000, init=None):
        """
        Constructor

        Parameters
        ----------
        num_inputs : int
            The number of elements in the state feature vector
        num_actions : int
            The dimensionality of the action vector
        hidden_dim : int
            The number of units in each hidden layer of the network
        action_space : gym.spaces.Space
            The action space of the environment
        clip_stddev : float, optional
            The value at which the standard deviation is clipped in order to
            prevent numerical overflow, by default 1000. If <= 0, then
            no clipping is done.
        init : str
            The initialization scheme to use for the weights, one of
            'xavier_uniform', 'xavier_normal', 'uniform', 'normal',
            'orthogonal', by default None. If None, leaves the default
            PyTorch initialization.
        """
        super(TruncatedGaussian, self).__init__()

        # Determine standard deviation clipping
        self.clip_stddev = clip_stddev > 0
        self.clip_std_threshold = np.log(clip_stddev)

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        # self.apply(weights_init_)
        self.apply(lambda module: weights_init_(module, init, activation))

        # action rescaling
        assert len(action_space.low.shape) == 1
        self.action_max = torch.FloatTensor(action_space.high)
        self.action_min = torch.FloatTensor(action_space.low)

        if activation == "relu":
            self.act = F.relu
        elif activation == "tanh":
            self.act = torch.tanh
        else:
            raise ValueError(f"unknown activation {activation}")

    def forward(self, state):
        """
        Performs the forward pass through the network, predicting the mean
        and the log standard deviation.

        Parameters
        ----------
        state : torch.Tensor of float
             The input state to predict the policy in

        Returns
        -------
        2-tuple of torch.Tensor of float
            The mean and log standard deviation of the Gaussian policy in the
            argument state
        """
        x = self.act(self.linear1(state))
        x = self.act(self.linear2(x))

        mean = torch.tanh(self.mean_linear(x))
        mean = ((mean + 1)/2) * (self.action_max - self.action_min) + \
            self.action_min  # ∈ [action_min, action_max]
        log_std = self.log_std_linear(x)

        # Works better with std dev clipping to ±1000
        if self.clip_stddev:
            log_std = torch.clamp(log_std, min=-self.clip_std_threshold,
                                  max=self.clip_std_threshold)
        return mean, log_std

    def rsample(self, state, num_samples=1):
        """
        Samples the policy for an action in the argument state

        Parameters
        ----------
        state : torch.Tensor of float
             The input state to predict the policy in

        Returns
        -------
        torch.Tensor of float
            A sampled action
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = TruncatedNormal(loc=mean, scale=std, a=self.action_min,
                                 b=self.action_max)

        # For re-parameterization trick (mean + std * N(0,1))
        # rsample() implements the re-parameterization trick
        x = normal.rsample((num_samples,))
        action = torch.clamp(x, self.action_min, self.action_max)
        if num_samples == 1:
            action = action.squeeze(0)

        log_prob = normal.log_prob(action)
        if num_samples == 1:
            log_prob = log_prob.sum(1, keepdim=True)
        else:
            log_prob = log_prob.sum(2, keepdim=True)

        return action, log_prob, mean

    def sample(self, state, num_samples=1):
        """
        Samples the policy for an action in the argument state

        Parameters
        ----------
        state : torch.Tensor of float
             The input state to predict the policy in
        num_samples : int
            The number of actions to sample

        Returns
        -------
        torch.Tensor of float
            A sampled action
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = TruncatedNormal(loc=mean, scale=std, a=self.action_min,
                                 b=self.action_max)

        # Non-differentiable
        x = normal.sample((num_samples,))
        action = torch.clamp(x, self.action_min, self.action_max)
        if num_samples == 1:
            action = action.squeeze(0)

        log_prob = normal.log_prob(action)
        if num_samples == 1:
            log_prob = log_prob.sum(1, keepdim=True)
        else:
            log_prob = log_prob.sum(2, keepdim=True)

        return action, log_prob, mean

    def log_prob(self, states, actions, show=False):
        """
        Returns the log probability of taking actions in states. The
        log probability is returned for each action dimension
        separately, and should be added together to get the final
        log probability
        """
        mean, log_std = self.forward(states)
        std = log_std.exp()
        normal = TruncatedNormal(loc=mean, scale=std, a=self.action_min,
                                 b=self.action_max)

        log_prob = normal.log_prob(actions)

        if show:
            print(torch.cat([mean, std], axis=1)[0])
            # print(log_prob.shape)

        return log_prob

    def eval_sample(self, state):
        with torch.no_grad():
            mean, _ = self.forward(state)
            return mean

    def to(self, device):
        """
        Moves the network to a device


        Parameters
        ----------
        device : torch.device
            The device to move the network to

        Returns
        -------
        nn.Module
            The current network, moved to a new device
        """
        self.action_max = self.action_max.to(device)
        self.action_min = self.action_min.to(device)
        return super(TruncatedGaussian, self).to(device)


class BetaPolicy(nn.Module):
    """
    Class BetaPolicy implements a policy following Beta distribution
    in each state, parameterized as an MLP.

    The MLP architecture is implemented as two shared hidden layers,
    followed by two separate output layers: one to predict the alpha parameter, and the other to predict the beta parameter.
    """
    def __init__(self, num_inputs, num_actions, hidden_dim, activation,
                 action_space, init=None, clip_min=1e-6, clip_max=1e3,
                 epsilon=1.0):
        """
        Constructor

        Parameters
        ----------
        num_inputs : int
            The number of elements in the state feature vector
        num_actions : int
            The dimensionality of the action vector
        hidden_dim : int
            The number of units in each hidden layer of the network
        action_space : gym.spaces.Space
            The action space of the environment
        clip_min : float, optional
            The lower value at which the parameters are clipped to prevent numerical overflow,
            by default 1. If <= 0, then no clipping is done.
        clip_max : float, optional
            The upper value at which the parameters are clipped to prevent numerical overflow,
            by default 1. If <= 0, then no clipping is done.
        init : str
            The initialization scheme to use for the weights, one of
            'xavier_uniform', 'xavier_normal', 'uniform', 'normal',
            'orthogonal', by default None. If None, leaves the default
            PyTorch initialization.
        epsilon : float
            A small positive number added to the final ReLU output for both the alpha network
            and the beta network, ensuring that estimates of these parameters are positive
        """
        super(BetaPolicy, self).__init__()

        self.num_actions = num_actions

        # Determine alpha and beta clipping
        self.clip = clip_min > 0 and clip_max > 0
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.epsilon = epsilon

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.alpha_linear = nn.Linear(hidden_dim, num_actions)
        self.beta_linear = nn.Linear(hidden_dim, num_actions)

        # Initialize weights
        self.apply(lambda module: weights_init_(module, init, activation))

        # Action rescaling
        self.action_max = torch.FloatTensor(action_space.high) 
        self.action_min = torch.FloatTensor(action_space.low)
        self.action_scale = self.action_max - self.action_min
        self.action_bias = self.action_min

        # Activation function
        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "tanh":
            self.act = nn.Tanh()
        else:
            raise ValueError(f"unknown activation {activation}")

    def forward(self, state):
        """
        Perform a forward pass through the network, predicting the
        alpha and beta parameters.

        Parameters
        ----------
        state : torch.Tensor of float
            The input state to predict the policy in

        Returns
        -------
        2-tuple of torch.Tensor of float
            The alph and beta parameters of the Beta policy in the
            argument state
        """
        x = self.act(self.linear1(state))
        x = self.act(self.linear2(x))

        # Ensure alpha and beta are both positive
        alpha = F.softplus(self.alpha_linear(x)) + self.epsilon
        beta = F.softplus(self.beta_linear(x)) + self.epsilon

        # Constrain alpha and beta to help avoid numerical issues
        if self.clip:
            alpha = torch.clamp(alpha, min=self.clip_min, max=self.clip_max)
            beta = torch.clamp(beta, min=self.clip_min, max=self.clip_max)

        return alpha, beta

    def rsample(self, state, num_samples=1):
        """
        Samples the policy for an action in the argument state using re-parametrization trick

        Parameters
        ----------
        state : torch.Tensor of float
            The input state to predict the policy in
        num_samples : int
            The number of actions to sample

        Returns
        -------
        action : torch.Tensor of float
            A sampled action
        log_prob : torch.Tensor of float
            The log probability of the sampled action
        alpha : torch.Tensor of float
            The predicted alpha parameter of the Beta distribution
        beta : torch.Tensor of float
            The predicted beta parameter of the Beta distribution
        """
        alpha, beta = self.forward(state)
        beta_dist = Beta(alpha, beta)

        if self.num_actions > 1:
            beta_dist = Independent(beta_dist, 1)

        action = beta_dist.rsample((num_samples,))
        transformed_action = self.action_scale * action + self.action_bias

        if num_samples == 1:
            action = action.squeeze(0)
            transformed_action = transformed_action.squeeze(0)

        log_prob = beta_dist.log_prob(self.clip_action(action)) - torch.log(self.action_scale).sum()
        if self.num_actions > 1:
            log_prob = log_prob.unsqueeze(-1)

        return transformed_action, log_prob

    def sample(self, state, num_samples=1):
        """
        Samples the policy for an action in the argument state

        Parameters
        ----------
        state : torch.Tensor of float
            The input state to predict the policy in
        num_samples : int
            The number of actions to sample

        Returns
        -------
        action : torch.Tensor of float
            A sampled action
        log_prob : torch.Tensor of float
            The log probability of the sampled action
        alpha : torch.Tensor of float
            The predicted alpha parameter of the Beta distribution
        beta : torch.Tensor of float
            The predicted beta parameter of the Beta distribution
        """
        alpha, beta = self.forward(state)
        beta_dist = Beta(alpha, beta)

        if self.num_actions > 1:
            beta_dist = Independent(beta_dist, 1)

        action = beta_dist.sample((num_samples,))
        transformed_action = self.action_scale * action + self.action_bias

        if num_samples == 1:
            action = action.squeeze(0)
            transformed_action = transformed_action.squeeze(0)

        log_prob = beta_dist.log_prob(self.clip_action(action)) - torch.log(self.action_scale).sum()
        if self.num_actions > 1:
            log_prob = log_prob.unsqueeze(-1)

        return transformed_action, log_prob, transformed_action

    def log_prob(self, states, actions, show=False):
        """
        Returns the log probability of taking actions in states. The
        log probability is returned for each action dimension
        separately, and should be added together to get the final
        log probability
        """
        alpha, beta = self.forward(states)
        beta_dist = Beta(alpha, beta)

        if self.num_actions > 1:
            beta_dist = Independent(beta_dist, 1)

        detransformed_actions = (actions - self.action_bias) / self.action_scale
        detransformed_actions = torch.clamp(detransformed_actions, 0, 1)

        log_prob = beta_dist.log_prob(self.clip_actions(detransformed_actions)) - torch.log(self.action_scale).sum()
        if self.num_actions > 1:
            log_prob = log_prob.unsqueeze(-1)

        if show:
            print(torch.cat([alpha, beta], axis=1)[0])

        return log_prob

    def clip_action(self, action):
        return torch.clamp(
            action,
            0 + FLOAT32_EPS,
            1 - FLOAT32_EPS
        )

    def clip_actions(self, actions):
        return torch.clamp(
            actions,
            0 + FLOAT32_EPS,
            1 - FLOAT32_EPS
        )

    def eval_sample(self, state):
        with torch.no_grad():
            alpha, beta = self.forward(state)
            action = alpha / (alpha + beta)
            return action

    def to(self, device):
        """
        Moves the network to a device

        Parameters
        ----------
        device : torch.device
            The device to move the network to

        Returns
        -------
        nn.Module
            The current network, moved to a new device
        """
        self.action_max = self.action_max.to(device)
        self.action_min = self.action_min.to(device)
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(BetaPolicy, self).to(device)


class BetaPolicyV2(BetaPolicy):

    def forward(self, state):
        """
        Perform a forward pass through the network, predicting the
        alpha and beta parameters.

        Parameters
        ----------
        state : torch.Tensor of float
            The input state to predict the policy in

        Returns
        -------
        2-tuple of torch.Tensor of float
            The alph and beta parameters of the Beta policy in the
            argument state
        """
        x = self.act(self.linear1(state))
        x = self.act(self.linear2(x))

        # Ensure alpha and beta are both positive
        alpha = F.sigmoid(self.alpha_linear(x))
        beta = F.sigmoid(self.beta_linear(x))

        alpha = 1 / alpha
        beta = 1 / beta

        # Constrain alpha and beta to help avoid numerical issues
        if self.clip:
            alpha = torch.clamp(alpha, min=self.clip_min, max=self.clip_max)
            beta = torch.clamp(beta, min=self.clip_min, max=self.clip_max)

        return alpha, beta


class GaussianMixture(nn.Module):
    """
    Class GaussianMixture implements a policy following a Gaussian mixture
    distribution in each state, parameterized as an MLP.

    The MLP architecture is implemented as two shared hidden layers,
    followed by two separate output layers: one to predict the mean, and the
    other to predict the log standard deviation.

    lmbda: float in [0, 1] or -1
        Parameter for the Chenoff alpha-divergence. If -1, then the
        alpha-divergence is not used.
    """
    def __init__(self, num_inputs, num_actions, hidden_dim, activation,
                 action_space=None, init=None, clip_stddev=1000,
                 clip_actions=True, num_components=2, temperature=0.1,
                 hard=False, share_std=False, lmbda=-1, eta=1.0,
                 repulsive_coef=0.0, impl='default', eps=1e-20):
        super(GaussianMixture, self).__init__()

        assert lmbda in [-1, -2, -3] or (lmbda >= 0 and lmbda <= 1)

        self.num_components = num_components
        self.num_actions = num_actions
        self.clip_action = clip_actions
        self.temperature = temperature
        self.hard = hard
        self.share_std = share_std
        self.lmbda = lmbda
        self.eta = eta
        self.repulsive_coef = repulsive_coef
        self.impl = impl
        self.eps = eps

        # Clipping strategy for numerical stability
        self.clip_stddev = clip_stddev > 0
        self.clip_std_threshold = np.log(clip_stddev)

        # Set up the layers
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, num_actions * num_components)
        if self.share_std:
            self.log_std_linear = nn.Linear(hidden_dim, num_actions)
        else:
            self.log_std_linear = nn.Linear(hidden_dim, num_actions * num_components)
        self.mixing_linear = nn.Linear(hidden_dim, num_actions * num_components)

        # Initialize weights
        self.apply(lambda module: weights_init_(module, init, activation))

        # Action rescaling
        self.action_max = torch.FloatTensor(action_space.high)
        self.action_min = torch.FloatTensor(action_space.low)
        self.action_scale = (self.action_max - self.action_min).view(-1, 1)

        if activation == "relu":
            self.act = F.relu
        elif activation == "tanh":
            self.act = torch.tanh
        else:
            raise ValueError(f"unknown activation function {activation}")

        if self.repulsive_coef > 0:
            self.repulsive_param = dict()

    def repulsive_loss(self):
        mean = self.repulsive_param['mean'].reshape(-1, self.num_components)

        # (B * A) x M x 1 -> (B * A) x M x M -> (B * A) x (M * M)
        pairwise_dist = torch.cdist(mean.unsqueeze(-1), 
                                    mean.unsqueeze(-1)).reshape(mean.shape[0], -1)
        # get median distance
        with torch.no_grad():
            median_dist = torch.median(pairwise_dist, dim=-1, keepdim=True)[0]

        # (B * A) x (M * M) -> (B * A) x 1
        loss = torch.exp(-pairwise_dist**2 / (2 * median_dist**2)).mean(dim=-1)
        return self.repulsive_coef * loss.mean()

    def forward(self, state, logit=False):
        x = self.act(self.linear1(state))
        x = self.act(self.linear2(x))
        
        mean_pre = torch.tanh(self.mean_linear(x))
        mean_pre = mean_pre.reshape(-1, self.num_actions, self.num_components)
        mean = ((mean_pre + 1)/2) * self.action_scale + self.action_min.view(-1, 1)  # ∈ [action_min, action_max]
        log_std = self.log_std_linear(x)
        if self.share_std:
            log_std = log_std.unsqueeze(-1).expand(-1, -1, self.num_components)
        else:
            log_std = log_std.reshape(-1, self.num_actions, self.num_components)
        mixing = self.mixing_linear(x).reshape(-1, self.num_actions, self.num_components)
        if not logit:
            mixing = torch.softmax(mixing, dim=-1)

        # Works better with std dev clipping to ±1000
        if self.clip_stddev:
            log_std = torch.clamp(log_std, min=-self.clip_std_threshold,
                                  max=self.clip_std_threshold)

        # Record the mean for repulsive loss
        if self.repulsive_coef > 0:
            self.repulsive_param['mean'] = mean_pre

        return mean, log_std, mixing

    def rsample(self, state, num_samples=1):
        mean, log_std, mixing = self.forward(state, logit=True)
        std = log_std.exp()

        # create a Gaussian mixture distribution
        mix = GumbelSoftmaxCategorical(logits=mixing, 
                                       temperature=self.temperature,
                                       hard=self.hard,
                                       impl=self.impl,
                                       eps=self.eps)
        comp = Normal(mean, std)
        gmm = MixtureModel(mix, comp)

        if self.num_actions > 1:
            gmm = Independent(gmm, 1)

        action = gmm.rsample((num_samples,))
        if self.clip_action:
            action = torch.clamp(action, self.action_min, self.action_max)

        if num_samples == 1:
            action = action.squeeze(0)

        log_prob = gmm.log_prob(action)
        if self.num_actions > 1:
            log_prob = log_prob.unsqueeze(-1)

        # CAUTION: to reduce computation, return the sampled action instead
        return action, log_prob, action

    def sample(self, state, num_samples=1):
        mean, log_std, mixing = self.forward(state)
        std = log_std.exp()

        # create a Gaussian mixture distribution
        mix = torch.distributions.Categorical(mixing)
        comp = Normal(mean, std)
        gmm = MixtureModel(mix, comp)

        if self.num_actions > 1:
            gmm = Independent(gmm, 1)

        # sample from the distribution, non-differentiable
        action, comp_idx = gmm.sample((num_samples,))
        # if max(action.shape) > 1:
        #     print(action.shape)
        #     exit()

        if self.clip_action:
            action = torch.clamp(action, self.action_min, self.action_max)

        if num_samples == 1:
            action = action.squeeze(0)
            comp_idx = comp_idx.squeeze(0)

        log_prob = gmm.log_prob(action)
        if self.lmbda == -1:
            log_prob_entropy = log_prob
        elif self.lmbda == -2:
            log_prob_entropy = gmm.log_prob_upper_bound(action, comp_idx, self.eta)
        elif self.lmbda == -3:
            log_prob_entropy = gmm.log_prob_kl(action, comp_idx, self.eta)
        else:
            log_prob_entropy = gmm.log_prob_alpha(action, comp_idx, self.eta, self.lmbda)
        if self.num_actions > 1:
            log_prob = log_prob.unsqueeze(-1)
            log_prob_entropy = log_prob_entropy.unsqueeze(-1)

        # DEBUG
        # if state.shape[0] == 1 or state.shape[0] == 100:
        #     print(f'mean: {mean[0].cpu().detach().numpy().round(2).flatten()}, \n' \
        #             f'std: {std[0].cpu().detach().numpy().round(2).flatten()}, \n' \
        #             f'mixing: {mixing[0].cpu().detach().numpy().round(2).flatten()} \n')

        return action, log_prob, action, log_prob_entropy

    def sample_stat(self, state):
        with torch.no_grad():
            mean, log_std, mixing = self.forward(state)
            std = log_std.exp()
            return mean, std, mixing

    def log_prob(self, states, actions, show=False):
        mean, log_std, mixing = self.forward(states)
        std = log_std.exp()

        # create a Gaussian mixture distribution
        mix = torch.distributions.Categorical(mixing)
        comp = Normal(mean, std)
        gmm = torch.distributions.MixtureSameFamily(mix, comp)

        if self.num_actions > 1:
            gmm = Independent(gmm, 1)

        # calculate the log probability of the actions
        log_prob = gmm.log_prob(actions)

        if show:
            print(torch.cat([mean, std], axis=1)[0])

        return log_prob

    def eval_sample(self, state):
        with torch.no_grad():
            mean, _, mixing = self.forward(state)
            mix = torch.distributions.Categorical(mixing)
            sampled_index = mix.sample((1,))
            sampled_index = sampled_index.squeeze(0)
            sampled_index = sampled_index.unsqueeze(-1)
            sampled_mean = torch.gather(mean, dim=2, index=sampled_index).squeeze(-1)
            return sampled_mean

    def to(self, device):
        self.action_max = self.action_max.to(device)
        self.action_min = self.action_min.to(device)
        self.action_scale = self.action_scale.to(device)
        return super(GaussianMixture, self).to(device)


class SquashedGaussianMixture(nn.Module):
    """
    Class SquashedGaussianMixture implements a policy following a squashed
    Gaussian mixture distribution in each state, parameterized as an MLP.

    The MLP architecture is implemented as two shared hidden layers,
    followed by three separate output layers: one to predict the mean, one to
    predict the log standard deviation, and the remaining one to predict 
    the logit of the mixing ratio.
    """
    def __init__(self, num_inputs, num_actions, hidden_dim, activation,
                 action_space=None, init=None, clip_stddev=1000,
                 num_components=2, temperature=0.1, hard=False,
                 share_std=False, repulsive_coef=0.0, impl='default',
                 eps=1e-20):
        super(SquashedGaussianMixture, self).__init__()

        self.num_components = num_components
        self.num_actions = num_actions
        self.temperature = temperature
        self.hard = hard
        self.share_std = share_std
        self.repulsive_coef = repulsive_coef
        self.impl = impl
        self.eps = eps

        # Clipping strategy for numerical stability
        self.clip_stddev = clip_stddev > 0
        self.clip_std_max = np.log(clip_stddev)
        self.clip_std_min = -np.log(clip_stddev)
        if clip_stddev == -1:
            self.clip_stddev = True
            self.clip_std_max = LOG_STD_MAX
            self.clip_std_min = LOG_STD_MIN

        # Set up the layers
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, num_actions * num_components)
        if self.share_std:
            self.log_std_linear = nn.Linear(hidden_dim, num_actions)
        else:
            self.log_std_linear = nn.Linear(hidden_dim, num_actions * num_components)
        self.mixing_linear = nn.Linear(hidden_dim, num_actions * num_components)

        # Initialize weights
        self.apply(lambda module: weights_init_(module, init, activation))

        # Action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

        if activation == "relu":
            self.act = F.relu
        elif activation == "tanh":
            self.act = torch.tanh
        else:
            raise ValueError(f"unknown activation function {activation}")

        if self.repulsive_coef > 0:
            self.repulsive_param = dict()

    def repulsive_loss(self):
        mean = self.repulsive_param['mean'].reshape(-1, self.num_components)
        mean = torch.tanh(mean)

        # (B * A) x M x 1 -> (B * A) x M x M -> (B * A) x (M * M)
        pairwise_dist = torch.cdist(mean.unsqueeze(-1), 
                                    mean.unsqueeze(-1)).reshape(mean.shape[0], -1)
        # get median distance
        with torch.no_grad():
            median_dist = torch.median(pairwise_dist, dim=-1, keepdim=True)[0]

        # (B * A) x (M * M) -> (B * A) x 1
        loss = torch.exp(-pairwise_dist**2 / (2 * median_dist**2)).mean(dim=-1)
        return self.repulsive_coef * loss.mean()

    def forward(self, state, logit=False):
        x = self.act(self.linear1(state))
        x = self.act(self.linear2(x))
        
        mean = self.mean_linear(x)
        mean = mean.reshape(-1, self.num_actions, self.num_components)
        log_std = self.log_std_linear(x)
        if self.share_std:
            log_std = log_std.unsqueeze(-1).expand(-1, -1, self.num_components)
        else:
            log_std = log_std.reshape(-1, self.num_actions, self.num_components)
        mixing = self.mixing_linear(x).reshape(-1, self.num_actions, self.num_components)
        if not logit:
            mixing = torch.softmax(mixing, dim=-1)

        # Works better with std dev clipping to ±1000
        if self.clip_stddev:
            log_std = torch.clamp(log_std, min=self.clip_std_min, max=self.clip_std_max)

        # Record the mean for repulsive loss
        if self.repulsive_coef > 0:
            self.repulsive_param['mean'] = mean

        return mean, log_std, mixing

    def mix_sample(self, state, num_samples=1):
        mean, log_std, mixing = self.forward(state)
        std = log_std.exp()

        # create a Gaussian mixture distribution
        mix = torch.distributions.Categorical(mixing)
        comp = Normal(mean, std)
        gmm = MixtureModel(mix, comp)

        if self.num_actions > 1:
            gmm = Independent(gmm, 1)
            x_t, comp_idx = gmm.base_dist.mix_sample((num_samples,))
        else:
            x_t, comp_idx = gmm.mix_sample((num_samples,))

        if num_samples == 1:
            x_t = x_t.squeeze(0)
            comp_idx = comp_idx.squeeze(0)
        y_t = torch.tanh(x_t)
        action = self.action_scale * y_t + self.action_bias

        log_prob = gmm.log_prob(x_t)
        shift = torch.log(self.action_scale * (1 - y_t.pow(2)) +
                          EPSILON).sum(axis=-1).reshape(log_prob.shape)
        log_prob = log_prob - shift
        if self.num_actions > 1:
            log_prob = log_prob.unsqueeze(-1)

        log_prob_mix = mix.log_prob(comp_idx)

        return action, log_prob, log_prob_mix

    def rsample(self, state, num_samples=1):
        mean, log_std, mixing = self.forward(state, logit=True)
        std = log_std.exp()

        # create a Gaussian mixture distribution
        mix = GumbelSoftmaxCategorical(logits=mixing,
                                       temperature=self.temperature,
                                       hard=self.hard,
                                       impl=self.impl,
                                       eps=self.eps)
        comp = Normal(mean, std)
        gmm = MixtureModel(mix, comp)

        if self.num_actions > 1:
            gmm = Independent(gmm, 1)

        x_t = gmm.rsample((num_samples,))
        if num_samples == 1:
            x_t = x_t.squeeze(0)
        y_t = torch.tanh(x_t)
        action = self.action_scale * y_t + self.action_bias

        log_prob = gmm.log_prob(x_t)
        shift = torch.log(self.action_scale * (1 - y_t.pow(2)) +
                          EPSILON).sum(axis=-1).reshape(log_prob.shape)
        log_prob = log_prob - shift
        if self.num_actions > 1:
            log_prob = log_prob.unsqueeze(-1)

        # DEBUG
        # if state.shape[0] == 1 or state.shape[0] == 32:
        #     if not hasattr(self, 'count'):
        #         self.count = 0
        #     if self.count % 1000 == 0:
        #         mixing = torch.softmax(mixing, dim=-1)
        #         print(f'mean: {torch.tanh(mean)[0].cpu().detach().numpy().round(2).flatten()}, \n' \
        #                 f'std: {std[0].cpu().detach().numpy().round(2).flatten()}, \n' \
        #                 f'mixing: {mixing[0].cpu().detach().numpy().round(2).flatten()} \n')
        #     self.count += 1

        # CAUTION: to reduce computation, return the sampled action instead
        return action, log_prob, action

    def sample(self, state, num_samples=1):
        mean, log_std, mixing = self.forward(state)
        std = log_std.exp()

        # create a Gaussian mixture distribution
        mix = torch.distributions.Categorical(mixing)
        comp = Normal(mean, std)
        gmm = torch.distributions.MixtureSameFamily(mix, comp)

        if self.num_actions > 1:
            gmm = Independent(gmm, 1)

        # sample from the distribution, non-differentiable
        x_t = gmm.sample((num_samples,))
        if num_samples == 1:
            x_t = x_t.squeeze(0)
        y_t = torch.tanh(x_t)
        action = self.action_scale * y_t + self.action_bias

        log_prob = gmm.log_prob(x_t)
        shift = torch.log(self.action_scale * (1 - y_t.pow(2)) +
                          EPSILON).sum(axis=-1).reshape(log_prob.shape)
        log_prob = log_prob - shift
        if self.num_actions > 1:
            log_prob = log_prob.unsqueeze(-1)

        # DEBUG
        if state.shape[0] == 1 or state.shape[0] == 32:
            if not hasattr(self, 'count'):
                self.count = 0
            if self.count % 5000 == 0:
                print(f'mean: {torch.tanh(mean)[0].cpu().detach().numpy().round(2).flatten()}, \n' \
                        f'std: {std[0].cpu().detach().numpy().round(2).flatten()}, \n' \
                        f'mixing: {mixing[0].cpu().detach().numpy().round(2).flatten()} \n')
            self.count += 1

        return action, log_prob, action

    def sample_stat(self, state):
        with torch.no_grad():
            mean, log_std, mixing = self.forward(state)
            std = log_std.exp()
            return mean, std, mixing

    def log_prob(self, states, actions):
        mean, log_std, mixing = self.forward(states)
        std = log_std.exp()

        # create a Gaussian mixture distribution
        mix = torch.distributions.Categorical(mixing)
        comp = Normal(mean, std)
        gmm = torch.distributions.MixtureSameFamily(mix, comp)

        if self.num_actions > 1:
            gmm = Independent(gmm, 1)
    
        actions -= self.action_bias
        actions /= self.action_scale
        if (
            torch.max(actions) > 1.0 or
            torch.min(actions) < -1.0
        ):
            raise ValueError("cannot have action ∉ (-1, 1)")

        gaussian_actions = torch.atanh(
            torch.clamp(actions, -1.0 + EPSILON, 1.0 - EPSILON)
        )
        y_t = actions

        # calculate the log probability of the actions
        log_prob = gmm.log_prob(gaussian_actions)
        shift = torch.log(self.action_scale * (1 - y_t.pow(2)) + 
                          EPSILON).sum(axis=-1).reshape(log_prob.shape)
        log_prob = log_prob - shift

        return log_prob

    def eval_sample(self, state):
        with torch.no_grad():
            mean, _, mixing = self.forward(state)
            mix = torch.distributions.Categorical(mixing)
            sampled_index = mix.sample((1,))
            sampled_index = sampled_index.squeeze(0)
            sampled_index = sampled_index.unsqueeze(-1)
            x_t = torch.gather(mean, dim=2, index=sampled_index).squeeze(-1)
            y_t = torch.tanh(x_t)
            action = self.action_scale * y_t + self.action_bias
            return action

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(SquashedGaussianMixture, self).to(device)


class BetaMixture(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, activation,
                 action_space=None, init=None, clip_min=1e-6, clip_max=1e3,
                 epsilon=1.0, num_components=2, temperature=0.1, hard=False,
                 impl='default', eps=1e-20):
        super(BetaMixture, self).__init__()

        self.num_components = num_components
        self.num_actions = num_actions
        self.clip = clip_min > 0 and clip_max > 0
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.epsilon = epsilon
        self.temperature = temperature
        self.hard = hard
        self.impl = impl
        self.eps = eps

        # Set up the layers
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.alpha_linear = nn.Linear(hidden_dim, num_actions * num_components)
        self.beta_linear = nn.Linear(hidden_dim, num_actions * num_components)
        self.mixing_linear = nn.Linear(hidden_dim, num_actions * num_components)

        # Initialize weights
        self.apply(lambda module: weights_init_(module, init, activation))

        # Action rescaling
        self.action_max = torch.FloatTensor(action_space.high)
        self.action_min = torch.FloatTensor(action_space.low)
        self.action_scale = self.action_max - self.action_min
        self.action_bias = self.action_min

        if activation == "relu":
            self.act = F.relu
        elif activation == "tanh":
            self.act = torch.tanh
        else:
            raise ValueError(f"unknown activation function {activation}")

    def forward(self, state, logit=False):
        x = self.act(self.linear1(state))
        x = self.act(self.linear2(x))

        alpha = F.softplus(self.alpha_linear(x)) + self.epsilon
        alpha = alpha.reshape(-1, self.num_actions, self.num_components)
        beta = F.softplus(self.beta_linear(x)) + self.epsilon
        beta = beta.reshape(-1, self.num_actions, self.num_components)
        mixing = self.mixing_linear(x).reshape(-1, self.num_actions, self.num_components)
        if not logit:
            mixing = torch.softmax(mixing, dim=-1)

        # Constrain alpha and beta to help avoid numerical issues
        if self.clip:
            alpha = torch.clamp(alpha, min=self.clip_min, max=self.clip_max)
            beta = torch.clamp(beta, min=self.clip_min, max=self.clip_max)

        return alpha, beta, mixing

    def rsample(self, state, num_samples=1):
        alpha, beta, mixing = self.forward(state, logit=True)

        # create a Beta mixture distribution
        mix = GumbelSoftmaxCategorical(logits=mixing,
                                       temperature=self.temperature,
                                       hard=self.hard,
                                       impl=self.impl,
                                       eps=self.eps)
        comp = Beta(alpha, beta)
        bmm = MixtureModel(mix, comp)

        if self.num_actions > 1:
            bmm = Independent(bmm, 1)

        action = bmm.rsample((num_samples,))
        transformed_action = self.action_scale * action + self.action_bias

        if num_samples == 1:
            action = action.squeeze(0)
            transformed_action = transformed_action.squeeze(0)

        log_prob = bmm.log_prob(self.clip_action(action)) - torch.log(self.action_scale).sum()
        if self.num_actions > 1:
            log_prob = log_prob.unsqueeze(-1)

        # CAUTION: to reduce computation, return the sampled action instead
        return transformed_action, log_prob, action

    def sample(self, state, num_samples=1):
        alpha, beta, mixing = self.forward(state)

        # create a Beta mixture distribution
        mix = torch.distributions.Categorical(mixing)
        comp = Beta(alpha, beta)
        bmm = torch.distributions.MixtureSameFamily(mix, comp)

        if self.num_actions > 1:
            bmm = Independent(bmm, 1)

        # sample from the distribution, non-differentiable
        action = bmm.sample((num_samples,))
        transformed_action = self.action_scale * action + self.action_bias

        if num_samples == 1:
            action = action.squeeze(0)
            transformed_action = transformed_action.squeeze(0)

        log_prob = bmm.log_prob(self.clip_action(action)) - torch.log(self.action_scale).sum()
        if self.num_actions > 1:
            log_prob = log_prob.unsqueeze(-1)

        return transformed_action, log_prob, transformed_action

    def sample_stat(self, state):
        with torch.no_grad():
            alpha, beta, mixing = self.forward(state)
            mean = alpha / (alpha + beta)
            mean = mean * self.action_scale + self.action_bias
            std = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
            std = std * self.action_scale
            return mean, std, mixing

    def log_prob(self, states, actions, show=False):
        alpha, beta, mixing = self.forward(states)

        # create a Beta mixture distribution
        mix = torch.distributions.Categorical(mixing)
        comp = Beta(alpha, beta)
        bmm = torch.distributions.MixtureSameFamily(mix, comp)

        if self.num_actions > 1:
            bmm = Independent(bmm, 1)

        detransformed_actions = (actions - self.action_bias) / self.action_scale
        detransformed_actions = torch.clamp(detransformed_actions, 0, 1)

        # calculate the log probability of the actions
        log_prob = bmm.log_prob(self.clip_action(detransformed_actions)) - torch.log(self.action_scale).sum()

        if self.num_actions > 1:
            log_prob = log_prob.unsqueeze(-1)

        if show:
            print(torch.cat([alpha, beta], axis=1)[0])

        return log_prob

    def clip_action(self, action):
        return torch.clamp(
            action,
            0 + FLOAT32_EPS,
            1 - FLOAT32_EPS
        )

    def eval_sample(self, state):
        with torch.no_grad():
            alpha, beta, mixing = self.forward(state)
            mix = torch.distributions.Categorical(mixing)
            sampled_index = mix.sample((1,))
            sampled_index = sampled_index.squeeze(0)
            sampled_index = sampled_index.unsqueeze(-1)
            sampled_alpha = torch.gather(alpha, dim=2, index=sampled_index).squeeze(-1)
            sampled_beta = torch.gather(beta, dim=2, index=sampled_index).squeeze(-1)
            action = sampled_alpha / (sampled_alpha + sampled_beta)
            action = action * self.action_scale + self.action_bias
            return action

    def to(self, device):
        self.action_max = self.action_max.to(device)
        self.action_min = self.action_min.to(device)
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(BetaMixture, self).to(device)
