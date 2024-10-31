import torch
import torch.nn.functional as F
from torch.distributions import MixtureSameFamily, Categorical
from torch.distributions.utils import probs_to_logits


# below implementation of gumple softmax is taken from
# https://gist.github.com/yzh119/fd2146d2aeb329d067568a493b20172f
def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature, eps):
    y = logits + sample_gumbel(logits.size(), eps).to(logits.device)
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, eps):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature, eps)
    
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    return  (y_hard - y).detach() + y


def st_softmax(logits):
    """
    Straight-through version of softmax
    """
    probs = F.softmax(logits, dim=-1)
    one_hot = torch.zeros_like(logits)
    one_hot.scatter_(1, probs.argmax(dim=-1, keepdim=True), 1)
    return one_hot - probs.detach() + probs


def alpha_divergence(mean_x, sigma_x, mean_y, sigma_y, lmbda):
    mix_sigma_sum = (1 - lmbda) * sigma_x + lmbda * sigma_y
    mix_sigma_prod = sigma_x.pow(1 - lmbda) * sigma_y.pow(lmbda)

    term1_coef = (1 - lmbda) * lmbda / 2
    term1 = (mean_x - mean_y).pow(2) / mix_sigma_sum
    term2_coef = 1 / 2
    term2 = torch.log(mix_sigma_sum / mix_sigma_prod)

    return term1_coef * term1 + term2_coef * term2


def kl_divergence(mean_x, sigma_x, mean_y, sigma_y):
    return (sigma_x.pow(2) + (mean_x - mean_y).pow(2)) / \
        (2 * sigma_y.pow(2)) - 0.5 + torch.log(sigma_y / sigma_x)


class GumbelSoftmaxCategorical(Categorical):

    def __init__(self, probs=None, logits=None, temperature=0.1, hard=False,
                 validate_args=None, impl='default', eps=1e-20):
        super().__init__(probs=probs, logits=logits, validate_args=validate_args)
        if self.logits is None:
            self.logits = probs_to_logits(probs)
        self.temperature = temperature
        self.hard = hard
        self.impl = impl
        self.eps = eps

        if self.impl not in ['default', 'straight_through', 'torch']:
            raise ValueError('Invalid implementation: {}'.format(self.impl))

    def rsample(self, sample_shape=torch.Size()):
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        logits_2d = self.logits.reshape(-1, self._num_events)
        if self.impl == 'default':
            samples_2d = gumbel_softmax(logits_2d, self.temperature, self.eps)
        elif self.impl == 'straight_through':
            samples_2d = st_softmax(logits_2d)
        elif self.impl == 'torch':
            samples_2d = F.gumbel_softmax(logits_2d, tau=self.temperature, hard=self.hard)
        result_shape = sample_shape + self._batch_shape + self._event_shape + torch.Size([-1])
        return samples_2d.reshape(result_shape)


class MixtureModel(MixtureSameFamily):

    def mix_sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            sample_len = len(sample_shape)
            batch_len = len(self.batch_shape)
            gather_dim = sample_len + batch_len
            es = self.event_shape

            # mixture samples [n, B]
            mix_sample = self.mixture_distribution.sample(sample_shape)
            mix_shape = mix_sample.shape

        # component samples [n, B, k, E]
        comp_samples = self.component_distribution.rsample(sample_shape)

        # Gather along the k dimension
        mix_sample_r = mix_sample.reshape(
            mix_shape + torch.Size([1] * (len(es) + 1)))
        mix_sample_r = mix_sample_r.repeat(
            torch.Size([1] * len(mix_shape)) + torch.Size([1]) + es)

        samples = torch.gather(comp_samples, gather_dim, mix_sample_r)
        return samples.squeeze(gather_dim), mix_sample

    def rsample(self, sample_shape=torch.Size()):
        sample_len = len(sample_shape)
        batch_len = len(self.batch_shape)
        gather_dim = sample_len + batch_len

        # mixture samples [n, B, E, k]
        mix_sample = self.mixture_distribution.rsample(sample_shape)

        # component samples [n, B, E, k]
        comp_samples = self.component_distribution.rsample(sample_shape)

        # Mix along the k dimension
        samples = torch.mul(comp_samples, mix_sample).sum(dim=gather_dim)
        return samples

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            sample_len = len(sample_shape)
            batch_len = len(self.batch_shape)
            gather_dim = sample_len + batch_len
            es = self.event_shape

            # mixture samples [n, B]
            mix_sample = self.mixture_distribution.sample(sample_shape)
            mix_shape = mix_sample.shape

            # component samples [n, B, k, E]
            comp_samples = self.component_distribution.sample(sample_shape)

            # Gather along the k dimension
            mix_sample_r = mix_sample.reshape(
                mix_shape + torch.Size([1] * (len(es) + 1)))
            mix_sample_r = mix_sample_r.repeat(
                torch.Size([1] * len(mix_shape)) + torch.Size([1]) + es)

            samples = torch.gather(comp_samples, gather_dim, mix_sample_r)
            return samples.squeeze(gather_dim), mix_sample

    def log_prob_upper_bound(self, x, comp_i, eta=1.0):
        '''The name of this function is misleading. It calculates an estimate
        of the upper bound of the entropy of the mixture distribution.
        '''
        if self._validate_args:
            self._validate_sample(x)
        orig_shape = x.shape

        # select the comp_i-th component
        log_prob_x = self.component_distribution.log_prob(self._pad(x))
        log_prob_x = torch.gather(log_prob_x, -1, self._pad(comp_i))

        # calculate the log_prob of the mixture distribution
        log_prob_comp_i = self.mixture_distribution.log_prob(comp_i)
        log_prob_comp_i = self._pad(log_prob_comp_i)

        return (log_prob_x + eta * log_prob_comp_i).reshape(orig_shape)

    def log_prob_kl(self, x, comp_i, eta):
        '''The name of this function is misleading. It calculates an estimate
        of the upper bound of the entropy of the mixture distribution.
        '''
        return self.log_prob_pairwise(x, comp_i, eta, kl_divergence)

    def log_prob_alpha(self, x, comp_i, eta, lmbda):
        '''The name of this function is misleading. It calculates an estimate
        of the lower bound of the entropy of the mixture distribution.
        '''
        def _alpha_divergence(mean_x, sigma_x, mean_y, sigma_y):
            return alpha_divergence(mean_x, sigma_x, mean_y, sigma_y, lmbda)
        return self.log_prob_pairwise(x, comp_i, eta, _alpha_divergence)

    def log_prob_pairwise(self, x, comp_i, eta, divergence):
        if self._validate_args:
            self._validate_sample(x)
        orig_shape = x.shape
        x = self._pad(x)
        comp_i = self._pad(comp_i)

        # select the comp_i-th component
        log_prob_x = self.component_distribution.log_prob(x)
        log_prob_x = torch.gather(log_prob_x, -1, comp_i)

        # calculate the pairwise divergence term
        means = self.component_distribution.mean
        variances = self.component_distribution.variance
        if len(comp_i.shape) > len(self.component_distribution.mean.shape):
            # repeat means and variances to match the shape of comp_i at dim 0
            repeat_shape = [comp_i.shape[0]] + [1] * len(means.shape)
            means = means.unsqueeze(0).repeat(repeat_shape)
            variances = variances.unsqueeze(0).repeat(repeat_shape)
        mean_i = torch.gather(means, -1, comp_i)
        sigma_i = torch.gather(variances, -1, comp_i)
        mix_prob = self.mixture_distribution.probs
        results = torch.zeros_like(x)
        for comp_j in range(self.mixture_distribution.param_shape[-1]):
            mean_j = self.component_distribution.mean[:, :, comp_j].unsqueeze(-1)
            sigma_j = self.component_distribution.variance[:, :, comp_j].unsqueeze(-1)
            results += mix_prob[..., comp_j].unsqueeze(-1) * torch.exp(
                - divergence(mean_i, sigma_i, mean_j, sigma_j))

        return (log_prob_x + eta * torch.log(results)).reshape(orig_shape)
