"""
Policy heads for RL^2 agents.
"""

import torch as tc


class LinearPolicyHead(tc.nn.Module):
    """
    Policy head for a reinforcement learning agent.
    """
    def __init__(self, num_features, num_actions):
        super().__init__()
        self._num_features = num_features
        self._num_actions = num_actions
        self._linear = tc.nn.Linear(
            in_features=self._num_features,
            out_features=self._num_actions,
            bias=True)
        tc.nn.init.xavier_normal_(self._linear.weight)
        tc.nn.init.zeros_(self._linear.bias)

    def forward(self, features: tc.FloatTensor) -> tc.distributions.Categorical:
        """
        Computes a policy distribution from features and returns it.

        Args:
            features: a tc.FloatTensor of shape [B, ..., F].

        Returns:
            tc.distributions.Categorical over actions, with batch shape [B, ...]
        """
        logits = self._linear(features)
        dists = tc.distributions.Categorical(logits=logits)
        return dists


class GaussianPolicyHead(tc.nn.Module):
    """
    Diagonal Gaussian policy head with state-dependent mean and log_std.
    Produces an Independent Normal over R^A.
    """
    def __init__(self, num_features: int, action_dim: int, init_log_std: float = -0.5,
                 log_std_min: float = -5.0, log_std_max: float = 2.0):
        super().__init__()
        self._num_features = num_features
        self._action_dim = action_dim
        self._mean = tc.nn.Linear(num_features, action_dim, bias=True)
        self._log_std = tc.nn.Linear(num_features, action_dim, bias=True)
        tc.nn.init.xavier_normal_(self._mean.weight)
        tc.nn.init.zeros_(self._mean.bias)
        tc.nn.init.xavier_normal_(self._log_std.weight)
        with tc.no_grad():
            self._log_std.bias.fill_(init_log_std)
        self._log_std_min = log_std_min
        self._log_std_max = log_std_max

    def forward(self, features: tc.FloatTensor) -> tc.distributions.Independent:
        mean = self._mean(features)
        log_std = self._log_std(features).clamp(self._log_std_min, self._log_std_max)
        std = log_std.exp()
        base = tc.distributions.Normal(mean, std)
        dist = tc.distributions.Independent(base, 1)
        return dist