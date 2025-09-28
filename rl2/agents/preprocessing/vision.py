"""
Implements preprocessing for vision-based MDPs/POMDPs.
"""

import abc

import torch as tc

from rl2.agents.preprocessing.common import one_hot, Preprocessing


class VisionNet(abc.ABC, tc.nn.Module):
    """
    Vision network abstract class.
    """
    @property
    @abc.abstractmethod
    def output_dim(self) -> int:
        pass

    @abc.abstractmethod
    def forward(self, curr_obs: tc.FloatTensor) -> tc.FloatTensor:
        """
        Embeds visual observations into feature vectors.

        Args:
            curr_obs: tc.FloatTensor of shape [B, C, H, W]

        Returns:
            a tc.FloatTensor of shape [B, F]
        """
        pass


class ConvVisionNet(VisionNet):
    """
    A simple CNN encoder that maps images [B, C, H, W] -> [B, F].
    - Uses several Conv-BN-ReLU blocks with downsampling.
    - AdaptiveAvgPool2d(1) + Linear projection to feature_dim.

    Args:
        in_channels: number of input channels C.
        feature_dim: output embedding dimension F.
        channels: an iterable defining the conv channel widths per stage.
    """
    def __init__(self, in_channels: int, feature_dim: int = 256,
                 channels=(32, 64, 128)):
        super().__init__()
        layers = []
        c_prev = in_channels
        for i, c in enumerate(channels):
            layers.append(tc.nn.Conv2d(c_prev, c, kernel_size=3, stride=2, padding=1, bias=False))
            layers.append(tc.nn.BatchNorm2d(c))
            layers.append(tc.nn.ReLU(inplace=True))
            # an extra 3x3 stride-1 conv for capacity
            layers.append(tc.nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(tc.nn.BatchNorm2d(c))
            layers.append(tc.nn.ReLU(inplace=True))
            c_prev = c
        self.backbone = tc.nn.Sequential(*layers)
        self.pool = tc.nn.AdaptiveAvgPool2d((1, 1))
        self.proj = tc.nn.Linear(c_prev, feature_dim)
        self._output_dim = feature_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, curr_obs: tc.FloatTensor) -> tc.FloatTensor:
        # curr_obs: [B, C, H, W]
        x = self.backbone(curr_obs.float())
        x = self.pool(x).flatten(1)  # [B, C]
        x = self.proj(x)             # [B, F]
        return x


class MDPPreprocessing(Preprocessing):
    def __init__(self, num_actions: int, vision_net: VisionNet):
        super().__init__()
        self._num_actions = num_actions
        self._vision_net = vision_net

    @property
    def output_dim(self):
        return self._vision_net.output_dim + self._num_actions + 2

    def forward(
        self,
        curr_obs: tc.FloatTensor,
        prev_action: tc.LongTensor,
        prev_reward: tc.FloatTensor,
        prev_done: tc.FloatTensor
    ) -> tc.FloatTensor:
        """
        Creates an input vector for a meta-learning agent.

        Args:
            curr_obs: tc.FloatTensor of shape [B, ..., C, H, W]
            prev_action: tc.LongTensor of shape [B, ...]
            prev_reward: tc.FloatTensor of shape [B, ...]
            prev_done: tc.FloatTensor of shape [B, ...]

        Returns:
            tc.FloatTensor of shape [B, ..., F+A+2]
        """

        curr_obs_shape = list(curr_obs.shape)
        curr_obs = curr_obs.view(-1, *curr_obs_shape[-3:])
        emb_o = self._vision_net(curr_obs)
        emb_o = emb_o.view(*curr_obs_shape[:-3], emb_o.shape[-1])

        emb_a = one_hot(prev_action, depth=self._num_actions)
        prev_reward = prev_reward.unsqueeze(-1)
        prev_done = prev_done.unsqueeze(-1)
        vec = tc.cat(
            (emb_o, emb_a, prev_reward, prev_done), dim=-1).float()
        return vec

class MDPPreprocessingContinuous(Preprocessing):
    """
    For continuous action: concat [vision_embedding, prev_action (float A-dim), reward, done]
    """
    def __init__(self, action_dim: int, vision_net: VisionNet):
        super().__init__()
        self._action_dim = action_dim
        self._vision_net = vision_net

    @property
    def output_dim(self):
        return self._vision_net.output_dim + self._action_dim + 2

    def forward(
        self,
        curr_obs: tc.FloatTensor,
        prev_action: tc.FloatTensor,
        prev_reward: tc.FloatTensor,
        prev_done: tc.FloatTensor
    ) -> tc.FloatTensor:
        curr_obs_shape = list(curr_obs.shape)
        curr_obs = curr_obs.view(-1, *curr_obs_shape[-3:])
        emb_o = self._vision_net(curr_obs)
        emb_o = emb_o.view(*curr_obs_shape[:-3], emb_o.shape[-1])

        # prev_action expected shape [..., A]
        prev_reward = prev_reward.unsqueeze(-1)
        prev_done = prev_done.unsqueeze(-1)
        vec = tc.cat((emb_o, prev_action.float(), prev_reward, prev_done), dim=-1).float()
        return vec
