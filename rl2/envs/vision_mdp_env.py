from typing import Tuple, Optional
import numpy as np

from rl2.envs.abstract import MetaEpisodicEnv

# 兼容 gymnasium 与 gym
import gym


def _to_chw01(img: np.ndarray) -> np.ndarray:
    # 输入 HxWxC uint8，输出 CxHxW float32 in [0,1]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return img


def _nn_resize_chw(img_chw: np.ndarray, out_shape: Tuple[int, int, int]) -> np.ndarray:
    # 最近邻缩放，无外部依赖
    C, H, W = img_chw.shape
    C2, H2, W2 = out_shape
    if (C, H, W) == (C2, H2, W2):
        return img_chw
    if C2 != C:
        raise ValueError(f"Channel mismatch: got {C2}, expected {C}")
    ys = np.clip(np.round(np.linspace(0, H - 1, H2)).astype(np.int32), 0, H - 1)
    xs = np.clip(np.round(np.linspace(0, W - 1, W2)).astype(np.int32), 0, W - 1)
    return img_chw[:, ys][:, :, xs]


class VisionMDPEnv(MetaEpisodicEnv[np.ndarray]):
    """
    Gym 视觉环境包装器，返回图像观测 (C,H,W)，值域 [0,1]。
    - 默认使用 CartPole-v1（离散动作），也可通过 env_id 更换。
    - 支持 max_episode_length 截断；auto_reset=True 时在 done 后自动 reset。
    - new_env() 重建并随机播种底层 Gym 环境，使元回合之间具有变化。
    """

    def __init__(
        self,
        num_states: int,  # 仅为接口占位，Gym 环境不使用
        num_actions: int,
        max_episode_length: int = 10,
        image_shape: Optional[Tuple[int, int, int]] = (3, 64, 64),
        env_id: str = "CartPole-v1",
        seed: Optional[int] = None,
    ):
        # 结构参数
        self._num_states = num_states
        self._num_actions = num_actions
        self._max_ep_length = max_episode_length
        self._image_shape = image_shape
        self._env_id = env_id
        self._seed = seed

        # 运行时状态
        self._env = None
        self._ep_steps_so_far = 0

        self.new_env()

    @property
    def max_episode_len(self) -> int:
        return self._max_ep_length

    @property
    def num_actions(self) -> int:
        return self._num_actions

    def _make_env(self, seed: Optional[int]):
        # 使用 rgb_array 渲染模式
        
        env = gym.make(self._env_id, render_mode="rgb_array")

        if seed is None:
            seed = np.random.randint(0, 2**31 - 1)
        env.reset(seed=seed)
        return env

    def _get_image_obs(self) -> np.ndarray:
        # 渲染并转为 (C,H,W) float32 in [0,1]
        frame = self._env.render()
        img = _to_chw01(frame)
        if self._image_shape is not None:
            img = _nn_resize_chw(img, self._image_shape)
        return img

    def new_env(self) -> None:
        if self._env is not None:
            self._env.close()
        seed = self._seed if self._seed is not None else np.random.randint(0, 2**31 - 1)
        self._env = self._make_env(seed=seed)

        space = self._env.action_space
        # 支持 Discrete(n) 与 Box(shape=(A,))
        if hasattr(space, "n"):
            env_actions = int(space.n)
            if env_actions != self._num_actions:
                raise ValueError(f"动作数量不匹配: 期望 {self._num_actions}, 但底层 {self._env_id} 为 {env_actions}。")
            self._is_discrete = True
            self._action_low = None
            self._action_high = None
        elif hasattr(space, "shape"):
            assert len(space.shape) == 1, "仅支持一维 Box 连续动作"
            self._is_discrete = False
            self._num_actions = int(space.shape[0])
            self._action_low = np.array(space.low, dtype=np.float32)
            self._action_high = np.array(space.high, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported action space: {space}")

        self._ep_steps_so_far = 0

    def reset(self) -> np.ndarray:
        self._ep_steps_so_far = 0
        self._env.reset()
        return self._get_image_obs()

    def step(self, action, auto_reset: bool = True):
        self._ep_steps_so_far += 1
        if getattr(self, "_is_discrete", True):
            a = int(action)
        else:
            a = np.asarray(action, dtype=np.float32)
            if self._action_low is not None:
                a = np.clip(a, self._action_low, self._action_high)
        _obs, reward, terminated, truncated, info = self._env.step(a)
        done_env = bool(terminated or truncated)
        done_len = not (self._ep_steps_so_far < self._max_ep_length)
        done_t = bool(done_env or done_len)
        obs_tp1 = self._get_image_obs()
        if done_t and auto_reset:
            obs_tp1 = self.reset()
        return obs_tp1, float(reward), done_t, info
