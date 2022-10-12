import gym_minigrid as mg
import gym
import cv2
from  .common_wrappers import FrameStack

def make_pytorch_minigrid(
        env_id,
        exploration=False,
        one_hot=False,
        image_only=True,
        living_reward=0.0,
        framestack=1,
):
    env = gym.make(env_id)
    if not one_hot:
        env = mg.wrappers.RGBImgPartialObsWrapper(env)
    else:
        env = mg.wrappers.OneHotPartialObsWrapper(env)
    if exploration:
        env = mg.wrappers.ActionBonus(env)
    if image_only:
        env = mg.wrappers.ImgObsWrapper(env)
    if living_reward:
        env = LivingReward(env, living_reward)
    env = PyTorchFrame(env)

    if framestack > 1:
        env = FrameStack(env, framestack)

    return env


class LivingReward(gym.core.Wrapper):

    def __init__(self, env, living_reward=-0.01):
        super(LivingReward, self).__init__(env)
        self.living_reward = living_reward

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward += self.living_reward
        return  obs, reward, done, info


class PyTorchFrame(gym.core.ObservationWrapper):
    def __init__(self, env, resize=None):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        :param env: (Gym Environment) the environment
        """
        super(PyTorchFrame, self).__init__(env)
        self.resize = resize
        H,W,C = self.env.observation_space.shape
        if resize: H,W = resize

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(C, H, W),
            dtype=env.observation_space.dtype)

    def observation(self, frame):
        """
        returns the current observation from a frame
        :param frame: ([int] or [float]) environment frame
        :return: ([int] or [float]) the observation
        """

        if self.resize:
            frame = cv2.resize(
                frame, self.resize, interpolation=cv2.INTER_AREA
            )

        return frame.transpose(2, 0, 1)
