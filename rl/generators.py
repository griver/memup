import gym.spaces
import numpy as np
import torch
from datagen import TrajGenerator, Trajectory, TrajectoryPreprocessor
from itertools import count
from collections import defaultdict

class Policy(object):
    def act(self, obs, **kwarg):
        raise NotImplementedError()


class RandomPolicy(Policy):

    def __init__(self, env):
        self.action_space = env.action_space

    def act(self, obs, **kwargs):
        return self.action_space.sample()

class EpsilonGreedyQRDQN(Policy):
    """
    Expects a value based model dealing with discrete action space
    Actually expects a QR-DQN model
    """
    def __init__(self, model, epsilon=0.01):
        self.model = model
        self.epsilon = epsilon
        self.device = next(model.parameters()).device

    def act(self, obs, **kwargs):
        obs = obs['observation']
        if obs.shape == self.model.input_shape:
            return self._act_batch(obs[None,:], **kwargs)[0]

        return self._act_batch(obs, **kwargs)

    torch.no_grad()
    def _act_batch(self, obs, **kwargs):
        obs = torch.as_tensor(obs, device=self.device)
        q_values = self.model.calculate_q(states=obs).cpu()

        is_random = torch.rand(obs.shape[0]) < self.epsilon
        num_random = is_random.sum().item()

        actions = q_values.argmax(-1)
        actions[is_random] = torch.randint(self.model.num_actions, size=(num_random,))

        return actions

    # def state2th(state, device):
    #     if isinstance(state, dict):
    #         # div by 255. removed
    #         obs = th.as_tensor(state['obs'], dtype=th.float, device=device).unsqueeze(0)  # / 255.
    #         memory = th.as_tensor(state['memory'], dtype=th.float, device=device)
    #         state = (obs, memory)
    #     else:
    #         # div by 255. removed
    #         obs = th.as_tensor(state, dtype=th.float, device=device).unsqueeze(0)  # / 255.
    #         state = obs
    #
    #     return state


class SequentialEpisodeGenerator(TrajGenerator):

    def __init__(
        self,
        env,
        policy=None,
        verbose=False,
    ):
        super().__init__()
        self.env = env
        self.policy = policy if policy else RandomPolicy(env)

        self.verbose=verbose

        if isinstance(env.observation_space, gym.spaces.Dict):
            self._append_obs = self._append_dict_obs
        else:
            self._append_obs = self._append_vec_obs

    def _append_dict_obs(self, obs, data):
        for k,v in obs.items():
            data[k].append(v)

    def _append_vec_obs(self, obs, data):
        data['observation'].append(obs)

    def gen_trajectory(self):
        """
        Plays an episode using self.policy and self.env
        """
        s = self.env.reset()

        data = defaultdict(list)
        self._append_obs(s, data)

        for t in count(1):

            action = self.policy.act(s)
            s, r, done, _ = self.env.step(action)

            self._append_obs(s, data)
            data['action'].append(action)
            data['reward'].append(r)
            data['done'].append(done)

            if done:
                data = {k:np.array(v) for k,v in data.items()}
                data['observation_shape'] = self.env.observation_space.shape
                data['num_actions'] = self.env.action_space.n

                return Trajectory(data)


class FilteringEpisodeGenerator(SequentialEpisodeGenerator):
    """
    Generates episodes the same way as SequentialEpisodeGenerator,
    but filters episodes without enough high rewards.
    New constructor arguments:
        min_abs_r: rewards with values greater or equal to min_abs_r are counted as high
        min_num_r: minimal number of high rewards in the episode
    """
    class CantGenerateEpisode(BaseException):
        pass

    def __init__(
            self,
            env,
            policy=None,
            min_abs_r=0.5,
            min_num_r=2,
            verbose=False
    ):
        super().__init__(env, policy, verbose)
        self.min_abs_r=min_abs_r
        self.min_num_r=min_num_r
        self.max_tries = 100

    def gen_trajectory(self):

        for i in range(self.max_tries):
            traj = super().gen_trajectory()
            is_high = np.abs(traj.data['reward']) > self.min_abs_r
            if sum(is_high) >= self.min_num_r:
                break
        else:
            raise ValueError("")
        return traj


class VizdoomEvalDataset(TrajGenerator):

    def __init__(self, filepath):
        self.trajs = self._load_eval_dataset(filepath)
        self._current = 0

    def _load_eval_dataset(self, filepath):
        episodes = np.load(filepath, allow_pickle=True)['episodes']

        trajs = []
        for i, e in enumerate(episodes):
            #print(f'#{i}, keys: {e.keys()}')
            #print(f'#{i}, targets: {e["target_ids"]}')
            if len(e['target_ids']) == 0:
                continue

            traj = Trajectory(dict(
                reward=e['rewards'],
                observation=e['observations'],
                action=e['actions'],
                done=e['terminals'],
                eval_targets=e['target_ids'],
                observation_shape=e['observation_shape'],
                num_actions=e['num_actions'],
            ))
            trajs.append(traj)

        return trajs

    def gen_trajectory(self):
        if self._current < len(self.trajs):
            traj = self.trajs[self._current]
            self._current += 1
            return traj

        return None


# class EpsilonGreedy(Policy):
#
#     def __init__(self, model, device, epsilon=0.):
#         self.model = model
#         self.epsilon = epsilon
#         self.device = device
#
#     th.no_grad()
#     def act(self, obs):
#         # div by 255. removed
#         obs = state2th(obs, self.device)
#
#         q_values = self.model.calculate_q(states=obs).squeeze()
#
#         assert len(q_values.shape) == 1, 'this code is not ready for batched observations'
#
#         if np.random.random() > self.epsilon:
#             action = q_values.argmax().item()
#         else:
#             action = np.random.randint(q_values.numel())
#         return action


#
# def state2th(state, device):
#     to_th = lambda v: th.as_tensor(v, dtype=th.float, device=device)
#
#     if isinstance(state, dict):
#         return {k:to_th(v) for k,v in state.items()}
#     else:
#         return to_th(state)



