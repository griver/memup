import numpy as np
import gym
import itertools
from rl.envs.common_wrappers import ImgObsWrapper, FrameStack, PrevActionAndReward
from enum import IntEnum, Enum
import logging

#logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logger = logging.getLogger()
logger.setLevel(logging.WARNING)

class MemoryLen(gym.Env):
    INERNAL_STATE = ['left', 'right']
    OBS_DIM = 3

    class Actions(IntEnum):
        left=0
        right=1

    def __init__(self, idx, seed=None, prob=0.5, length=10, length_deviation=0):

        super(MemoryLen, self).__init__()
        self.idx = idx
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Dict({
            "observation":gym.spaces.Box(-1.,1., shape=(self.OBS_DIM,),dtype=np.float32),
        })

        self.len_min = length
        self.len_div = length_deviation
        self.prob = prob
        self.obs_dim = self.OBS_DIM
        self.rnd = np.random.RandomState(
            seed if seed is None else seed*(self.idx+1)
        )
        self.actions = MemoryLen.Actions

    def step(self, action):

        self.t += 1
        if self.t >= self.length:
            if self._internal_state == "left":
                r = 1. if action == 0 else -1.
            elif self._internal_state == "right":
                r = 1. if action == 1 else -1.
            else:
                raise ValueError("internal_state should be in {}".format(self.INERNAL_STATE))

            obs = self.make_obs(self.t) #np.zeros(self.obs_dim, dtype=np.float32)
            return obs, r, True, {
                'episode_success':r == 1.,
                'episode_reward':r,
                'episode_len': self.t,
            }
        else:
            return self.make_obs(self.t), 0., False, {}

    def reset(self):
        self._internal_state = self.rnd.choice(self.INERNAL_STATE)
        self.t = 0

        self.length = self.len_min
        if self.len_div > 0:
            self.length += self.rnd.randint(0, self.len_div + 1)

        #print("NEW EPISODE LENGTH =", self.length)

        return self.make_obs(self.t)

    def make_obs(self, t):
        time_mark = t/(self.length-1)
        obs = np.full(self.obs_dim, time_mark, dtype=np.float32)
        obs[0] = obs[-1] = 0.
        if t == 0:
            if self._internal_state == 'left':
                obs[0] = -1
            elif self._internal_state == 'right':
                obs[0] = 1.
            else:
                raise ValueError("Internal states should be either 'left' or 'right'!")
        elif t >= self.length:
            obs[:] = 0.

        return {"observation": obs}

    def close(self):
        del self.rnd


class TMazeAMRL(MemoryLen):
    """
    AMRL style T-maze. See: https://openreview.net/forum?id=Bkl7bREtDr

    Modes:
    L: like MemoryLen, the only difference is presence of T-junction indicator
    and absence of observation that tells you how far you are in the corridor

    LN: L+noise last element of observation are randomly switches between -1 and 1

    LS: like LN but to move forward in the corridor you need to select action correspondent to the observed noisy element
    e.g select "left" if obs[-1] == -1 otherwise select "right"
    """
    class Mode(Enum):
        L = "L"
        LR = "LR"
        LN = "LN"
        LNR = "LNR"
        LS = "LS"

    def __init__(self, idx, seed=None, prob=0.5, length=10,
                 mode="LN", success_r=4., fail_r=-3., movement_r=0.0):
        assert isinstance(mode, str), "Hack that introduces LR and LNR modes doesn't work when type(mode) == Mode "

        self.mode = mode if isinstance(mode, self.Mode) else self.Mode[mode]  # str or Mode

        self.use_random_len = self.mode.value[-1] == "R"

        super(TMazeAMRL, self).__init__(
            idx, seed, prob=prob, length=length,
            length_deviation=10*int(self.use_random_len)
        )

        if self.mode is self.Mode.LS:
            logging.warning("LS mode is not finished yet: no episode time limit")

        self.success_r = success_r
        self.fail_r = fail_r
        assert movement_r == 0 or self.mode == self.Mode.LS, 'Movement reward is pointless in L and LN modes.'
        self.movement_r = movement_r
        self._prev_obs = None
        self.total_r=0.
        #print('success_r: ', self.success_r, "fail_r:", self.fail_r)

    def reset(self):
        self.total_r = 0.
        return super(TMazeAMRL,self).reset()

    def step(self, action):

        if self.mode == self.Mode.LS and self.t+1 < self.length: #not at T-junction
            # check if action matches to the noise value from previous step
            act_matches = (action > 0) == (self._prev_obs[-1] > 0) #0 and -1., or 1 and 1.
            if not act_matches:
                return self.make_obs(self.t, repeat_prev=True), 0., False, {}

        self.t += 1
        if self.t >= self.length:
            if self._internal_state == "left":
                r = self.success_r if action == 0 else self.fail_r
            elif self._internal_state == "right":
                r = self.success_r if action == 1 else self.fail_r
            else:
                raise ValueError("internal_state should be in {}".format(self.INERNAL_STATE))

            self.total_r += r
            obs = self.make_obs(self.t) #np.zeros(self.obs_dim, dtype=np.float32)
            return obs, r, True, {
                'episode_success': r == self.success_r,
                'episode_reward':self.total_r,
                'episode_len': self.t,
            }
        else:
            self.total_r += self.movement_r
            return self.make_obs(self.t), self.movement_r, False, {}

    def make_obs(self, t, repeat_prev=False):
        if repeat_prev:
            return {'observation': self._prev_obs}

        use_noise = self.mode not in (self.Mode.L, self.Mode.LR)
        noise = self.rnd.choice([1., -1.]) if use_noise else 0.
        at_t_junction = t == (self.length-1)
        hint = 0.

        if t == 0:
            if self._internal_state == 'left':
                hint = -1.
            elif self._internal_state == 'right':
                hint = 1.
            else:
                raise ValueError("Internal states should be either 'left' or 'right'!")
        elif t >= self.length:
            noise = hint = at_t_junction = 0.

        obs = np.array([hint, at_t_junction, noise], dtype=np.float32)

        self._prev_obs = obs

        return {"observation": obs}


class TMazeWithDistractorRewards(TMazeAMRL):

    EMPTY_CORRIDOR_LENGTH=5

    def __init__(self, idx, seed=None,
                 prob=0.5, length=10, mode="LN",
                 success_r=4., fail_r=-3.,
                 movement_r=0.0,
                 num_distractors=1,
                 fixed_choice_step=True
                 ):
        assert num_distractors < (length - self.EMPTY_CORRIDOR_LENGTH), 'Maze starts with at least 5 steps without rewards'
        assert mode[-1] != 'S', "TMazeWithDistractorRewards doesn't work with LS mode!"
        logger.info(f"TMazeWithDistractorRewards(idx={idx}, seed={seed}, length={length}, num_distr={num_distractors}, fixed_choice_step={fixed_choice_step})")

        super().__init__(idx, seed, prob, length, mode, success_r, fail_r, movement_r)
        self.num_distractors = num_distractors
        self.fixed_choice_step=fixed_choice_step

        self._current_choice_step = None
        self._distractors_and_choice_steps = None
        self._episode_success = None

    def reset(self):

        self._distractors_and_choice_steps = [-100]
        self._index_pointer = 0

        obs = super().reset()

        self._episode_success = None
        self._distractors_and_choice_steps = self.rnd.choice(
            self.length - self.EMPTY_CORRIDOR_LENGTH - 1,
            self.num_distractors,
            replace=False) + self.EMPTY_CORRIDOR_LENGTH

        self._distractors_and_choice_steps = sorted(self._distractors_and_choice_steps)
        self._distractors_and_choice_steps.append(self.length-1)

        if self.fixed_choice_step:
            self._current_choice_step = self.num_distractors #id of the last element
        else:
            self._current_choice_step = self.rnd.choice(self.num_distractors+1)

        # logger.info(
        # f"""TMazeWithDistractorRewards(idx={self.idx}):
        #     curr_L={self.length},
        #     seed={self.seed}
        #     reward_steps={self._distractors_and_choice_steps},
        #     true_choice={self._current_choice_step}""")

        return obs

    def _compute_rewards_for_true_choice(self, action):
        r = 0.
        if self._internal_state == "left":
            r = self.success_r if action == 0 else self.fail_r
        elif self._internal_state == "right":
            r = self.success_r if action == 1 else self.fail_r
        else:
            raise ValueError("internal_state should be in {}".format(self.INERNAL_STATE))
        self._episode_success = (r == self.success_r)
        return r

    def step(self, action):

        #if previous observation had a signal about future reward
        if (self.t) == self._distractors_and_choice_steps[self._index_pointer]:
            if self._index_pointer == self._current_choice_step:
                r = self._compute_rewards_for_true_choice(action)
            else:
                r = self.rnd.choice([self.success_r, self.fail_r])

            self._index_pointer += 1
        else:
            r = self.movement_r

        self.total_r += r

        self.t += 1
        obs = self.make_obs(self.t)

        if self.t >= self.length:
             #np.zeros(self.obs_dim, dtype=np.float32)
            return obs, r, True, {
                'episode_success': self._episode_success, #this won't work with self.fixed_choice_step == False
                'episode_reward': self.total_r,
                'episode_len': self.t,
            }
        else:
            return obs, r, False, {}


    def make_obs(self, t, repeat_prev=False):
        if repeat_prev:
            return {'observation': self._prev_obs}

        use_noise = self.mode not in (self.Mode.L, self.Mode.LR)
        noise = self.rnd.choice([1., -1.]) if use_noise else 0.
        hint = 0.

        if t == 0:
            at_t_junction = 0
            if self._internal_state == 'left':
                hint = -1.
            elif self._internal_state == 'right':
                hint = 1.
            else:
                raise ValueError("Internal states should be either 'left' or 'right'!")

        elif t < self.length:
            #at_t_junction = (t == self._distractors_and_choice_steps[self._index_pointer])
            at_t_junction = 0
            if t == self._distractors_and_choice_steps[self._index_pointer]:
                at_t_junction = (self._index_pointer+1)/len(self._distractors_and_choice_steps)
        elif t >= self.length:
            noise = hint = at_t_junction = 0.

        obs = np.array([hint, at_t_junction, noise], dtype=np.float32)

        self._prev_obs = obs

        return {"observation": obs}



def make_pytorch_memory_len(
        envid, id=0, seed=None, framestack=1, obs_only=False, **kwargs
    ):
    envid = envid.lower()

    if envid.startswith('memory-len'):
        length = envid.split('-')[2:] #e.g. memory-len-100
        env = MemoryLen(id, seed, length=int(length), **kwargs)

    elif envid.startswith('t-maze-distractors'):
    # t-maze-distractors-{num_distractors}-{mode}-{length}: t-maze-distractors-5-lnr-50
        num_distractors, mode, length = envid.split('-')[3:] #ignore t-maze-distractors
        env = TMazeWithDistractorRewards(
            id, seed, length=int(length),
            mode=mode.upper(), num_distractors=int(num_distractors),
            **kwargs
        )

    elif envid.startswith('t-maze'):
        mode, length = envid.split('-')[2:]
        env = TMazeAMRL(id, seed, length=int(length), mode=mode.upper(), **kwargs)

    #if not dict_obs:
    #    env = ImgObsWrapper(env)

    if framestack > 1:
        env = FrameStack(env, framestack)

    if obs_only: return env
    return PrevActionAndReward(env)


def read_action():
    action = input("Input Your Action: ")
    if action in ['0', '1']:
        print('act:', action)
        return int(action)
    else:
        action = np.random.choice([0,1])
        print('random act:', action)
        return action


def play_dummy_memory(num_episodes, EnvCLS, **env_kwargs):
    seed = env_kwargs.get('seed', None)
    idx = env_kwargs.get('idx', 0)
    env = EnvCLS(idx, seed, **env_kwargs)
    np.set_printoptions(precision=2)
    for i in range(num_episodes):
        obs = env.reset()
        print("\n====== EPISODE #{} ======".format(i))
        print("episode_length:", env.length)
        print('choice and distractors steps:', env._distractors_and_choice_steps)
        print('true choice:', env._distractors_and_choice_steps[env._current_choice_step])
        for t in itertools.count():
            print("===== STEP#{} =====".format(t))
            print("obs: ",obs['observation'])
            act = read_action()
            obs, r, done, info = env.step(act)
            print("r={:.2f}, done={}, info={}".format(r,done, info))
            if done: break


def check_dummy_memory(seed, num_episodes, episode_length):
    env = MemoryLen(0, seed, length=episode_length)
    count_left = 0
    count_success = 0
    for i in range(num_episodes):
        obs = env.reset()
        count_left += int(obs['obs'][0] == -1.)
        #print("\n====== EPISODE #{} ======".format(i))
        for t in itertools.count():
            #print("===== STEP#{} =====".format(t))
            #print("obs: ",obs['obs'])
            act = np.random.choice([0,1]) # read_action()
            obs, r, done, info = env.step(act)
            #print("r={:.1f}, done={}, info={}".format(r,done, info))
            if done:
                count_success += int(info['episode_success'] == True)
                break

    print(f"Played {num_episodes} episodes")
    print("Generated {}/({:.2f}%) Left Configs".format(count_left, 100*count_left/num_episodes))
    print("Random policy success rate: {:.1f}%".format(100*count_success/num_episodes))


if __name__ == "__main__":
    #MemoryLen config: length, prob
    #TMazeAMRL config: length, mode=T_LN, success_r = 1., fail_r = -0.75, movement_r = 0.0)
    play_dummy_memory(10, TMazeWithDistractorRewards, length=10, mode='LN', movement_r=0.0, num_distractors=3, fixed_choice_step=True)
    #check_dummy_memory(None, 20000, 10)
