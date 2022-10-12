from typing import List, Tuple

from datagen.trajectory import Trajectory
from memup import nets
from memup import UncertaintyDetector, MemupBatch
from rl.memup_trainer import RLMemUPTrainer

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils.rnn as utils_rnn


from rl import envs, memup_trainer
from rl.generators import SequentialEpisodeGenerator
from rl.preprocessing import DiscountedReturn

from memup import MemUPMemory, MemUPPredictor, training
from memup import TopKSelector, CompositeSelector, CurrentStepSelector
from memup import TruncatedMemUPSampler, MemUPEvalSampler
from memup import PredictionErrorBasedDetector
from metrics import MSEMetric


from datagen import TrajectoryBuffer, RandomSampler, OrderedSampler
from datagen.preprocessing import Composite, AddTailEvalTargets


class TMazeRecMemory(nets.RecurrentModule):

    def __init__(
            self, input_shape, num_actions, hidden_dim=256,
            rnn_layers=1, ignore_prev_reward=False, dropout=0.
    ):
        super(TMazeRecMemory, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.rnn_layers = rnn_layers
        self.dropout=dropout
        self.ignore_prev_reward = ignore_prev_reward
        self._input_keys = ['observation', 'prev_action', 'prev_reward', 'done']
        self._create_network()

    def input_keys(self):
        return list(self._input_keys)

    def _create_network(self):
        obs_dim = 64
        self.obs_encoder = nn.Sequential(
            nn.Linear(np.prod(self.input_shape), obs_dim),
            nn.ReLU(),
        )
        # obs_encoding+prev_onehot_act+prev_r = embedding_dim
        self.embedding_dim = obs_dim + self.num_actions + 1
        self.lstm = nn.LSTM(  #
            self.embedding_dim, self.hidden_dim,
            num_layers=self.rnn_layers, batch_first=True,
            dropout=self.dropout,
        )

    def forward(self, input_dict, mem_state, **kwargs):
        lengths = [len(e) for e in input_dict['prev_reward']]

        obs, prev_actions, prev_rewards = nets.pad_input_sequence(
            input_dict['observation'],
            input_dict['prev_action'],
            input_dict['prev_reward']
        )

        if self.ignore_prev_reward:
            prev_rewards *= 0.

        embeds = self.encoder(obs.to(torch.float), prev_actions, prev_rewards)
        packed_embeds = utils_rnn.pack_padded_sequence(
            embeds, lengths,
            batch_first=True,
            enforce_sorted=False
        )

        outputs, mem_state = self.lstm(packed_embeds, mem_state)
        padded_outputs, lengths = utils_rnn.pad_packed_sequence(outputs, batch_first=True)
        mem_state = self.mask_hidden_state(mem_state, input_dict['done'])

        return padded_outputs, mem_state

    def forward2(self, input_dict, mem_state, lengths, **kwargs):
        obs, prev_actions, prev_rewards = input_dict['observation'], input_dict['prev_action'], input_dict['prev_reward']

        if self.ignore_prev_reward:
            prev_rewards *= 0.

        embeds = self.encoder(obs.to(torch.float), prev_actions, prev_rewards)
        packed_embeds = utils_rnn.pack_padded_sequence(
            embeds, lengths,
            batch_first=True,
            enforce_sorted=False
        )

        outputs, mem_state = self.lstm(packed_embeds, mem_state)
        padded_outputs, lengths = utils_rnn.pad_packed_sequence(outputs, batch_first=True)
        mem_state = self.mask_hidden_state(mem_state, input_dict['done'])

        return padded_outputs, mem_state


    def encoder(self, obs, prev_acts, prev_rewards):
        N, T = prev_acts.shape[:2]

        flatten_obs, flatten_acts, flatten_rewards = nets.flatten(
            obs, prev_acts, prev_rewards
        )
        flatten_o_embed = self.obs_encoder(flatten_obs)

        flatten_embed = torch.cat(
            [flatten_o_embed,
             flatten_acts.to(torch.float32),
             flatten_rewards.to(torch.float32)]
            , dim=-1
        )

        embed = flatten_embed.view(N, T, *flatten_embed.shape[1:])
        return embed


class TMazePredictor(nets.PredictorModule):

    def __init__(self,
                 memory_dim,
                 input_shape,
                 num_actions,
                 input_key='observation'):
        super(TMazePredictor, self).__init__()
        self.memory_dim = memory_dim
        self.num_actions = num_actions
        self.input_shape = input_shape
        self.input_key = input_key

        self._create_network()

    def _create_network(self):
        # obs_encoding + prev_onehot_act + prev_r == memory.embedding_dim
        encoder_dim = 64
        question_dim = encoder_dim + self.num_actions

        self.obs_encoder = nn.Sequential(
            nn.Linear(np.prod(self.input_shape), encoder_dim),
            nn.ReLU()
        )

        self.quest_fc = nn.Linear(question_dim, self.memory_dim)
        predictor_input_dim = self.memory_dim * 2
        self.pred_mlp = nn.Sequential(
            nn.Linear(predictor_input_dim, predictor_input_dim),
            nn.ReLU(),
            nn.Linear(predictor_input_dim, 1)
        )

    def input_keys(self):
        return ['action', self.input_key]

    def forward(self, context_input, memory_states, **kwargs):
        obs = torch.cat(context_input[self.input_key])
        acts = F.one_hot(torch.cat(context_input['action']).type(torch.int64), self.num_actions)

        question = torch.cat(
            [self.obs_encoder(obs), acts.to(torch.float32)], dim=-1
        )
        quest_embed = self.quest_fc(question)
        predictions = self.pred_mlp(
            torch.cat([memory_states, quest_embed], dim=-1)
        )
        return predictions.squeeze(-1)


class TMazeMemoryTrainer(RLMemUPTrainer):

    def __init__(self):
        # train and test env have different seeds
        train_env, test_env = envs.train_and_test_from_config(env_config)
        # generates episodes given env and policy
        # uses random policy if policy is not specified:
        train_gen = SequentialEpisodeGenerator(train_env)
        test_gen = SequentialEpisodeGenerator(test_env)

        obs_shape = train_env.observation_space['observation'].shape
        num_actions = train_env.action_space.n

        # if memory_as_context=True then accumulated memory is used for
        # context embedding in target events,
        # otherwise we use respective observation
        pred_obs_shape = (rnn_dim,) if memory_as_context else obs_shape
        pred_input_key = 'context' if memory_as_context else 'observation'

        # =========== Memory Init ====================
        create_mem = lambda:TMazeRecMemory(
            obs_shape, num_actions, rnn_dim
        ).to(device)

        memory_net = create_mem()

        memup_memory = MemUPMemory(memory_net)

        mem_state = memup_memory.init_state(batch_size)
        # =============================================
        # =========== Predictor Init ===================
        create_pred = lambda:TMazePredictor(
            rnn_dim, pred_obs_shape, num_actions, pred_input_key
        ).to(device)

        predictor_net = create_pred()

        memup_predictor = MemUPPredictor(predictor_net, target_key)
        # =============================================
        # =========== optimization part ===============
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(
            chain(memory_net.parameters(), predictor_net.parameters()),
            lr=5e-4,  # weight_decay=1e-7
        )
        # =============================================
        # Initialize stuff that create MemUPBatches:
        print('loading train buffer...')
        train_buffer = TrajectoryBuffer(buffer_size, DiscountedReturn(gamma))
        train_buffer.add_trajectories(train_gen.gen_trajs(buffer_size))
        # memory and predictor accumulators are used
        # for uncertainty detection
        # and (optionally) memory_acc can be used for context embedding
        predictor_acc = create_pred()
        memory_acc = create_mem()

        # Uses accumulated versions of memory_acc and predictor_acc
        # to estimate uncartainty via prediction error
        # if context_key is specified then
        # hidden_states of memory_acc are added to the trajectories
        unc_detector = PredictionErrorBasedDetector(
            MemUPMemory(memory_acc),
            MemUPPredictor(predictor_acc, target_key),
            error_metric=torch.nn.MSELoss(reduction='none'),
            context_key='context' if memory_as_context else None
        )

        # Samples from buffer and constructs MemUPBatch
        train_sampler = TruncatedMemUPSampler(
            RandomSampler(train_buffer),
            # how to select targets for prediction
            # right now at each prediction step two targets are selected
            # (step with the highest uncertainty estimate, current step)
            CompositeSelector([
                TopKSelector(1, time_dependent_selection=True),
                CurrentStepSelector()
            ]),
            # uncertainty detector is simply a preprocessor that
            # adds uncertainty_key to the trajectories in the batch
            unc_detector,
            # determines which steps in rollout are selected for prediction:
            prediction_frequency=pred_freq,
            # length of trajectory subsequences to be processed:
            rollout=rollout,
        )

        # =========== creating evaluation sampler ==============
        # We need to test model not on the different samples:
        print('loading eval buffer...')
        eval_buffer = TrajectoryBuffer(
            num_eval_episodes,
            # Eval buffer assumes that trajectories have
            # a special key that contains target steps at which
            # we want to test our models predictions
            # AddTailEvalTargets(1) preprocessor
            # adds this key to trajectories
            # it stores only index of the last step for each trajectory
            Composite([DiscountedReturn(gamma), AddTailEvalTargets(1)])
        )
        eval_buffer.add_trajectories(test_gen.gen_trajs(num_eval_episodes))

        if memory_as_context:
            eval_prepoc = unc_detector
        eval_sampler = MemUPEvalSampler(
            OrderedSampler(eval_buffer),
            dynamic_preprocessor=unc_detector if memory_as_context else None
        )