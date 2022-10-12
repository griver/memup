from typing import Dict, List

from ray.rllib.models import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils import override
from ray.rllib.utils.typing import TensorType
from torch import nn
import torch
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
import numpy as np
from examples.tmaze.rllib.srnn import SRNN
from examples.tmaze.tmaze_networks import TMazeRecMemory
from rl.training_utils import disable_gradients


class LSTMModel(TorchRNN, nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        fc_size=64,
        lstm_state_size=128
    ):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.obs_size = get_preprocessor(obs_space)(obs_space).size
        self.fc_size = fc_size
        self.lstm_state_size = lstm_state_size

        self.fc1 = nn.Linear(self.obs_size, self.fc_size)
        self.lstm = nn.LSTM(self.fc_size, self.lstm_state_size, batch_first=True, num_layers=2, dropout=0.1)
        self.action_branch = nn.Sequential(
            nn.Linear(self.lstm_state_size, self.lstm_state_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.lstm_state_size, self.lstm_state_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.lstm_state_size, num_outputs),
        )
        self.value_branch = nn.Sequential(
            nn.Linear(self.lstm_state_size, self.lstm_state_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.lstm_state_size, self.lstm_state_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.lstm_state_size, 1),
        )
        # Holds the current "base" output (before logits layer).
        self._features = None

    @override(ModelV2)
    def get_initial_state(self):
        h = [
            self.fc1.weight.new(2, self.lstm_state_size).zero_().squeeze(0),
            self.fc1.weight.new(2, self.lstm_state_size).zero_().squeeze(0),
        ]
        return h

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return torch.reshape(self.value_branch(self._features), [-1])

    @override(TorchRNN)
    def forward_rnn(self, inputs, state, seq_lens):
        if inputs.shape[1] > 1:
            print(inputs.shape[1])
        x = nn.functional.relu(self.fc1(inputs))
        self._features, [h, c] = self.lstm(
            x, [state[0].transpose(0, 1).contiguous(), state[1].transpose(0, 1).contiguous()]
        )
        action_out = self.action_branch(self._features)
        return action_out, [h.transpose(0, 1).contiguous(), c.transpose(0, 1).contiguous()]


class NoGradRNNModel(TorchRNN, nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        module
    ):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.obs_size = get_preprocessor(obs_space)(obs_space).size
        self.lstm: TMazeRecMemory = module()
        disable_gradients(self.lstm)
        self.lstm.eval()
        self.lstm_state_size = self.lstm.hidden_dim

        self.action_branch = nn.Sequential(
            nn.Linear(self.lstm_state_size, self.lstm_state_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.lstm_state_size, self.lstm_state_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.lstm_state_size, num_outputs),
        )
        self.value_branch = nn.Sequential(
            nn.Linear(self.lstm_state_size, self.lstm_state_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.lstm_state_size, self.lstm_state_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.lstm_state_size, 1),
        )
        # Holds the current "base" output (before logits layer).
        self._features = None

    @override(ModelV2)
    def get_initial_state(self):
        h = [
            self.value_branch[0].weight.new(1, self.lstm_state_size).zero_(),
            self.value_branch[0].weight.new(1, self.lstm_state_size).zero_()
        ]
        return h

    @override(TorchRNN)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        """Adds time dimension to batch before sending inputs to forward_rnn().

        You should implement forward_rnn() in your subclass."""

        if isinstance(seq_lens, np.ndarray):
            seq_lens = torch.Tensor(seq_lens).int()

        dones = input_dict['dones'] if 'dones' in input_dict else torch.zeros_like(input_dict["obs"]['prev_reward'])

        memory_input = {
            'observation': input_dict["obs"]['observation'],
            'prev_action': input_dict["obs"]['prev_action'],
            'prev_reward': input_dict["obs"]['prev_reward'],
            'done': dones
        }

        max_seq_len = memory_input['observation'].shape[0] // seq_lens.shape[0]

        for k in memory_input.keys():
            memory_input[k] = add_time_dimension(
                memory_input[k],
                max_seq_len=max_seq_len,
                framework="torch",
                time_major=False
            )
            # memory_input[k] = [memory_input[k][i][:seq_lens[i]] for i in range(memory_input[k].shape[0])]

        output, new_state = self.forward_rnn(memory_input, state, seq_lens)
        output = torch.reshape(output, [-1, self.num_outputs])
        return output, new_state

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return torch.reshape(self.value_branch(self._features), [-1])

    @override(TorchRNN)
    def forward_rnn(self, inputs, state, seq_lens):
        x = inputs
        self._features, [h, c] = self.lstm.forward2(
            x, [state[0].transpose(0, 1).contiguous(), state[1].transpose(0, 1).contiguous()], seq_lens.cpu()
        )
        action_out = self.action_branch(self._features)
        return action_out, [h.transpose(0, 1).contiguous(), c.transpose(0, 1).contiguous()]


class SRNNModel(TorchRNN, nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        fc_size=64,
        lstm_state_size=128,
    ):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.obs_size = get_preprocessor(obs_space)(obs_space).size
        self.fc_size = fc_size
        self.lstm_state_size = lstm_state_size

        self.fc1 = nn.Linear(self.obs_size, self.fc_size)
        self.lstm = SRNN(self.fc_size, self.lstm_state_size, self.lstm_state_size, 2, single_output=False, embedding=False)
        self.action_branch = nn.Sequential(
            nn.Linear(self.lstm_state_size, self.lstm_state_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.lstm_state_size, self.lstm_state_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.lstm_state_size, num_outputs),
        )
        self.value_branch = nn.Sequential(
            nn.Linear(self.lstm_state_size, self.lstm_state_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.lstm_state_size, self.lstm_state_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.lstm_state_size, 1),
        )
        # Holds the current "base" output (before logits layer).
        self._features = None

    @override(ModelV2)
    def get_initial_state(self):
        h = [
            self.fc1.weight.new(self.lstm_state_size).zero_()
        ]
        return h

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return torch.reshape(self.value_branch(self._features), [-1])

    @override(TorchRNN)
    def forward_rnn(self, inputs, state, seq_lens):
        x = nn.functional.relu(self.fc1(inputs))
        self._features, h = self.lstm(
            x, state[0]
        )
        action_out = self.action_branch(self._features)
        return action_out, [h]
