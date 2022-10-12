from torch import nn
import torch
from abc import ABC, abstractmethod
from typing import Sequence


class Network(nn.Module):
    """
    I need all modules to have some universal methods.
    This class is for those methods:
    """

    def __init__(self):
        super(Network, self).__init__()
    #    self.device = torch.device('cpu')

    #def to(self, *args, **kwargs):
    #    self.device, _, _, = torch._C._nn._parse_to(*args, **kwargs)
    #    return super().to(*args, **kwargs)

    @property
    def device(self):
         return next(self.parameters()).device

    @abstractmethod
    def input_keys(self) -> Sequence[str]:
        """
        Returns a list of keys that this module processes in the input_dict
        """

    # def _th(self, data, dtype=torch.float32):
    #    return torch.as_tensor(data, device=self.device, dtype=dtype)


class RecurrentModule(Network, ABC):
    """
    Base class for LSTM based nets to use with MemUP.
    """

    @abstractmethod
    def forward(self, input_dict, mem_state, **kwargs):
        """
        :param input_dict: each key in input dict contains a
                         torch.tensor of shape (B, T, D1,...)
        :param mem_state: stores memory of an RNN
        :param kwargs: just in case idk
        :return: (hidden_state from all intermediate steps, updated memory step)
        """

        # padded_outputs, mem_state = self.process_seq(
        #     input_dict, mem_state, **kwargs
        # )
        # mem_state = self.mask_hidden_state(mem_state, input_dict['done'])
        # return padded_outputs, mem_state

    # @abstractmethod
    # def process_seq(self, input_dict, mem_state, **kwargs):

    def mask_hidden_state(self, mem_state, dones):
        dones = torch.as_tensor(
            [d[-1] for d in dones],
            dtype=torch.float32,
            device=self.device
        )
        not_dones = (1. - dones).view(1, -1, 1)
        hx, cx = mem_state
        return (hx * not_dones, cx * not_dones)

    def init_state(self, batch_size):
        shape = (self.lstm.num_layers, batch_size, self.hidden_dim)
        hx = torch.zeros(*shape, dtype=torch.float32, device=self.device)
        cx = torch.zeros(*shape, dtype=torch.float32, device=self.device)
        return (hx, cx)

    def detach_state(self, mem_state):
        return mem_state[0].detach(), mem_state[1].detach()


class PredictorModule(Network, ABC):
    """
    Base class for predictor.
    Just to specify input arguments used during MemUP training
    """
    @abstractmethod
    def forward(self, input_dict, memory_states, **kwargs):
        pass


class PredictorModuleWithContext(Network, ABC):
    """
    Base class for predictor.
    Just to specify input arguments used during MemUP training
    """
    @abstractmethod
    def forward(self, input_dict, memory_states, state, **kwargs):
        pass

    @abstractmethod
    def predict_context(self, input_dict, state, **kwargs):
        pass


#=== Some Utilitary functions used in more than one place ===

def pad_input_sequence(*input_modalities):
    """
    RNN can process several different inputs(images, actions, embeddings). Each of these inputs consists of several tensors of possibly different length.
    This functions pads those sequences. For example:
    padded_obs, padded_acts, padded_rewards = pad_input_sequences(obs, acts, rewards)

    :param input_modalities: List[List[Tensors]]
    :return: List of Tensors of size ``B x T x *``
    """
    return [
        nn.utils.rnn.pad_sequence(seqs, batch_first=True)
        for seqs in input_modalities
    ]


def flatten(*padded_tensors, num_dims=2):
    """
    Receives list of tensors and flattens first num_dims in each
    :param padded_tensors: list of tensors
    :param num_dims: number of leading dimensions to flatten
    :return: list of flattened tensors
    """
    return [t.flatten(0, num_dims-1) for t in padded_tensors]


def accumulate(src_net: nn.Module, dest_net: nn.Module, decay=0.99) -> None:
    """
    Computes a single update of exponential moving average
    for nn.Modules. i.e.:
        dest_net = decay * dest_net + (1-decay)*source_net

    :param src_net: nn.Module with new weights we want to accumulate
    :param dest_net: nn.Module that stores accumulated weights
    :return: None
    """
    params = dict(src_net.named_parameters())
    acc_params = dict(dest_net.named_parameters())
    device = dest_net.device

    for k in params.keys():
        acc_params[k].data.mul_(decay)
        acc_params[k].data += (1 - decay) * params[k].data.to(device)

#===========================================================