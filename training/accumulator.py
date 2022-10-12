import copy
from typing import Generic, TypeVar

from torch import nn


T = TypeVar("T", bound=nn.Module)


class Accumulator(nn.Module, Generic[T]):

    def __init__(self, module: T, module_acc: T, decay=0.9):
        super().__init__()

        self.module_acc = module_acc
        self.module_acc.load_state_dict(module.state_dict())
        self.decay = decay

    def accumulate(self, module):
        params = dict(module.named_parameters())
        acc_params = dict(self.module_acc.named_parameters())

        for k in params.keys():
            acc_params[k].data.mul_(self.decay)
            acc_params[k].data += (1 - self.decay) * params[k].data

    def get_module(self) -> T:
        return self.module_acc
