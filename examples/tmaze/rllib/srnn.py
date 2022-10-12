import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.jit as jit
from typing import Optional, Tuple
import numpy as np
from torch.utils.cpp_extension import load
# import srnn

# load(name='srnn', sources=['/home/jovyan/projects/uncertaint_events_predictor/examples/supervised/models/cuda/srnn.cpp',
#                            '/home/jovyan/projects/uncertaint_events_predictor/examples/supervised/models/cuda/srnn_kernels.cu'], is_python_module=True,
#      verbose=True)


class SRNNCell(nn.Module):
    __constants__ = ['do_embedding', 'single_output', 'multihead']

    def __init__(self, input_size, hidden_size, num_layers=1, hyper_size=64, **kwargs):
        super().__init__()
        l_list = [nn.Linear(input_size, hyper_size), nn.ReLU()]
        for i in range(1, num_layers):
            l_list.extend([nn.Linear(hyper_size, hyper_size), nn.ReLU()])
        l_list.append(nn.Linear(hyper_size, hidden_size))
        self.fc = nn.Sequential(*l_list)

        self.fc2 = nn.Linear(input_size, hidden_size)
        if 'multihead' not in kwargs:
            self.multihead = True
        else:
            self.multihead = kwargs['multihead']

    def forward(self, x, hidden: Optional[torch.Tensor] = None):
        if len(x.shape) == 2:
            batch_size, seq_len = x.shape
        else:
            batch_size, seq_len, inp_size = x.shape

        b = self.fc(x)

        if self.multihead:
            sig_alphas = torch.sigmoid(self.fc2(x))
            b = b * sig_alphas

        if hidden is not None:
            outputs = [torch.relu(b[:, 0] + torch.roll(hidden, 1, -1))]
        else:
            outputs = [torch.relu(b[:, 0])]

        for i in range(1, seq_len):
            outputs.append(torch.relu(b[:, i] + torch.roll(outputs[-1], 1, -1)))

        outputs = torch.stack(outputs, 1)
        hidden = outputs[:, -1, :]

        outputs = outputs.squeeze(2)

        return outputs, hidden


# class cudasumrelushift(torch.autograd.Function):
#
#     @staticmethod
#     def forward(ctx, b, hidden):
#         out = srnn.roll_sum_relu(b, hidden)
#         ctx.save_for_backward(b, hidden, out)
#         return out
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         b, hidden, out = ctx.saved_tensors
#         grad_input = srnn.calc_roll_grad(grad_output, out)
#         return grad_input, None


# class SRNNCellFast(torch.nn.Module):
#
#     def __init__(self, input_size, hidden_size, num_layers=1, hyper_size=64, **kwargs):
#         super().__init__()
#         self.fc_start = nn.Linear(input_size, hyper_size)
#         self.fc_list = torch.nn.ModuleList([nn.Linear(hyper_size, hyper_size) for _ in range(num_layers - 1)])
#         self.fc_end = nn.Linear(hyper_size, hidden_size)
#
#         self.fc2 = nn.Linear(input_size, hidden_size)
#
#         # grad
#         self.d_fc_w = [torch.Tensor() for _ in self.fc_list]
#         self.d_fc_b = [torch.Tensor() for _ in self.fc_list]
#
#         # weights
#         self.fc_w = [fc.weight for fc in self.fc_list]
#         self.fff = cudasumrelushift.apply
#
#     def forward(self, x, hidden: torch.Tensor):
#         # Apply fc.
#         fc_results = []
#         fc_results.append(F.relu(self.fc_start(x)))
#         for fc in self.fc_list:
#             fc_results.append(F.relu(fc(fc_results[-1])))
#         fc_end_result = self.fc_end(fc_results[-1])
#         sig_alphas_result = torch.sigmoid(self.fc2(x))
#         b = fc_end_result * sig_alphas_result
#
#         # outputs = torch.ops.srnn.roll_sum_relu(b, hidden)
#         outputs = self.fff(b, hidden)
#         outputs_results = outputs
#         hidden = outputs[:, -1, :]
#
#         return outputs, hidden


class SRNN(nn.Module):
    # class SRNN(nn.Module):
    __constants__ = ['do_embedding', 'single_output']

    def __init__(self, input_size, output_size, hidden_size, num_layers, *args, **kwargs):
        super().__init__()
        self.single_output = kwargs['single_output']
        if kwargs['embedding']:
            self.embedding = nn.Embedding(input_size, kwargs['hyper_size'])
            self.rnncell = SRNNCell(kwargs['hyper_size'], hidden_size, num_layers, **kwargs)
            self.do_embedding = True
        else:
            self.embedding = nn.Embedding(1, 1)
            self.rnncell = SRNNCell(input_size, hidden_size, num_layers, **kwargs)
            self.do_embedding = False

        self.end_fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden: Optional[torch.Tensor] = None):

        if self.do_embedding:
            x = self.embedding(x.squeeze(-1))

        outputs, hidden = self.rnncell(x, hidden)

        outputs = self.end_fc(outputs)
        if self.single_output:
            outputs = outputs[:, -1, :].unsqueeze(1)

        return outputs, hidden


# class SRNNFast(torch.nn.Module):
#     __constants__ = ['do_embedding', 'single_output']
#
#     def __init__(self, input_size, output_size, hidden_size, num_layers, *args, **kwargs):
#         super().__init__()
#         self.single_output = kwargs['single_output']
#         if kwargs['embedding']:
#             # CHECK THIS MOD
#             self.embedding = nn.Embedding(input_size, kwargs['hyper_size'])
#             self.rnncell = SRNNCellFast(kwargs['hyper_size'], hidden_size, num_layers, **kwargs)
#             self.do_embedding = True
#         else:
#             self.embedding = nn.Embedding(1, 1)
#             self.rnncell = SRNNCellFast(input_size, hidden_size, num_layers, **kwargs)
#             self.do_embedding = False
#
#         self.end_fc = nn.Linear(hidden_size, output_size)
#         self.hidden_size = hidden_size
#         self.last_outputs = torch.Tensor()
#
#     def forward(self, x, hidden: Optional[torch.Tensor] = None):
#         if self.do_embedding:
#             x = self.embedding(x.squeeze(-1))
#
#         if hidden is None:
#             hidden = torch.zeros(x.size(0), self.hidden_size, device=x.device)
#
#         outputs, hidden = self.rnncell(x, hidden)
#
#         self.last_outputs = outputs[:, -1, :]
#         outputs = self.end_fc(outputs)
#         if self.single_output:
#             outputs = outputs[:, -1, :].unsqueeze(1)
#
#         return outputs, hidden
#
#     def backward(self, loss, x):
#         self.end_fc.bias.grad = sum(loss.squeeze(), 0)
#         self.end_fc.weight.grad = torch.sum(
#             torch.bmm(loss.view(-1, loss.size(2), 1), self.last_outputs.view(-1, 1, self.last_outputs.size(1))), 0)
#         loss = torch.matmul(loss, self.end_fc.weight)
#         self.rnncell.backward(loss, x)



if __name__ == '__main__':
    # hr = HyperRNN(2, 128, 1).cuda()
    x = torch.randn(21, 5, 2).cuda()
    # print(hr(x.cuda()))
    model = SRNN(2, 3, 30, 2, single_output=False, embedding=False).cuda()
    # model2 = SRNNFast(2, 3, 30, 2, single_output=False, embedding=False).cuda()
    y, s = model(x)
    print(s.shape)
