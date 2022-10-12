from typing import Sequence

from torch import nn
import torch
import torch.nn.utils.rnn as utils_rnn
import memup.nets as nets


class SLRecMemory(nets.RecurrentModule):

    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim,
            batch_first=True, num_layers=2,
            dropout=0.1,
        )

    def input_keys(self):
        return ['x', 'done']

    def forward(self, data, mem_state, **kwargs):
        lengths = [len(e) for e in data['x']]
        x, = nets.pad_input_sequence(data['x'])

        embeds = self.embeddings(x)
        packed_embeds = utils_rnn.pack_padded_sequence(
            embeds, lengths,
            batch_first=True,
            enforce_sorted=False
        )
        outputs, mem_state = self.lstm(packed_embeds, mem_state)
        padded_outputs, lengths = utils_rnn.pad_packed_sequence(outputs, batch_first=True)

        mem_state = self.mask_hidden_state(mem_state, data['done'])
        return padded_outputs, mem_state


class SLRecPredictor(nets.RecurrentModule, nets.PredictorModuleWithContext):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_outputs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim,
            batch_first=True, num_layers=2,
            dropout=0.1,
        )
        self.memory_dim = hidden_dim
        self.context_dim = hidden_dim
        self.num_outputs = num_outputs
        self.head = nn.Sequential(
            nn.Linear(self.memory_dim + self.context_dim + self.embedding_dim, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, self.num_outputs)
        )

    def input_keys(self):
        return ['x', 'context', 'done']

    def predict_context(self, data, state, **kwargs):
        lengths = [len(e) for e in data['x']]
        x, = nets.pad_input_sequence(data['x'])
        if len(x.shape) == 3:
            x = x[:, :, 0]

        embeds = self.embeddings(x)
        packed_embeds = utils_rnn.pack_padded_sequence(
            embeds, lengths,
            batch_first=True,
            enforce_sorted=False
        )
        outputs, new_state = self.lstm(packed_embeds, state)
        padded_outputs, lengths = utils_rnn.pad_packed_sequence(outputs, batch_first=True)

        new_state = self.mask_hidden_state(new_state, data['done'])
        return padded_outputs, new_state

    def forward(self, data, mem_output, state=None, **kwargs):

        x = torch.cat(data['x'])
        embeds = self.embeddings(x)

        new_state = None
        if state is None:
            context = torch.cat(data['context'])
        else:
            context, new_state = self.predict_context(data, state)
            lengths = [len(e) for e in data['x']]
            context = torch.cat([context[i, 0:lengths[i]] for i in range(len(lengths))])

        pred_input = torch.cat([embeds, mem_output, context], dim=-1)
        predictions = self.head(pred_input)
        return predictions, context, new_state


class SLPredictor(nets.PredictorModule):

    def __init__(self, memory_dim, context_dim, vocab_size, num_outputs, hidden_size=256):
        super(SLPredictor, self).__init__()
        self.memory_dim = memory_dim
        self.context_dim = context_dim
        self.num_outputs = num_outputs
        self.hidden_size = hidden_size

        self.context_encoder = nn.Embedding(vocab_size, context_dim)
        self.mlp = self._create_mlp()

    def _create_mlp(self):
        return nn.Sequential(
            nn.Linear(self.memory_dim+self.context_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_outputs)
        )

    def input_keys(self):
        return ['x']

    def forward(self, input_dict, memory_states, **kwargs):
        context_input = torch.cat(input_dict['x'])
        context = self.context_encoder(context_input)
        pred_input = torch.cat([memory_states, context], dim=-1)
        predictions = self.mlp(pred_input)
        return predictions


class SLPredictorWithShortMemory(SLPredictor):

    def __init__(self, memory_dim, context_dim, vocab_size, num_outputs, hidden_size=256):
        super().__init__(memory_dim * 2, context_dim, vocab_size, num_outputs, hidden_size)

    def input_keys(self):
        return ['x', 'mem_context']

    def forward(self, input_dict, memory_states, **kwargs):
        context_input = torch.cat(input_dict['x'])
        context = self.context_encoder(context_input)
        context_mem = torch.cat(input_dict['mem_context'])
        pred_input = torch.cat([memory_states, context, context_mem], dim=-1)
        predictions = self.mlp(pred_input)
        return predictions