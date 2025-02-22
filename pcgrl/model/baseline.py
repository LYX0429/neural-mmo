import torch
from torch import nn
from torch.nn.utils import rnn

from forge.blade.io.stimulus.static import Stimulus
from forge.ethyr.torch import policy
from pcgrl.model.torch.io import Input, Output


class Base(nn.Module):
    def __init__(self, config):
        '''Base class for baseline policies

        Args:
           config: A Configuration object
        '''
        super().__init__()
        self.embed = config.EMBED
        self.config = config

        self.output = Output(config)
        #print('base output:{}'.format(self.output))
        self.input = Input(config,
                                       embeddings=policy.BiasedInput,
                                       attributes=policy.Attention)

        self.valueF = nn.Linear(config.HIDDEN, 1)

    def hidden(self, obs, state=None, lens=None):
        '''Abstract method for hidden state processing, recurrent or otherwise,
        applied between the input and output modules

        Args:
           obs: An observation dictionary, provided by forward()
           state: The previous hidden state, only provided for recurrent nets
           lens: Trajectory segment lengths used to unflatten batched obs
        '''
        raise NotImplementedError('Implement this method in a subclass')

    def forward(self, obs, state=None, lens=None):
        '''Applies builtin IO and value function with user-defined hidden
        state subnetwork processing. Arguments are supplied by RLlib
        '''
        entityLookup = self.input(obs)
        hidden, state = self.hidden(entityLookup, state, lens)
        self.value = self.valueF(hidden).squeeze(1)
        actions = self.output(hidden, entityLookup)

        return actions, state


class Simple(Base):
    def __init__(self, config):
        '''Simple baseline model with flat subnetworks'''
        super().__init__(config)
        h = config.HIDDEN

        self.conv = nn.Conv2d(h, h, 3)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(h * 3 * 3, h)

        self.proj = nn.Linear(2 * h, h)
        self.attend = policy.Attention(self.embed, h)

    def hidden(self, obs, state=None, lens=None):
        # Attentional agent embedding
        agents, _ = self.attend(obs[Stimulus.Entity])

        # Convolutional tile embedding
        tiles = obs[Stimulus.Tile]
        self.attn = torch.norm(tiles, p=2, dim=-1)

        w = self.config.WINDOW
        batch = tiles.size(0)
        hidden = tiles.size(2)
        # Dims correct?
        tiles = tiles.reshape(batch, w, w, hidden).permute(0, 3, 1, 2)
        tiles = self.conv(tiles)
        tiles = self.pool(tiles)
        tiles = tiles.reshape(batch, -1)
        tiles = self.fc(tiles)

        hidden = torch.cat((agents, tiles), dim=-1)
        hidden = self.proj(hidden)

        return hidden, state


class Recurrent(Simple):
    def __init__(self, config):
        '''Recurrent baseline model'''
        super().__init__(config)
        self.lstm = policy.BatchFirstLSTM(input_size=config.HIDDEN,
                                          hidden_size=config.HIDDEN)

    # Note: seemingly redundant transposes are required to convert between
    # Pytorch (seq_len, batch, hidden) <-> RLlib (batch, seq_len, hidden)
    def hidden(self, obs, state, lens):
        # Attentional input preprocessor and batching
        hidden, _ = super().hidden(obs)
        config = self.config
        h, c = state

        TB = hidden.size(0)  # Padded batch of size (seq x batch)
        B = len(lens)  # Sequence fragment time length
        T = TB // B  # Trajectory batch size
        H = config.HIDDEN  # Hidden state size

        # Pack (batch x seq, hidden) -> (batch, seq, hidden)
        hidden = rnn.pack_padded_sequence(input=hidden.view(B, T, H),
                                          lengths=lens,
                                          enforce_sorted=False,
                                          batch_first=True)

        # Main recurrent network
        hidden, state = self.lstm(hidden, state)

        # Unpack (batch, seq, hidden) -> (batch x seq, hidden)
        hidden, _ = rnn.pad_packed_sequence(sequence=hidden, batch_first=True)

        return hidden.reshape(TB, H), state


class Attentional(Base):
    def __init__(self, config):
        '''Transformer-based baseline model'''
        super().__init__(config)
        self.agents = nn.TransformerEncoderLayer(d_model=config.HIDDEN,
                                                 nhead=4)
        self.tiles = nn.TransformerEncoderLayer(d_model=config.HIDDEN, nhead=4)
        self.proj = nn.Linear(2 * config.HIDDEN, config.HIDDEN)

    def hidden(self, obs, state=None, lens=None):
        # Attentional agent embedding
        agents = self.agents(obs[Stimulus.Entity])
        agents, _ = torch.max(agents, dim=-2)

        # Attentional tile embedding
        tiles = self.tiles(obs[Stimulus.Tile])
        self.attn = torch.norm(tiles, p=2, dim=-1)
        tiles, _ = torch.max(tiles, dim=-2)

        hidden = torch.cat((tiles, agents), dim=-1)
        hidden = self.proj(hidden)

        return hidden, state
