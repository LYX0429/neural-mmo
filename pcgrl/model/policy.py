from pdb import set_trace as T
import numpy as np

from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.policy.rnn_sequencing import add_time_dimension

import torch
from torch import nn

from forge.ethyr.torch import io
from forge.ethyr.torch import policy
from pcgrl.model import baseline

from pcgrl.game.io.action.static import actionSpace

class PCGPolicy(RecurrentNetwork, nn.Module):
   '''Wrapper class for using our baseline models with RLlib'''
   def __init__(self, *args, **kwargs):
      self.config = args[3]['custom_model_config']['config']
      #FIXME: hack
      args = (args[0], actionSpace(self.config), 10) + args[3:]
      print('pcg policy args', args)
      super().__init__(*args, **kwargs)
      nn.Module.__init__(self)
      self.space  = actionSpace(self.config).spaces

      #Select appropriate baseline model

      if self.config.MODEL == 'attentional':
         self.model  = baseline.Attentional(self.config)
      elif self.config.MODEL == 'convolutional':
         self.model  = baseline.Simple(self.config)
      else:
         self.model  = baseline.Recurrent(self.config)

   #Initial hidden state for RLlib Trainer
   def get_initial_state(self):
      return [self.model.valueF.weight.new(1, self.config.HIDDEN).zero_(),
              self.model.valueF.weight.new(1, self.config.HIDDEN).zero_()]

   def forward(self, input_dict, state, seq_lens):
      logitDict, state = self.model(input_dict['obs'], state, seq_lens)

      logits = []
#     print('pcgrl.Policy.space:\n{}'.format(self.space))
#     print('logitDict\n{}'.format(logitDict))
      #Flatten structured logits for RLlib

      for atnKey, atn in sorted(self.space.items()):
#        print('atnKey {}, atn {}'.format(atnKey, atn))
         for argKey, arg in sorted(atn.spaces.items()):
#           print('logitDict[atnKey]: \n {}'.format(logitDict[atnKey]))
            logits.append(logitDict[atnKey][argKey])

      return torch.cat(logits, dim=1), state

   def value_function(self):
      return self.model.value

   def attention(self):
      return self.model.attn




