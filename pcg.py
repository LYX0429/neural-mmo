#TILE_TYPES = ['grass', 'stone', 'water', 'lava', 'forest', 
#        #'tree', 'iron_ore'
#        ]
from forge.blade.lib import material
from collections import OrderedDict

def get_tile_data(griddly=False):
   if not griddly:
      TILE_PROB_DICT = {
#           'lava':      0.01,
            'grass':     0.32,
            'stone':     0.1,
            'water':     0.2,
            'scrub':     0.1,
            'forest':    0.2,
            'tree':      0.03,
            'ore':       0.03,
            'spawn':     0.01,
            'crystal':   0.01,
            'stump':     0.01,
            'fish':      0.01,
            'ocean':     0.01,
            'weeds':     0.01,
            'slag':      0.01,
            'fragment':  0.01,
          'herb': 0.01,
      }
     #TILE_TYPES = [None for _ in range(len(material.All.materials))]
     #TILE_PROBS = [None for _ in range(len(material.All.materials))]
      TILE_TYPES = [None for _ in range(len(TILE_PROB_DICT))]
      TILE_PROBS = [None for _ in range(len(TILE_PROB_DICT))]
      for mat in material.All.materials:
         val = mat
         # no inland lava allowed
       # if val.tex == 'lava':
       #    continue
         if val.index in TILE_PROB_DICT:
             TILE_TYPES[val.index] = val.tex
             TILE_PROBS[val.index] = TILE_PROB_DICT[val.tex]
      #print('pcg TILE_TYPES:', TILE_TYPES)
   else:
      from griddly_nmmo.map_gen import TILE_PROB_DICT
      TILE_TYPES = [k for k in TILE_PROB_DICT]
      TILE_PROBS = [TILE_PROB_DICT[tile_type] for tile_type in TILE_TYPES]
   return TILE_TYPES, TILE_PROBS

