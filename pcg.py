#TILE_TYPES = ['grass', 'stone', 'water', 'lava', 'forest', 
#        #'tree', 'iron_ore'
#        ]
from forge.blade.lib.enums import Material
from collections import OrderedDict

def get_tile_data(griddly=False):
   if not griddly:
      TILE_PROB_DICT = {
            'lava':      0.01,
            'grass':     0.32,
            'stone':     0.1,
            'water':     0.2,
            'scrub':     0.1,
            'forest':    0.2,
            'tree':      0.03,
            'iron_ore':  0.03,
            'spawn':     0.01,
              }
      TILE_TYPES = [None for _ in range(len(Material))]
      TILE_PROBS = [None for _ in range(len(Material))]
      for mat in Material:
         val = mat.value
         # no inland lava allowed
       # if val.tex == 'lava':
       #    continue
         TILE_TYPES[val.index] = val.tex
         TILE_PROBS[val.index] = TILE_PROB_DICT[val.tex]
      #print('pcg TILE_TYPES:', TILE_TYPES)
   else:
      TILE_PROB_DICT = OrderedDict({
            'lava':      0.01,
            'grass':     0.49,
            'water':     0.2,
            'shrub':     0.3,
            })
      TILE_TYPES = [k for k in TILE_PROB_DICT]
      TILE_PROBS = [TILE_PROB_DICT[tile_type] for tile_type in TILE_TYPES]
   return TILE_TYPES, TILE_PROBS

