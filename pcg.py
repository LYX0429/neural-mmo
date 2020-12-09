#TILE_TYPES = ['grass', 'stone', 'water', 'lava', 'forest', 
#        #'tree', 'iron_ore'
#        ]
from forge.blade.lib.enums import Material
TILE_PROB_DICT = {
      'grass':     0.34,
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
   if val.tex == 'lava':
      continue
   TILE_TYPES[val.index] = val.tex
   TILE_PROBS[val.index] = TILE_PROB_DICT[val.tex]
#print('pcg TILE_TYPES:', TILE_TYPES)
