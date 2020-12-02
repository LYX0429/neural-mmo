#TILE_TYPES = ['grass', 'stone', 'water', 'lava', 'forest', 
#        #'tree', 'iron_ore'
#        ]
from forge.blade.lib.enums import Material
TILE_TYPES = [None for _ in range(len(Material))]
for mat in Material:
    val = mat.value
    TILE_TYPES[val.index] = val.tex
#print('pcg TILE_TYPES:', TILE_TYPES)
