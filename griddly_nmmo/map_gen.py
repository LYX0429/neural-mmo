from pdb import set_trace as T
from imageio import imread, imsave
from enum import Enum
import griddly
from forge.blade.core.terrain import MapGenerator
import os
import numpy as np
import yaml
from collections import OrderedDict
np.set_printoptions(threshold=5000, linewidth=200)

TILE_PROB_DICT = OrderedDict({
               'lava':     1,
               'grass':    3,
               'water':    2,
               'forest':   5.,
               'stone':    1,
               'iron_ore': 1,
               'tree':     1,
               'gnome_spawn': 0.025,
              #'chicken_spawn': 0.005
            })
temp = np.sum([v for v in TILE_PROB_DICT.values()])
[TILE_PROB_DICT.update({k: v/temp}) for k, v in TILE_PROB_DICT.items()]

class Tile():
   pass 

class Terrain:
   pass


class Lava(Tile):
   index = 0
   tex = 'lava'
class Water(Tile):
   index = 2
   tex = 'water'
class Grass(Tile):
   index = 1
   tex = 'grass'
#class Scrub(Tile):
#   index = 3
#   tex = 'scrub'
class Forest(Tile):
   index = 3
#     degen = self.Scrub
   tex = 'forest'
   #capacity = 3
   capacity = 1
   respawnProb = 0.025
   def __init__(self):
      super().__init__()
      self.harvestable = True
      #self.dropTable = DropTable.DropTable()
class Stone(Tile):
   index = 4
   tex = 'stone'
class Orerock(Tile):
   index = 5
#     degen = Grass
   tex = 'iron_ore'
   capacity = 1
   respawnprob = 0.025
   def __init__(self):
      super().__init__()
      self.harvestable = True
      #self.dropTable = systems.DropTable()
      #self.dropTable.add(ore.Copper, 1)
class Tree(Tile):
   index = 6
  #degen = Forest
   tex = 'tree'
   capacity = 1
   respawnProb = 0.025
   def __init__(self):
      super().__init__()
      self.harvestable = True
class Spawn(Tile):
   index = 7
   tex = 'spawn'

class GdyMaterial(Enum):


   LAVA     = Lava
   WATER    = Water
   GRASS    = Grass
#  SCRUB    = Scrub
   FOREST   = Forest
   STONE    = Stone
   OREROCK  = Orerock
   TREE     = Tree
   SPAWN    = Spawn

class GriddlyMapGenerator(MapGenerator):
   def loadTextures(self):
      lookup = {}
      for mat in GdyMaterial:
         mat = mat.value
         tex = imread(
               'resource/assets/tiles/' + mat.tex + '.png')
         key = mat.tex
         mat.tex = tex[:, :, :3][::4, ::4]
         lookup[mat.index] = mat.tex
         setattr(Terrain, key.upper(), mat.index)
      self.textures = lookup


def replace_vars(yaml_contents, var_dict):
    if isinstance(yaml_contents, dict):
        for key, val in yaml_contents.items():
            new_entry = replace_vars(val, var_dict)
            if new_entry is not None:
                yaml_contents[key] = new_entry
    elif isinstance(yaml_contents, list):
        for el in yaml_contents:
            replace_vars(el, var_dict)
    elif isinstance(yaml_contents, str):
        if yaml_contents in var_dict:
            return var_dict[yaml_contents]
        else:
            return None
    else:
        pass
       #raise Exception('Unexpected type, {}, while parsing yaml.'.format(
       #    type(yaml_contents)))

class MapGen():
    INIT_DELAY = 0 # how long to wait before decrementing hunger & thirst
    INIT_HEALTH = -1
    INIT_THIRST = 0
    INIT_HUNGER = 0
    SHRUB_RESPAWN = 30


    def __init__(self, config=None):
        if config is not None:
            self.N_PLAYERS = config.NENT
            self.MAP_WIDTH = config.TERRAIN_SIZE
        else:
            self.N_PLAYERS = 3
            self.MAP_WIDTH = 20
        self.VAR_DICT = {
                '${_init_delay}':    self.INIT_DELAY * self.N_PLAYERS,
                '${_delay}':         1,# * self.N_PLAYERS,
                '${_init_health}':   self.INIT_HEALTH,
                '${_init_hunger}':   self.INIT_HUNGER,
                '${_init_thirst}':   self.INIT_THIRST,
                '${_shrub_respawn}': self.SHRUB_RESPAWN * self.N_PLAYERS,
                }
        self.probs = TILE_PROB_DICT
        # self.probs = {
      #         'grass':  0.80,
      #         'water':  0.00,
      #         'shrub':  0.00,
      #         'rock':   0.00,
      #         'lava':   0.20,
      #         }

        self.chars = {
                'grass': '.'
                }
        self.border_tile = 'lava'

    def get_init_tiles(self, yaml_path, write_game_file=False):
        # Using a template to generate the runtime file allows for preservation of comments and structure. And possibly other tricks... (evolution of game entities and mechanics)
        yaml_path_og = os.path.join(griddly.__path__[0], 'resources', 'games',  'nmmo.yaml')
       #yaml_path = os.path.join('griddly_nmmo',  yaml_path)
        yaml_template_path = yaml_path.strip('.yaml') + '_template.yaml'
        yaml_template_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), yaml_template_path)
        init_tiles = [self.chars['grass']]
        probs = [self.probs['grass']]
        self.tile_types = tile_types = list(self.probs.keys())
        with open(yaml_template_path) as f:
            contents = yaml.load(f, Loader=yaml.FullLoader)
        self.utf_enc = 'U' + str(len(str(self.N_PLAYERS)) + 1)
        objects = contents['Objects']
        for obj in objects:
            obj_name = obj['Name']
            if obj_name in tile_types:
                char = obj['MapCharacter']
                init_tiles.append(char)
                probs.append(self.probs[obj_name])
                self.chars[obj_name] = char
            if obj['Name'] == 'gnome':
                self.player_char = obj['MapCharacter']#+ '1'
        assert hasattr(self, 'player_char')
        init_tiles.append(self.player_char)
        probs.append(self.probs['gnome_spawn'])
        # Add a placeholder level so that we can make the env from yaml (this will be overwritten during reset)
        level_string = self.gen_map(init_tiles, probs)
        print(level_string)
        contents['Environment']['Levels'] = [level_string] # placeholder map
        contents['Environment']['Player']['Count'] = self.N_PLAYERS # set num players
        contents['Environment']['Name'] = 'nmmo'
        skills = []
        for var_dict in contents['Environment']['Variables']:
           var_name = var_dict['Name']
           if 'skill' in var_name:
              skills.append(var_name)
        #HACK: scale delays to num players
        if write_game_file:
           replace_vars(contents, self.VAR_DICT)
           #FIXME: sloppy redundancy
           with open(yaml_path, 'w') as f:
               yaml.safe_dump(contents, f, default_style=None, default_flow_style=False)
           with open(yaml_path_og, 'w') as f:
               print('saving nmmo game mechanics at {}'.format(yaml_path_og))
               yaml.safe_dump(contents, f, default_style=None, default_flow_style=False)

        return init_tiles, probs, skills

    def gen_map(self, init_tiles, probs):
        # max 3 character string in each tile
        # need to take into account column of newlines
        # we'll add spawns later
        if self.player_char in init_tiles:
            spawn_idx = init_tiles.index(self.player_char)
            # NB: We're actually popping from object attributes here
            init_tiles.pop(spawn_idx)
            probs.pop(spawn_idx)
        probs = np.array(probs) / np.sum(probs)
        level_string = np.random.choice(init_tiles, size=(self.MAP_WIDTH, self.MAP_WIDTH+1), p=probs).astype(self.utf_enc)
#       idxs = np.where(level_string[1:-1, 1:-2] == self.player_char)
        idxs = np.where(level_string[1:-1, 1:-2] == ".")
        idxs = np.array(list(zip(idxs[0], idxs[1]))) + 1
        ixs = np.random.choice(len(idxs), min(self.N_PLAYERS, len(idxs)), replace=False)
        coords = idxs[ixs]
        border_tile = self.border_tile
        level_string[0, :] = self.chars[border_tile]
        level_string[-1, :] = self.chars[border_tile]
        level_string[:, 0] = self.chars[border_tile]
        level_string[:, -2] = self.chars[border_tile]
        for j, coord in enumerate(coords[:self.N_PLAYERS]):
            level_string[coord[0], coord[1]] = self.player_char + str(j+1)
#       level_string[coords[:, 0], coords[:, 1]] = self.player_char

        level_string[:, -1] = '\n'
        level_string = ' '.join(s for s in level_string.reshape(-1))
        return level_string[:-2]

