import neat
from numpy.random import default_rng
import neat.nn
import torch
from torch.nn import Conv2d
import copy
import numpy as np
from forge.blade.lib import enums
from pdb import set_trace as TT
from qdpy.phenotype import Individual, Fitness, Features
from evolution.paint_terrain import PRIMITIVE_TYPES
import vec_noise
from pytorch_neat.cppn import create_cppn
# from numba import njit
from typing import Any
import itertools
import gc

# Not using this
class SpawnPoints():
    def __init__(self, map_width, n_players):
        self.max_spawns = n_players * 3
        n_spawns = np.random.randint(n_players, self.max_spawns)
        self.spawn_points = np.random.randint(0, map_width (2, n_spawns))

    def mutate(self):
        n_spawns = len(self.spawn_points)
        n_delete = np.random.randint(0, 5)
        n_add = np.random.randint(0, 5)

# Not using these currently
MELEE_MIN = 0.4
MELEE_MAX = 1.4
MAGE_MIN = 0.6
MAGE_MAX = 1.6
RANGE_MIN = 0.2
RANGE_MAX = 1

# This is busted do not use! Maybe this tie-breaking happens elsewhere in qdpy anyway?
#class EvoFitness(Fitness):
#   ''' Slight modification to allow for random drift.'''
#   def __init__(self, *args, **kwargs):
#      super().__init__(*args, **kwargs)
#
#   def dominates(self, other: Any, obj: Any = slice(None)) -> bool:
#      # FROM DEAP
#      def dominates(self, other: Any, obj: Any = slice(None)) -> bool:
#         """Return true if each objective of ``self`` is not strictly worse than
#         the corresponding objective of ``other`` and at least one objective is
#         strictly better.
#         """
#         #    :param obj: Slice indicating on which objectives the domination is
#         #                tested. The default value is `slice(None)`, representing
#         #                every objectives.  """
#         not_equal: bool = False
#         for self_wvalue, other_wvalue in zip(self.wvalues[obj], other.wvalues[obj]):
#            if self_wvalue > other_wvalue:
#               not_equal = True
#            elif self_wvalue < other_wvalue:
#               return False
#         # Here we randomly pick a winner if the two are otherwise tied
#         if not_equal:
#            return not_equal
#         return np.random.random() < 0.5

# Not using this attack-multiplier stuff atm. Different policies will prioritize different attacks now, but still don't
# observe a single policy learning to use a diversity of attacks...
#TODO: move these back into evolver class?
def gen_atk_mults():
    # generate melee, range, and mage attack multipliers for automatic game-balancing

    atk_mults = {
        # b/w 0.2 and 1.0
        'MELEE_MULT': np.random.random() * (MELEE_MAX - MELEE_MIN) + MELEE_MIN,
        'MAGE_MULT': np.random.random() * (MAGE_MAX - MAGE_MIN) + MAGE_MIN,
        # b/w 0.0 and 0.8
        'RANGE_MULT': np.random.random() * (RANGE_MAX - RANGE_MIN) + RANGE_MIN,
    }

    return atk_mults

def mutate_atk_mults(atk_mults):
    rand = np.random.random()

    if rand < 0.2:
        atk_mults = gen_atk_mults()
    else:
        atk_mults = {
            'MELEE_MULT': max(min(atk_mults['MELEE_MULT'] + (np.random.random() * 2 - 1) * 0.3,
                                  MELEE_MAX), MELEE_MIN),
            'MAGE_MULT': max(min(atk_mults['MAGE_MULT'] + (np.random.random() * 2 - 1) * 0.3,
                                 MAGE_MAX), MAGE_MIN),
            'RANGE_MULT': max(min(atk_mults['RANGE_MULT'] + (np.random.random() * 2 - 1) * 0.3,
                                  RANGE_MAX), RANGE_MIN),
        }

    return atk_mults


def mate_atk_mults(atk_mults_0, atk_mults_1, single_offspring=False):
    new_atk_mults_0, new_atk_mults_1 = {}, {}

    for k, v in atk_mults_0.items():
        if np.random.random() < 0.5:
            new_atk_mults_0[k] = atk_mults_1[k]
        else:
            new_atk_mults_0[k] = atk_mults_0[k]

        if single_offspring:
            continue

        if np.random.random() < 0.5:
            new_atk_mults_1[k] = atk_mults_0[k]
        else:
            new_atk_mults_1[k] = atk_mults_1[k]

    return new_atk_mults_0, new_atk_mults_1

class Genome():
    def __init__(self, n_tiles, map_width):
        self.multi_hot = None
        self.map_arr = None
        self.n_tiles = n_tiles
        self.map_width = map_width
        self.atk_mults = gen_atk_mults()

    def clone(self):
        child = copy.deepcopy(self)

        return child

    def mutate(self):
        self.atk_mults = mutate_atk_mults(self.atk_mults)

    def validate_map(self, *args):
        pass

    def get_iterable(self):
        return []

def initialize_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

    if type(m) == torch.nn.Conv2d:
        torch.nn.init.orthogonal_(m.weight)

class NeuralCA(torch.nn.Module):

    N_HIDDEN = 32
    N_CHAN = 9
#   N_WEIGHTS = 2 * N_HIDDEN * N_CHAN * 3 * 3 + N_HIDDEN + 2 * N_HIDDEN + N_HIDDEN + N_HIDDEN * N_CHAN + N_CHAN + 1
    def __init__(self, n_chan):
        #FIXME
#       print('NeuralCA has {} input channels'.format(n_chan))
        assert self.N_CHAN == n_chan
        super().__init__()
        m = self.N_HIDDEN
        self.l0 = Conv2d(n_chan, 2 * m, 3, 1, 1, bias=True)
        self.l1 = Conv2d(2 * m, m, 1, 1, 0, bias=True)
        self.l2 = Conv2d(m, n_chan, 1, 1, 0, bias=True)
        self.layers = [self.l0, self.l1, self.l2]
        self.n_passes = np.random.randint(10, 100)
        self.weights = self.get_weights()
        self.apply(initialize_weights)

    def forward(self, x):
       x = torch.Tensor(x).unsqueeze(0)
       with torch.no_grad():
          for _ in range(max(1, int(min(200, self.n_passes)))):
             x = self.l0(x)
             x = torch.nn.functional.relu(x)
             x = self.l1(x)
             x = torch.nn.functional.relu(x)
             x = self.l2(x)
             x = torch.sigmoid(x)
             tile_idxs = x.argmax(axis=1).unsqueeze(1)
             x = torch.FloatTensor(x).zero_()
             x.scatter_(1, tile_idxs, 1)

       return x

    def set_weights(self, weights):
        n_el = 0
        self.weights = weights

        for layer in self.layers:
            l_weights = weights[n_el:n_el + layer.weight.numel()]
            n_el += layer.weight.numel()
            l_weights = l_weights.reshape(layer.weight.shape)
            layer.weight = torch.nn.Parameter(torch.Tensor(l_weights))
            layer.weight.requires_grad = False
            b_weights = weights[n_el:n_el + layer.bias.numel()]
            n_el += layer.bias.numel()
            b_weights = b_weights.reshape(layer.bias.shape)
            layer.bias = torch.nn.Parameter(torch.Tensor(b_weights))
            layer.bias.requires_grad = False
       #self.n_passes = max(1, weights[n_el])
#       print('Neural CAL n_passes', self.n_passes)

    def get_weights(self):
        weights = []
        #   n_par = 0
        for lyr in self.layers:
            #       n_par += np.prod(lyr.weight.shape)
            #       n_par += np.prod(lyr.bias.shape)
            weights.append(lyr.weight.view(-1).detach().numpy())
            weights.append(lyr.bias.view(-1).detach().numpy())
        init_weights = np.hstack(weights)

        return init_weights

class CAGenome(Genome):
    def __init__(self, n_tiles, map_width, seed):
       self.map_width = map_width
       self.nn = NeuralCA(n_tiles)
#      self.seed = seed.reshape(1, *seed.shape)
       if seed is None:
          seed = np.zeros((n_tiles, map_width, map_width))
          x = map_width // 2
          y = map_width // 2
          seed[1, x - 5: x + 5, y - 5:y + 5] = 1
         #seed = np.random.randint(0, 1, (n_tiles, map_width, map_width))
         #seed = np.random.randint(0, 1, (n_tiles, map_width, map_width))
       self.seed = seed
       self.atk_mults = gen_atk_mults()
       self.age = -1
       self.epsilon = 0.05
       self.rng = default_rng()

    def gen_map(self):
       nn_out = self.nn(self.seed)
       self.multi_hot = nn_out.squeeze(0).detach().numpy()
       self.map_arr = self.multi_hot.argmax(axis=0)


    def validate_map(self, SPAWN_IDX, FOOD_IDX, WATER_IDX, N_ENT):
        map_width = self.map_arr.shape[0]
        spawn_idxs = np.where(self.map_arr == SPAWN_IDX)[0]
        food_idxs = np.where(self.map_arr == FOOD_IDX)[0]
        water_idxs = np.where(self.map_arr == WATER_IDX)[0]
        n_spawns = spawn_idxs.shape[0]
        if n_spawns < N_ENT:
            self.map_arr[
                np.random.randint(0, map_width, N_ENT), np.random.randint(0, map_width, N_ENT)] = SPAWN_IDX
        if food_idxs.shape[0] < N_ENT:
            self.map_arr[
                np.random.randint(0, map_width, N_ENT), np.random.randint(0, map_width, N_ENT)] = FOOD_IDX
        if water_idxs.shape[0] < N_ENT:
            self.map_arr[
                np.random.randint(0, map_width, N_ENT), np.random.randint(0, map_width, N_ENT)] = WATER_IDX

    def mutate(self):

        noise = np.random.random(self.nn.weights.shape) * self.epsilon - (self.epsilon / 2)
        n_mut = max(1, int(self.rng.exponential(scale=1.0, size=1)))
        x = np.argwhere(noise)
        idxs = np.random.choice(x.shape[0], n_mut, replace=False)
        x = x[idxs]
        weights = self.nn.get_weights()
        weights[x] = noise[x]
#       mask = (np.random.random(size=(self.nn.weights.shape)) < 5/noise.size) * 1
#       mutation = np.multiply(mask, noise)
#       weights = self.nn.weights + mutation
        self.nn.n_passes += int(np.random.uniform(-2.5, 2.5, 1)[0])
        self.nn.n_passes = int(max(1, max(self.nn.n_passes, 100)))
        self.nn.set_weights(weights)
        self.atk_mults = mutate_atk_mults(self.atk_mults)
        self.age = -1

    def clone(self):
        child = copy.deepcopy(self)
        child.age = -1

        return child

    def get_iterable(self):
        return np.hstack((self.nn.weights, [self.nn.n_passes]))


# Actually CPPNGenome, but needs to be named this way for neat-python to stomach it? Whack. Should hack out neat-python
# and make it our own
class DefaultGenome(neat.genome.DefaultGenome, Genome):
   @staticmethod
   def gen_map_arrs(n_tiles, map_width, map_height, cppn):
      multi_hot_1 = np.zeros((n_tiles, map_width, map_height), dtype=np.float)
      map_arr = np.zeros((map_width, map_height), dtype=np.uint8)
      X = np.arange(map_width)
      # center from -1 to 1
      X = X * 2 / map_width - 1
      # scale (ad hoc)
      X = X * 2
      X, Y = np.meshgrid(X, X)
      # swapping axes to match naive implementation
      multi_hot = [node(x_in=torch.Tensor(X), y_in=torch.Tensor(Y)).numpy() for node in cppn]
      chan_types = [type(multi_hot[i]) for i in range(len(multi_hot))]
      if float in chan_types:
         raise Exception
      multi_hot = np.array(multi_hot)
      # multi_hot = multi_hot.swapaxes(1, 2)
      map_arr = np.argmax(multi_hot, axis=0)
      # for x in range(map_width):
      #    for y in range(map_height):
      #        # a decent scale for NMMO
      #        x_i, y_i = x * 2 / map_width - 1, y * 2 / map_width - 1
      #        x_i, y_i = x_i * 2, y_i * 2
      #        v = cppn.activate((x_i, y_i))

      #    #    if self.config.THRESHOLD:
      #    #        # use NMMO's threshold logic
      #    #        assert len(v) == 1
      #    #        v = v[0]
      #    #        v = self.map_generator.material_evo(self.config, v)
      #    #    else:
      #        # CPPN has output channel for each tile type; take argmax over channels
      #        # also a spawn-point tile
      #        assert len(v) == n_tiles
      #        multi_hot_1[:, x, y] = v
      #        v = np.array(v)
      #        # Shuffle before selecting argmax to prevent bias for certain tile types in case of ties
      #        # v = np.random.choice(np.flatnonzero(v == v.max()))
      #        v = np.argmax(v)
      #        map_arr[x, y] = v
      # # Debugging: make sure we match the naive approach, within reason
      # assert np.sum(abs(multi_hot_1 - multi_hot)) < 1e-5

      return map_arr, multi_hot

   ''' A wrapper class for a NEAT genome, which smuggles in other evolvable params for NMMO,
   beyond the map.'''
   def __init__(self, key, neat_config, n_tiles, map_width):
       super().__init__(key)
       self.configure_new(neat_config.genome_config)
       self.map_arr = None
       self.multi_hot = None
       self.atk_mults = gen_atk_mults()
       self.age = 0
       self.neat_config = neat_config
       self.map_width = self.map_height = map_width
       self.n_tiles = n_tiles

   def configure_crossover(self, parent1, parent2, config):
       super().configure_crossover(parent1, parent2, config)
       mults_1, mults_2 = parent1.atk_mults, parent2.atk_mults
       self.atk_mults, _ = mate_atk_mults(mults_1, mults_2, single_offspring=True)

   def mutate(self):
       super().mutate(self.neat_config.genome_config)
       self.atk_mults = mutate_atk_mults(self.atk_mults)
       self.age = 0

   def clone(self):
       child = copy.deepcopy(self)
       child.age = 0

       return child

   def gen_map(self):
      #if self.map_arr is not None and self.multi_hot is not None:
      #    return self.map_arr, self.multi_hot

      # cppn = neat.nn.FeedForwardNetwork.create(self, self.neat_config)
      cppn = create_cppn(self, self.neat_config, ['x_in', 'y_in'], ['tile_{}'.format(i) for i in range(self.n_tiles)])
      #       if self.config.NET_RENDER:
      #          with open('nmmo_cppn.pkl', 'wb') a

      self.map_arr, self.multi_hot = DefaultGenome.gen_map_arrs(self.n_tiles, self.map_width, self.map_height, cppn)


class PatternGenome(Genome):
   @staticmethod
   def mutate_patterns(rng, patterns, max_patterns, n_tiles, map_width):
      n_patterns = len(patterns)
      n_add = int(rng.exponential(scale=1.0, size=1))
      n_add = min(max_patterns - n_patterns, n_add)
      n_del = int(rng.exponential(scale=1.0, size=1))
      n_del = min(n_patterns - 1, int(rng.exponential(scale=1.0, size=1)))
      n_mut = max(1, int(rng.exponential(scale=1.0, size=1)))
      #       print('n_add: {}, n_mut: {}, n_del: {}'.format(n_add, n_mut, n_del))
      mutees = np.random.choice(patterns, n_mut)
      for m in mutees:
         m.mutate()
      for i in range(n_del):
         patterns.pop(np.random.randint(n_patterns - i))
      new_types = np.random.choice(PRIMITIVE_TYPES, n_add)
      [patterns.append(n.generate(n,
                                       tile_i=np.random.randint(0, n_tiles - 1),
                                       intensity=np.random.random(),
                                       n_tiles=n_tiles,
                                       map_width=map_width)) for n in new_types]
      #       print('added {} patterns, mutated {}, deleted {}'.format(n_add, n_mut, n_del))
      return patterns

   def __init__(self, n_tiles, map_width, default_tile):
        super().__init__(n_tiles, map_width)
        self.map_width = map_width
        self.n_tiles = n_tiles
       #self.max_patterns = map_width ** 2 / 20
        self.max_patterns = 100
        self.default_tile = default_tile
#       self.pattern_templates = [Line, Rectangle, RectanglePerimeter, Gaussian, Circle, CirclePerimeter]
        #     self.pattern_templates = [Gaussian]
        self.weights =  [2/3,  1/3]
        # some MAP-Elites dimensions
#       self.features = None
#       self. = None
#       self.multi_hot = None
        self.rng = default_rng()

        n_patterns = 25
        self.patterns = np.random.choice(PRIMITIVE_TYPES,
                                         n_patterns).tolist()
        #       self.features = [0, 0]  # Number of lines, circle perimeters

        for i, p in enumerate(self.patterns):
            # if p in [Line, Rectangle]:
            intensity = np.random.random()
            tile_i = np.random.randint(0, self.n_tiles)
            p = p.generate(p, tile_i=tile_i, intensity=intensity,
                           n_tiles=self.n_tiles, map_width=self.map_width)
            self.patterns[i] = p

    #  def init_endpoint_pattern(self, p):
    #     p = p(
    #           np.random.randint(0, self.map_width, 2),
    #           np.random.randint(0, self.map_width, 2),
    #           np.random.randint(0, self.n_tiles),
    #           np.random.random(),
    #           self.map_width,
    #           )

    #     return p

   def gen_map(self):

#      self.update_features()
       return self.paint_map()

#  def update_features(self):
#      self.features = [0, 0]
#      for p in self.patterns:
#          if isinstance(p, (Line, Rectangle, RectanglePerimeter)):
#              self.features[0] += 1
#          elif isinstance(p, (CirclePerimeter, Circle, Gaussian)):
#              self.features[1] += 1

   def get_iterable(self):
       # each pattern has: type, intensity, p1, p2, p3, p4
       it = np.zeros(shape=())

       return []

#      #FIXME: hack
#      return [self.__hash__]

   def mutate(self):
      super().mutate()
      patterns = PatternGenome.mutate_patterns(rng=self.rng, patterns=self.patterns, max_patterns=self.max_patterns,
                                      n_tiles=self.n_tiles, map_width=self.map_width)
      self.patterns = patterns
      self.multi_hot = None
#      self.flat_map = None
#      self.update_features()


   def paint_map(self):
#      if hasattr(self, 'flat_map') and self.flat_map is not None:
#          return self.flat_map, self.multi_hot
       multi_hot = np.zeros((self.n_tiles, self.map_width, self.map_width))
       multi_hot[self.default_tile, :, :] = 1e-10

       for p in self.patterns:
           p.paint(multi_hot)
       map_arr = np.argmax(multi_hot, axis=0)
       self.map_arr, self.multi_hot = map_arr, multi_hot

#      return map_arr, multi_hot

class TileFlipGenome(Genome):
    def __init__(self, n_tiles, map_width):
        super().__init__(n_tiles, map_width)
        self.map_arr = np.random.randint(0, n_tiles, (map_width, map_width))
        self.rng = default_rng()

    def mutate(self):
        #TODO: uncomment and tweak this newer/smarter stuff

        # TODO: actually it's not very smart at all. Use gaussians or something to make it smarter and
        #  make it so that local/global entropy might actually change from drift!!!
        map_width = self.map_width
        mut_scale = map_width ** 2 * 0.005
#       mut_scale = 1
        n_muts = max(1, int(self.rng.exponential(scale=mut_scale, size=1)))
        new = np.random.randint(1, self.n_tiles, (n_muts))
#       new = np.random.randint(0, self.n_tiles, (n_muts))
        idxs = np.argwhere(np.zeros((map_width, map_width)) == 0)
        pos_idxs = np.random.choice(idxs.shape[0], n_muts, replace=False)
        mut_idxs = idxs[pos_idxs]
#       print(mut_idxs)
#       print(new)
        self.map_arr[mut_idxs[:,0], mut_idxs[:,1]] = (self.map_arr[mut_idxs[:,0], mut_idxs[:,1]] + new) % self.n_tiles
#       self.map_arr[mut_idxs[:,0], mut_idxs[:,1]] = new

    def gen_map(self):
        pass

    def clone(self):
        child = TileFlipGenome(self.n_tiles, self.map_width)
        child.map_arr = self.map_arr.copy()

        return child

    def get_iterable(self):
        return self.map_arr.reshape(-1)


class SimplexNoiseGenome(Genome):
   def __init__(self, n_tiles, map_width, baseline=False):
      super().__init__(n_tiles, map_width)
      self.baseline = baseline
      if baseline:
         return self.init_baseline()
      self.x0, self.y0 = np.random.randint(-1e4, 1e4, size=2)
      self.step_size = np.random.random() * 2
      n_bands = np.random.randint(n_tiles, n_tiles + 3)
      threshes = np.random.random(n_bands)
      # values between 0 and 1 that represent thresholds between tile types
      self.threshes = np.array([t + i for (i, t) in enumerate(threshes)]) / n_bands
      # the tile types to be thresholded
      self.thresh_tiles = np.random.randint(0, n_tiles, size=n_bands+1)

   def init_baseline(self):
      self.x0, self.y0 = np.random.randint(-1e4, 1e4, size=2)
      self.step_size = 0.125
      # Following the parameters for the baseline simplex noise maps -- see projekt/config
      self.n_bands = 9
      self.threshes = np.array([
         0.25,
         0.4,
         0.45,
         0.5,
         0.715,
         #              0.35,
         0.75,
         0.8,
         0.85,
      ])
      self.thresh_tiles = np.array([
         enums.MaterialEnum.WATER.value.index,
         enums.MaterialEnum.GRASS.value.index,
         enums.MaterialEnum.LAVA.value.index,
         enums.MaterialEnum.SPAWN.value.index,
         enums.MaterialEnum.GRASS.value.index,
         #              enums.MaterialEnum.FOREST.value.index,
         enums.MaterialEnum.FOREST.value.index,
         enums.MaterialEnum.TREE.value.index,
         enums.MaterialEnum.OREROCK.value.index,
         enums.MaterialEnum.STONE.value.index,
      ])
      return

   def mutate(self):
      if self.baseline:
         # If running baseline, never mutate (this is a failsafe, we should never end up here)
         return self.init_baseline()
      n_actions = 4
      actions = np.random.random(n_actions) < 0.3
      full_threshes = np.concatenate((self.threshes, [1]))
      if actions.sum() == 0:
         actions[np.random.randint(0, n_actions)] = True
      if actions[0]:
         if np.random.random() < 0.5 and self.threshes.shape[0] > self.n_tiles:
            j = np.random.randint(0, self.threshes.shape[0])
            self.threshes = np.concatenate((self.threshes[:j], self.threshes[j+1:]))
            # kinda weird that we'll never bonk off the last one
            self.thresh_tiles = np.concatenate((self.thresh_tiles[:j], self.thresh_tiles[j+1:]))
         elif self.threshes.shape[0] < 2 * self.n_tiles:
            j = np.random.randint(0, self.threshes.shape[0])
            self.threshes = np.concatenate((self.threshes[:j],
                                           np.random.uniform(self.threshes[j], full_threshes[j+1], 1),
                                            self.threshes[j+1:]))
            self.thresh_tiles = np.concatenate((self.thresh_tiles[:j],
                                               [np.random.randint(0, self.n_tiles)],
                                               self.thresh_tiles[j+1:]))
         else:
             pass
             # oops no-op
      if actions[1]:
         j = np.random.randint(0, self.threshes.shape[0])
         self.threshes[j] = np.random.uniform(self.threshes[j-1], full_threshes[j+1], 1)
      if actions[2]:
          j = np.random.randint(0, self.thresh_tiles.shape[0])
          self.thresh_tiles[j] = np.random.randint(0, self.n_tiles)
      if actions[3]:
          if np.random.random() < 0.5:
              self.x0, self.y0 = np.random.randint(-1e4, 1e4, size=2)
          else:
              self.step_size = np.random.random() * 2

   def gen_map(self):
      map_width = self.map_width
      s = np.arange(map_width)
      X, Y = np.meshgrid(s, s)
      val = np.zeros((map_width, map_width), dtype=float)
      map_arr = np.zeros((map_width, map_width), dtype=np.uint8)
      val = vec_noise.snoise2(self.x0 + X * self.step_size, self.y0 + Y * self.step_size)
      full_threshes = np.concatenate((self.threshes, [1]))
      if full_threshes.shape[0] != self.thresh_tiles.shape[0]:
         raise Exception('Number of thresholds ({}) does not match number of tile "bands" ({}).'.format(full_threshes.shape[0], self.thresh_tiles.shape[0]))
      for i in range(map_arr.shape[0]):
         for j in range(map_arr.shape[1]):
            t = np.where(0.5 + 0.5 * val[i, j] <= full_threshes)[0][0]
            if t >= self.thresh_tiles.shape[0]:
               raise Exception("Selected tile is out of bounds in list of tiles for simplex genome.")
            map_arr[i, j] = self.thresh_tiles[t]
         self.map_arr = map_arr.astype(np.uint8)

class MazeGenome(Genome):
   def __init__(self, n_tiles, map_width):
      super().__init__(n_tiles, map_width)

      self.cell_thickness = 3
      self.wall_thickness = 1

      self.tiles = np.array([
         enums.MaterialEnum.WATER.value.index,
         enums.MaterialEnum.GRASS.value.index,
         enums.MaterialEnum.LAVA.value.index,
         enums.MaterialEnum.SPAWN.value.index,
         enums.MaterialEnum.GRASS.value.index,
         enums.MaterialEnum.FOREST.value.index,
         enums.MaterialEnum.TREE.value.index,
         enums.MaterialEnum.OREROCK.value.index,
         enums.MaterialEnum.STONE.value.index,
      ])

      self.init_baseline()


   def init_baseline(self):
      self.x0, self.y0 = np.random.randint(-1e4, 1e4, size=2)
      self.step_size = 0.125
      # Following the parameters for the baseline simplex noise maps -- see projekt/config
      self.n_bands = 9
      self.thresholds = np.array([
         0.20,
         0.4,
         0.41,
         0.48,
         0.715,
         0.75,
         0.8,
         0.85,
         1.0
      ])
      return

   def mutate(self):
      pass

   def gen_map(self):
      self.map_arr = np.full((self.map_width, self.map_width), enums.MaterialEnum.STONE.value.index, dtype=float)
      self.generate_maze()
      self.layout_base_map()

      def count_spawns(matl):
         spawn_count = 0
         for i in range(matl.shape[0]):
            for j in range(matl.shape[1]):
               if matl[i, j] == enums.MaterialEnum.SPAWN.value.index:
                  spawn_count += 1
         return spawn_count

   def layout_base_map(self):
      map_width = self.map_width
      s = np.arange(map_width)
      X, Y = np.meshgrid(s, s)
      val = np.zeros((map_width, map_width), dtype=float)
      val = vec_noise.snoise2(self.x0 + X * self.step_size, self.y0 + Y * self.step_size)
      if self.tiles.shape[0] != self.tiles.shape[0]:
         raise Exception('Number of thresholds ({}) does not match number of tile "bands" ({}).'.format(self.landscape_threshes.shape[0], self.landscape_tiles.shape[0]))
      for i in range(self.map_arr.shape[0]):
         for j in range(self.map_arr.shape[1]):
            t = np.where(0.5 + 0.5 * val[i, j] <= self.thresholds)[0][0]
            if t >= self.tiles.shape[0]:
               raise Exception("Selected tile is out of bounds in list of tiles for niche genome.")
            self.map_arr[i, j] = self.tiles[t] if self.map_arr[i,j] == enums.MaterialEnum.GRASS.value.index else self.map_arr[i,j]

   def generate_maze(self):
      grid_size = (self.map_width - self.wall_thickness) // (self.cell_thickness + self.wall_thickness)
      nodes = [(i,j) for j in range(grid_size) for i in range(grid_size)]
      neighbors = lambda n: [(n[0] + x, n[1] + y) for x,y in [(-1,0),(1,0),(0,-1),(0,1)] if n[0] + x >= 0 and n[0] + x < grid_size and n[1] + y >= 0 and n[1] + y < grid_size]

      class DJS:
         def __init__(self, n_s):
            self.n_mapping = {}
            for i, v in enumerate(n_s):
               n = self.DSN(v, i)
               self.n_mapping[v] = n

         def find(self, n):
            return self.find_n(n).p

         def find_n(self, n):
            if type(self.n_mapping[n].p) is int:
               return self.n_mapping[n]
            else:
               p_n = self.find_n(self.n_mapping[n].p.val)
               self.n_mapping[n].p = p_n
               return p_n

         def union(self, n1, n2):
            p1 = self.find_n(n1)
            p2 = self.find_n(n2)
            if p1.p != p2.p:
               p1.p = p2

         class DSN:
            def __init__(self, val, p):
               self.val = val
               self.p = p

      E = [(n, nb) for n in nodes for nb in neighbors(n)]
      maze = []
      ds = DJS(nodes)

      while len(maze) < len(nodes) - 1:
         e = E.pop(np.random.randint(0, len(E) - 1))
         if ds.find(e[0]) != ds.find(e[1]):
            ds.union(e[0], e[1])
            maze.append(e)

      for e in maze:
         min_x = self.wall_thickness+min(e[0][1], e[1][1])*(self.cell_thickness + self.wall_thickness)
         max_x = self.wall_thickness+max(e[0][1], e[1][1])*(self.cell_thickness + self.wall_thickness)
         min_y = self.wall_thickness+min(e[0][0], e[1][0])*(self.cell_thickness + self.wall_thickness)
         max_y = self.wall_thickness+max(e[0][0], e[1][0])*(self.cell_thickness + self.wall_thickness)
         self.map_arr[min_x:max_x+self.cell_thickness,min_y:max_y+self.cell_thickness] = enums.MaterialEnum.GRASS.value.index

      del E
      del maze
      del ds
      gc.collect()

class ResourceNichesGenome(Genome):
   def __init__(self, n_tiles, map_width, lava_border_width, access_bottleneck=False):
      super().__init__(n_tiles, map_width)
      self.niche_zone_radius = 5
      self.niche_loc_spacing = 25
      self.num_niche_loc_gen_tries = 30
      self.wall_width = 2
      self.resource_wall_buf = 1
      self.niche_resource_drop_len = 2
      self.access_bottleneck = access_bottleneck
      self.lava_border_width = lava_border_width

      self.resource_tiles = np.array([
         enums.MaterialEnum.FOREST.value.index, # SHOULD WE EQUALLY DISTRIBUTE FOOD AND WATER?
         enums.MaterialEnum.WATER.value.index,
         enums.MaterialEnum.TREE.value.index,
         enums.MaterialEnum.OREROCK.value.index,
      ])

      self.landscape_tiles = np.array([
         enums.MaterialEnum.FOREST.value.index,
         enums.MaterialEnum.WATER.value.index,
         enums.MaterialEnum.SPAWN.value.index,
         enums.MaterialEnum.LAVA.value.index,
         enums.MaterialEnum.STONE.value.index,
         enums.MaterialEnum.GRASS.value.index
      ])

      self.niche_locuses = []

      self.init_baseline()

   def init_baseline(self):
      self.x0, self.y0 = np.random.randint(-1e4, 1e4, size=2)
      self.step_size = 0.125
      # Following the parameters for the baseline simplex noise maps -- see projekt/config
      self.n_bands = 4
      self.landscape_threshes = np.array([
         0.05, # STARTER FOOD
         0.1, # STARTER WATER
         0.2,  # SPAWNS
         0.21, # LAVA
         0.25 if self.access_bottleneck else 0.30, # STONE
         1,    # GRASS
      ])
      return

   def mutate(self):
      pass

   def gen_map(self):
      self.layout_base_map()
      self.select_niche_locuses()
      self.create_niche_clusters()
      self.add_basic_resource_near_clusters()
      self.map_arr = self.map_arr.astype(np.uint8)

   def layout_base_map(self):
      map_width = self.map_width
      s = np.arange(map_width)
      X, Y = np.meshgrid(s, s)
      val = np.zeros((map_width, map_width), dtype=float)
      map_arr = np.zeros((map_width, map_width), dtype=np.uint8)
      val = vec_noise.snoise2(self.x0 + X * self.step_size, self.y0 + Y * self.step_size)
      if self.landscape_threshes.shape[0] != self.landscape_tiles.shape[0]:
         raise Exception('Number of thresholds ({}) does not match number of tile "bands" ({}).'.format(self.landscape_threshes.shape[0], self.landscape_tiles.shape[0]))
      for i in range(map_arr.shape[0]):
         for j in range(map_arr.shape[1]):
            t = np.where(0.5 + 0.5 * val[i, j] <= self.landscape_threshes)[0][0]
            if t >= self.landscape_tiles.shape[0]:
               raise Exception("Selected tile is out of bounds in list of tiles for niche genome.")
            map_arr[i, j] = self.landscape_tiles[t]
      self.map_arr = map_arr.astype(np.uint8)

   def select_niche_locuses(self):
      niche_rad = self.niche_zone_radius + self.resource_wall_buf + self.wall_width

      rand_pt = lambda: (np.random.randint(low=niche_rad + self.lava_border_width, high=self.map_width - self.lava_border_width - niche_rad, size=1)[0],
                            np.random.randint(low=niche_rad + self.lava_border_width, high=self.map_width - self.lava_border_width - niche_rad, size=1)[0])

      pop_rand = lambda l: l.pop(np.random.randint(len(l), size=1)[0])

      on_map = lambda pt: (pt[0] > (niche_rad + self.lava_border_width) and pt[0] < self.map_width - self.lava_border_width - niche_rad) \
                           and (pt[1] > (niche_rad + self.lava_border_width) and pt[1] < self.map_width - self.lava_border_width - niche_rad)

      l2_norm = lambda a, b: np.linalg.norm(np.array(a) - np.array(b))

      in_range = lambda pt, pts, min_dist: any([l2_norm(ept, pt) < min_dist for ept in pts])

      def rand_pt_near(pt, min_dist):
         (r1, r2) = np.random.uniform(size=2)
         rad = min_dist * (r1 + 1)
         ang = 2 * np.pi * r2
         return (int(pt[0] + rad * np.cos(ang)), int(pt[1] + rad * np.sin(ang)))

      # Select point via poisson disk sampling
      process_list = []
      sample_pts = []

      first_pt = rand_pt()
      process_list.append(first_pt)
      sample_pts.append(first_pt)

      while len(sample_pts) < self.resource_tiles.shape[0]:
         while len(process_list) > 0:
            pt = pop_rand(process_list)
            for i in range(0, self.num_niche_loc_gen_tries):
               new_pt = rand_pt_near(pt, self.niche_loc_spacing)
               if on_map(new_pt) and (not in_range(new_pt, sample_pts, self.niche_loc_spacing)):
                  process_list.append(new_pt)
                  sample_pts.append(new_pt)

               if len(sample_pts) >= self.resource_tiles.shape[0]:
                  break
            if len(sample_pts) >= self.resource_tiles.shape[0]:
               break


         if len(sample_pts) < self.resource_tiles.shape[0]:
            process_list = []
            sample_pts = []

            first_pt = rand_pt()
            process_list.append(first_pt)
            sample_pts.append(first_pt)

         pt_pairs = list(itertools.product(sample_pts, sample_pts))
         pt_pairs = list(filter(lambda p: not(p[0] == p[1]), pt_pairs))
         if any([l2_norm(p[0], p[1]) < self.niche_loc_spacing for p in pt_pairs]):
#           print("points are too close, resetting")
            process_list = []
            sample_pts = []

            first_pt = rand_pt()
            process_list.append(first_pt)
            sample_pts.append(first_pt)

      self.niche_locuses = sample_pts

   def create_niche_clusters(self):
      def create_circular_mask(h, w, center=None, radius=None):
         if center is None: # use the middle of the image
            center = (int(w/2), int(h/2))

         Y, X = np.ogrid[:h, :w]
         dist_from_center = np.sqrt((X - center[1])**2 + (Y-center[0])**2)

         mask = dist_from_center <= radius
         return mask

      for (resource_idx, loc) in enumerate(self.niche_locuses):
         if self.access_bottleneck:
            mask = create_circular_mask(self.map_width, self.map_width, loc, self.niche_zone_radius + self.resource_wall_buf + self.wall_width)
            self.map_arr[mask] = enums.MaterialEnum.STONE.value.index
         mask = create_circular_mask(self.map_width, self.map_width, loc, self.niche_zone_radius + self.resource_wall_buf)
         self.map_arr[mask] = enums.MaterialEnum.GRASS.value.index
         mask = create_circular_mask(self.map_width, self.map_width, loc, self.niche_zone_radius)
         self.map_arr[mask] = self.resource_tiles[resource_idx]

   def add_basic_resource_near_clusters(self):
      niche_wall_rad = self.niche_zone_radius + self.resource_wall_buf + self.wall_width
      disp = (int(np.sqrt((niche_wall_rad ** 2) / 2)), int(np.sqrt((niche_wall_rad ** 2) / 2)))
      el_add = lambda a, b: tuple(map(lambda i, j: i + j, a, b))
      el_mul = lambda a, b: tuple(map(lambda i, j: i * j, a, b))

      def inv_idx(a, i):
         l_a = list(a)
         l_a[i] = a[i] * -1
         return tuple(l_a)

      def layout_resources(self, loc, disp_vec, tbv, bbv, rv):
         resource_drop_start = el_add(el_mul(disp_vec, rv), loc)
#        print(loc, resource_drop_start, rv)
         top_band_start = el_add(resource_drop_start, tbv)
         bot_band_start = el_add(resource_drop_start, bbv)
         self.map_arr[resource_drop_start] = enums.MaterialEnum.LAVA.value.index
         for i in range(0, self.niche_resource_drop_len * 2, 2):
            t = el_add(top_band_start, el_mul((i, i), rv))
            c = el_add(resource_drop_start, el_mul((i, i), rv))
            c1 = el_add(resource_drop_start, el_mul((i + 1, i), rv))
            c2 = el_add(resource_drop_start, el_mul((i, i + 1), rv))
            b = el_add(bot_band_start, el_mul((i, i), rv))
            self.map_arr[t] = enums.MaterialEnum.WATER.value.index
            self.map_arr[b] = enums.MaterialEnum.WATER.value.index
            self.map_arr[c] = enums.MaterialEnum.GRASS.value.index
            self.map_arr[c1] = enums.MaterialEnum.GRASS.value.index
            self.map_arr[c2] = enums.MaterialEnum.GRASS.value.index

            t = el_add(top_band_start, el_mul((i + 1, i + 1), rv))
            c = el_add(resource_drop_start, el_mul((i + 1, i + 1), rv))
            c1 = el_add(resource_drop_start, el_mul((i + 2, i + 1), rv))
            c2 = el_add(resource_drop_start, el_mul((i + 1, i + 2), rv))
            b = el_add(bot_band_start, el_mul((i + 1, i + 1), rv))
            self.map_arr[t] = enums.MaterialEnum.FOREST.value.index
            self.map_arr[b] = enums.MaterialEnum.FOREST.value.index
            self.map_arr[c] = enums.MaterialEnum.GRASS.value.index
            self.map_arr[c1] = enums.MaterialEnum.GRASS.value.index
            self.map_arr[c2] = enums.MaterialEnum.GRASS.value.index

         for i in range(self.niche_resource_drop_len * 2, (self.niche_resource_drop_len * 2) + 2):
            t = el_add(top_band_start, el_mul((i, i), rv))
            c = el_add(resource_drop_start, el_mul((i, i), rv))
            c1 = el_add(resource_drop_start, el_mul((i + 1, i), rv))
            c2 = el_add(resource_drop_start, el_mul((i, i + 1), rv))
            b = el_add(bot_band_start, el_mul((i, i), rv))
            self.map_arr[t] = enums.MaterialEnum.GRASS.value.index
            self.map_arr[b] = enums.MaterialEnum.GRASS.value.index
            self.map_arr[c] = enums.MaterialEnum.GRASS.value.index
            self.map_arr[c1] = enums.MaterialEnum.GRASS.value.index
            self.map_arr[c2] = enums.MaterialEnum.GRASS.value.index

         if self.access_bottleneck:
            wv = el_mul(rv, (-1, -1))
            for i in range(self.wall_width):
               c = el_add(resource_drop_start, el_mul((i, i), wv))
               c1 = el_add(resource_drop_start, el_mul((i + 1, i), wv))
               c2 = el_add(resource_drop_start, el_mul((i, i + 1), wv))
               self.map_arr[c] = enums.MaterialEnum.GRASS.value.index
               self.map_arr[c1] = enums.MaterialEnum.GRASS.value.index
               self.map_arr[c2] = enums.MaterialEnum.GRASS.value.index


      for loc in self.niche_locuses:
         if loc[0] < self.map_width // 2 and loc[1] < self.map_width // 2:
            # TL should go BR
            layout_resources(self, loc, disp, (-1,1), (1,-1), (1,1))

         elif loc[0] < self.map_width // 2 and loc[1] > self.map_width // 2:
            # TR -> BL
            layout_resources(self, loc, disp, (-1,-1), (1,1), (1,-1))

         elif loc[0] > self.map_width // 2 and loc[1] < self.map_width // 2:
            # BL -> TR
            layout_resources(self, loc, disp, (-1,-1), (1,1), (-1,1))

         elif loc[0] > self.map_width // 2 and loc[1] > self.map_width // 2:
            # BR -> TL
            layout_resources(self, loc, disp, (-1,1), (1,-1), (-1,-1))

class RiverBottleneckGenome(Genome):
   def __init__(self, n_tiles, map_width, lava_border_width):
      super().__init__(n_tiles, map_width)
      self.lava_border_width = lava_border_width
      return self.init_baseline()

   def init_baseline(self):
      self.x0, self.y0 = np.random.randint(-1e4, 1e4, size=2)
      self.step_size = 0.125
      # Following the parameters for the baseline simplex noise maps -- see projekt/config
      self.n_bands = 9
      self.threshes = np.array([
         0.25,
         0.4,
         0.45,
         0.5,
         0.715,
         #              0.35,
         0.75,
         0.8,
         0.85,
      ])
      self.thresh_tiles = np.array([
         enums.MaterialEnum.WATER.value.index,
         enums.MaterialEnum.GRASS.value.index,
         enums.MaterialEnum.LAVA.value.index,
         enums.MaterialEnum.SPAWN.value.index,
         enums.MaterialEnum.GRASS.value.index,
         #              enums.MaterialEnum.FOREST.value.index,
         enums.MaterialEnum.FOREST.value.index,
         enums.MaterialEnum.TREE.value.index,
         enums.MaterialEnum.OREROCK.value.index,
         enums.MaterialEnum.STONE.value.index,
      ])
      self.n_bridges = 2
      self.starter_food_thresh = (0.25, 0.26)
      return

   def mutate(self):
      return self.init_baseline()

   def gen_map(self):
      map_width = self.map_width
      s = np.arange(map_width)
      X, Y = np.meshgrid(s, s)
      val = np.zeros((map_width, map_width), dtype=float)
      map_arr = np.zeros((map_width, map_width), dtype=np.uint8)
      val = vec_noise.snoise2(self.x0 + X * self.step_size, self.y0 + Y * self.step_size)
      full_threshes = np.concatenate((self.threshes, [1]))
      if full_threshes.shape[0] != self.thresh_tiles.shape[0]:
         raise Exception('Number of thresholds ({}) does not match number of tile "bands" ({}).'.format(full_threshes.shape[0], self.thresh_tiles.shape[0]))
      for i in range(map_arr.shape[0]):
         for j in range(map_arr.shape[1]):
            t = np.where(0.5 + 0.5 * val[i, j] <= full_threshes)[0][0]
            if t >= self.thresh_tiles.shape[0]:
               raise Exception("Selected tile is out of bounds in list of tiles for river bottleneck genome.")
            map_arr[i, j] = self.thresh_tiles[t]
         self.map_arr = map_arr.astype(np.uint8)

      #WHAT DEFINES BORDER SIZE FOR THE LAVA?
      self.map_arr = self.apply_river_features(self.map_arr, self.lava_border_width, num_bridges=self.n_bridges)

   def apply_river_features(self, matl, border_size, num_bridges=1, bridge_width=5):
      # Utility function to count the number of spawns to assess map validitity
      def count_spawns(matl):
         spawn_count = 0
         for i in range(matl.shape[0]):
            for j in range(matl.shape[1]):
               if matl[i, j] == enums.MaterialEnum.SPAWN.value.index:
                  spawn_count += 1
         return spawn_count

      # Create a bisecting river that follows the path x=y
      def create_river(self, matl):
         sz = matl.shape[0]
         for y in range(2, sz, 2):
            for x in range(2, sz, 2):
               if y == x:
                  for e in itertools.product(list(range(y - 2, y +2 )), list(range(x - 2, x + 2))):
                     matl[e] = enums.MaterialEnum.WATER.value.index
         return matl

      # Select num_bridges locations (from uniform distribution) on the line x=y to create bridges spanning the river
      def create_bridges(self, matl, border_size, num_bridges, bridge_width):
         locs = np.random.randint(0, matl.shape[0], num_bridges)
         for loc in locs:
            y_range = range(max(border_size, loc - (bridge_width // 2)), min(matl.shape[0] - border_size, loc + (bridge_width // 2)))
            x_range = range(max(border_size, loc - (bridge_width // 2)), min(matl.shape[0] - border_size, loc + (bridge_width // 2)))
            for e in itertools.product(list(y_range), list(x_range)):
               matl[e] = enums.MaterialEnum.GRASS.value.index

         return matl

      # Despawn spawn points and resources from different halves of the maps
      def depopulate_halves(self, matl):
         # Remove all resources from either all points where x > y or x < y depending on below_x_eq_y
         def remove_all_except_spawns(self, matl, below_x_eq_y=False):
            for y in range(matl.shape[0]):
               for x in range(matl.shape[1]):
                  if (x < y and below_x_eq_y) or (x > y and not below_x_eq_y):
                     if matl[y, x] not in [enums.MaterialEnum.WATER.value.index,
                                           enums.MaterialEnum.GRASS.value.index,
                                           enums.MaterialEnum.STONE.value.index,
                                           enums.MaterialEnum.SPAWN.value.index]:
                        matl[y, x] = enums.MaterialEnum.GRASS.value.index
            return matl

         # Remove all spawn points from either all points where x > y or x < y depending on below_x_eq_y
         def remove_spawns(self, matl, below_x_eq_y=True):
            #print("pre_removal: " + str(count_spawns(matl)))
            for y in range(matl.shape[0]):
               for x in range(matl.shape[1]):
                  if (x < y and below_x_eq_y) or (x > y and not below_x_eq_y):
                     if matl[y, x] == enums.MaterialEnum.SPAWN.value.index:
                        matl[y, x] = enums.MaterialEnum.GRASS.value.index
            #print("post_removal: " + str(count_spawns(matl)))
            return matl

         # Add back food on the spawn side to give agents enough resources to get started
         def add_starter_food(self, matl, below_x_eq_y=True):
            map_width = self.map_width
            s = np.arange(map_width)
            X, Y = np.meshgrid(s, s)
            val = np.zeros((map_width, map_width), dtype=float)
            val = vec_noise.snoise2(self.x0 + X * self.step_size, self.y0 + Y * self.step_size)
            full_threshes = np.concatenate((self.threshes, [1]))
            if full_threshes.shape[0] != self.thresh_tiles.shape[0]:
               raise Exception('Number of thresholds ({}) does not match number of tile "bands" ({}).'.format(full_threshes.shape[0], self.thresh_tiles.shape[0]))
            for i in range(matl.shape[0]):
               for j in range(matl.shape[1]):
                  if (j < i and below_x_eq_y) or (j > i and not below_x_eq_y):
                     # SHOULD IT BE GREATER THAN OR LESS THAN?
                     if (0.5 + 0.5 * val[i, j] >= self.starter_food_thresh[0]) and (0.5 + 0.5 * val[i, j] < self.starter_food_thresh[1]):
                        matl[i, j] = enums.MaterialEnum.FOREST.value.index
            return matl.astype(np.uint8)

         # Coin flip to select if the resources will be on the bottom half or the top half
         resource_half_is_bot = np.random.randint(0, 2, 1)

         matl = remove_all_except_spawns(self, matl, below_x_eq_y=resource_half_is_bot)
         matl = remove_spawns(self, matl, below_x_eq_y=(not resource_half_is_bot))
         matl = add_starter_food(self, matl, below_x_eq_y=resource_half_is_bot)

         return matl

      matl = create_river(self, matl)
      matl = create_bridges(self, matl, border_size, num_bridges, bridge_width)
      matl = depopulate_halves(self, matl)

      return matl



class LSystemGenome(Genome):
   def __init__(self, n_tiles, map_width):
       super().__init__(n_tiles, map_width)
       self.map_width = map_width
       self.n_tiles = n_tiles
       self.axiom_width = axiom_width = 4
       #FIXME: not currently guaranteeing mutation will have an effect, since we may change only one expansion rule,
       # which then may not be used when expanding.
       self.n_rnd_expansions = 5  # how many (stochastic) expansion rules per tile type
       self.axiom = np.random.randint(0, n_tiles, (axiom_width, axiom_width), dtype=np.uint8)
       self.expansions = dict([
           (i, np.repeat(np.random.randint(0, n_tiles, (1, 2, 2)), self.n_rnd_expansions, axis=0)) for i in range(n_tiles)
       ])
       # Count up number of expansions needed and make a random choice of expansion rule for each one
       expanding_w = axiom_width
       n_expansions = 0
       while expanding_w < map_width:
          n_expansions += expanding_w ** 2
          expanding_w *= 2
       self.rule_choices = np.random.randint(0, self.n_rnd_expansions, size=n_expansions)
       self.rng = default_rng()

   def mutate(self):
      super().mutate()
      rnd = np.random.random()
      n_rule_mut = int(self.rng.exponential(scale=self.n_rnd_expansions, size=1))
      # roughly: change 1 per each random rule per each expansion
      choice_scale = np.log(self.map_width) / np.log(self.axiom_width)
      n_choice_mut = int(self.rng.exponential(scale=choice_scale, size=1))
      if n_rule_mut + n_choice_mut == 0:
         min_ax_mut = 1
      else:
         min_ax_mut = 0
      n_ax_mut = max(min_ax_mut, int(self.rng.exponential(scale=1.0, size=1)))
      for _ in range(n_ax_mut):
         x, y = (np.random.randint(0, self.axiom_width, (2)))
         # make sure to mutate to new tile type
         self.axiom[x, y] = (self.axiom[x, y] + np.random.randint(self.n_tiles)) % self.n_tiles
      for _ in range(n_rule_mut):
         rule_id = np.random.randint(len(self.expansions))
         expansion_id = np.random.randint(self.n_rnd_expansions)
         x, y = (np.random.randint(0, 2, (2)))
         self.expansions[rule_id][expansion_id][x, y] = (self.expansions[rule_id][expansion_id][x, y] + np.random.randint(self.n_tiles)) % self.n_tiles
      for _ in range(n_choice_mut):
         self.rule_choices[np.random.randint(len(self.rule_choices))] = np.random.randint(self.n_rnd_expansions)

#     n_rule_cells = len(self.expansions) * self.n_rnd_expansions * 2 * 2
#     self.axiom = np.where(np.random.random(self.axiom.shape) < 1/8, np.random.randint(0, self.n_tiles, (self.axiom_width, self.axiom_width)), self.axiom)
#     [self.expansions.update({k: np.where(np.random.random(expansion.shape) < 1.5/n_rule_cells, np.random.randint(0, self.n_tiles, (2, 2)), expansion)})
#      for (k, expansion) in self.expansions.items()]

   def gen_map(self):
      # Eliminate randomness from the generation process
      map_arr = self.axiom.copy()
      i_exp = 0
      while map_arr.shape[0] < self.map_width:
         # Expand until we match the map width
         new_arr = np.empty((map_arr.shape[0] * 2, map_arr.shape[1] * 2), dtype=np.uint8)
         for i in range(map_arr.shape[0]):
            for j in range(map_arr.shape[1]):
               new_arr[2*i:2*i+2, 2*j:2*j+2] = self.expansions[map_arr[i, j]][self.rule_choices[i_exp]]
               i_exp += 1
         map_arr = new_arr
      # Crop the map to match the target map width
      curr_width = map_arr.shape[0]
      crop_0 = int(np.floor((curr_width - self.map_width) / 2))
      crop_1 = int(np.ceil((curr_width - self.map_width) / 2))
      map_arr = map_arr[crop_0: -crop_1, crop_0: -crop_1]
#     if self.map_arr is not None and (map_arr == self.map_arr).all():
#        raise Exception("Apparently this shouldn't have happened?")
      self.map_arr = map_arr


class EvoIndividual(Individual):
   def __eq__(self, other):
       # For qdpy when adding to container and checking against existing elites
       return self is other

   def __init__(self, iterable, rank, evolver):
       super().__init__(iterable=iterable)
       self.iterable = iterable
       self.rank = rank
       self.fitness = Fitness([0])
       # invalidate fitness
       self.fitness.delValues()
       self.idx = rank
       self.n_tiles = len(evolver.TILE_PROBS)
       self.SPAWN_IDX = evolver.SPAWN_IDX
       self.FOOD_IDX = evolver.mats.FOREST.value.index
       self.WATER_IDX = evolver.mats.WATER.value.index
       self.EMPTY_IDX = evolver.mats.GRASS.value.index
       self.GRIDDLY = evolver.config.GRIDDLY
       if not self.GRIDDLY:
           self.SCRUB_IDX = evolver.mats.SCRUB.value.index
       self.NENT = evolver.config.NENT
       self.TERRAIN_BORDER = evolver.config.TERRAIN_BORDER
       self.SINGLE_SPAWN = evolver.config.SINGLE_SPAWN
       if evolver.BASELINE_SIMPLEX or evolver.config.PRETRAIN:
          # If we're training a baseline agent on frozen maps, iniialize our genomes to generate baseline maps
          self.chromosome = SimplexNoiseGenome(self.n_tiles, evolver.map_width, baseline=True)
       elif evolver.ALL_GENOMES:
           rnd = np.random.random()
           n_genomes = 6
           if rnd < 1/n_genomes:
               self.chromosome = DefaultGenome(self.idx, evolver.neat_config, self.n_tiles, evolver.map_width)
           elif rnd < 2/n_genomes:
               self.chromosome = PatternGenome(self.n_tiles, evolver.map_width,
                                               self.EMPTY_IDX)
           elif rnd < 3/n_genomes:
               self.chromosome = LSystemGenome(self.n_tiles, evolver.map_width)
           elif rnd < 4/n_genomes:
               self.chromosome = TileFlipGenome(self.n_tiles, evolver.map_width)
           elif rnd < 5 / n_genomes:
               self.chromosome = SimplexNoiseGenome(self.n_tiles, evolver.map_width)
           else:
#              seed = np.random.random((self.n_tiles, evolver.map_width, evolver.map_width))
               seed = None
               self.chromosome = CAGenome(self.n_tiles, evolver.map_width, seed=seed)
       elif evolver.CPPN:
           #FIXME: yeesh
           self.chromosome = DefaultGenome(self.idx, evolver.neat_config, self.n_tiles, evolver.map_width)
#          self.chromosome = evolver.chromosomes[self.idx]
       elif evolver.CA:
#          seed = np.random.random((self.n_tiles, evolver.map_width, evolver.map_width))
           seed = None
           self.chromosome = CAGenome(self.n_tiles, evolver.map_width, seed)
       elif evolver.LSYSTEM:
           self.chromosome = LSystemGenome(self.n_tiles, evolver.map_width)
       elif evolver.TILE_FLIP:
           self.chromosome = TileFlipGenome(self.n_tiles, evolver.map_width)
       elif evolver.PRIMITIVES:
           self.chromosome = PatternGenome(self.n_tiles, evolver.map_width,
                                           enums.MaterialEnum.GRASS.value.index)
       elif evolver.SIMPLEX_NOISE:
           self.chromosome = SimplexNoiseGenome(self.n_tiles, evolver.map_width)
       elif evolver.RIVER_BOTTLENECK_BASELINE:
           self.chromosome = RiverBottleneckGenome(self.n_tiles, evolver.map_width, lava_border_width=self.TERRAIN_BORDER)
       elif evolver.RESOURCE_NICHES_BASELINE:
           self.chromosome = ResourceNichesGenome(self.n_tiles, evolver.map_width, self.TERRAIN_BORDER)
       elif evolver.BOTTLENECKED_RESOURCE_NICHES_BASELINE:
           self.chromosome = ResourceNichesGenome(self.n_tiles, evolver.map_width, self.TERRAIN_BORDER, access_bottleneck=True)
       elif evolver.MAZE_BASELINE:
           self.chromosome = MazeGenome(self.n_tiles, evolver.map_width)

       self.chromosome.gen_map()
       self.validate_map()
       self.score_hists = []
#      self.feature_hists = {}
       self.age = -1
       self.ALPs = []
   #     self.map_arr = None
   #     self.multi_hot = None
   #     self.atk_mults = None

   def update_fitness(self, new_f, ALP, config):
      self.score_hists.append(new_f)

      if ALP:
         score = self.score_hists[-2] - self.score_hists[-1]

         if self.ALPs is None:
            assert self.age == 0
            self.ALPs = []
         self.ALPs.append(score)
         #        score = abs(np.mean(self.ALPs))
         score = np.mean(self.ALPs)
         self.fitness.setValues([-score])
         self.fitness_stds = [np.std([self.ALPS]) for _ in self.fitness.getValues()]
      else:
         score = np.mean(self.score_hists)
         self.fitness_stds = [np.std([self.score_hists]) for _ in self.fitness.getValues()]

      if len(self.score_hists) >= config.ROLLING_FITNESS:
         self.score_hists = self.score_hists[-config.ROLLING_FITNESS:]

      if ALP:
         if len(self.ALPs) >= config.ROLLING_FITNESS:
            self.ALPs = self.ALPs[-config.ROLLING_FITNESS:]

      self.fitness.setValues([-score])


   def clone(self, evolver):
          child = copy.deepcopy(self)
          child.score_hists = []
   #      child.feature_hists = {}
          child.age = -1
          child.idx = None
          child.ALPs = []

          return child

   def mutate(self):
#     old_map = self.chromosome.map_arr
      self.chromosome.mutate()
      self.chromosome.gen_map()
      self.validate_map()
#     print("Number of new tiles after map re-generation:", (old_map != self.chromosome.map_arr).sum())
      self.iterable = self.chromosome.get_iterable()
      self.features = []

   def validate_map(self):
 #    if not hasattr(self.chromosome, 'multi_hot'):
 #        multi_hot = None
 #    else:
 #        multi_hot = self.chromosome.multi_hot
      # Let the chromosome sort itself out if applicable
      map_arr = self.chromosome.map_arr
      assert map_arr.shape[0] == map_arr.shape[1] == self.chromosome.map_width
      self.chromosome.validate_map(self.SPAWN_IDX, self.FOOD_IDX, self.WATER_IDX, self.NENT)
      self.add_border(map_arr, None)
      spawn_idxs = map_arr == self.SPAWN_IDX
      food_idxs = map_arr == self.FOOD_IDX
      water_idxs = map_arr == self.WATER_IDX
      spawn_points = np.vstack(np.where(spawn_idxs)).transpose()
      # ad hoc replace shitty scrub

#     if not self.GRIDDLY:
      map_arr = np.where(map_arr == self.SCRUB_IDX, self.EMPTY_IDX, map_arr)

      n_spawns = len(spawn_points)
      n_food = (1 * food_idxs).sum()
      n_water = (1 * water_idxs).sum()
      if not self.SINGLE_SPAWN:
          if n_spawns < 1 or n_food < self.NENT or n_water < self.NENT:
              self.valid_map = False
          else:
              self.valid_map = True
      else:
          if n_spawns < 1 or n_food < self.NENT or n_water < self.NENT:
              self.valid_map = False
          else:
              self.valid_map = True
          # In single-spawn mode, remove duplicate spawn tiles
          sp = spawn_points[:-1]
          np.random.shuffle(sp)
          sp = sp.T
          map_arr[sp[0], sp[1]] = self.EMPTY_IDX
#         for i in range(n_spawns - 1):
#             sp = spawn_points[i]
#             map_arr[sp[0], sp[1]] = self.EMPTY_IDX
      self.chromosome.map_arr = map_arr


   def add_border(self, map_arr, multi_hot=None):
       b = self.TERRAIN_BORDER
       # the border must be lava
       map_arr[0:b, :]= enums.MaterialEnum.LAVA.value.index
       map_arr[:, 0:b]= enums.MaterialEnum.LAVA.value.index
       map_arr[-b:, :]= enums.MaterialEnum.LAVA.value.index
       map_arr[:, -b:]= enums.MaterialEnum.LAVA.value.index

       if multi_hot is not None:
           multi_hot[:, 0:b, :]= -1
           multi_hot[:, :, 0:b]= -1
           multi_hot[:, -b:, :]= -1
           multi_hot[:, :, -b:]= -1
