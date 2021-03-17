import neat
from numpy.random import default_rng
import neat.nn
import torch
from torch.nn import Conv2d
import copy
import numpy as np
from forge.blade.lib import enums
from evolution.paint_terrain import Line, Rectangle, RectanglePerimeter, Circle, CirclePerimeter, Gaussian
from pdb import set_trace as T
from qdpy.phenotype import Individual, Fitness, Features
from evolution.paint_terrain import PRIMITIVE_TYPES

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


MELEE_MIN = 0.4
MELEE_MAX = 1.4
MAGE_MIN = 0.6
MAGE_MAX = 1.6
RANGE_MIN = 0.2
RANGE_MAX = 1


#NOTE: had to move this multiplier stuff outside of the evolver so the CPPN genome could incorporate them
# without pickling error.
#TODO: move these back into evolver class?
def gen_atk_mults():
    # generate melee, range, and mage attack multipliers for automatic game-balancing
    #atks = ['MELEE_MULT', 'RANGE_MULT', 'MAGE_MULT']
    #mults = [(atks[i], 0.2 + np.random.random() * 0.8) for i in range(3)]
    #atk_mults = dict(mults)
    # range is way too dominant, always

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


    def get_iterable(self):
        return []

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(-1.01)

    if type(m) == torch.nn.Conv1d:
        torch.nn.init.orthogonal_(m.weight)

class NeuralCA(torch.nn.Module):

    N_HIDDEN = 10
    N_CHAN = 9
    N_WEIGHTS = 2 * N_HIDDEN * N_CHAN * 3 * 3 + N_HIDDEN + 2 * N_HIDDEN + N_HIDDEN + N_HIDDEN * N_CHAN + N_CHAN + 1
    def __init__(self, n_chan):
        #FIXME
#       print('NeuralCA has {} input channels'.format(n_chan))
        assert self.N_CHAN == n_chan
        super().__init__()
        m = self.N_HIDDEN
        self.l0 = Conv2d(n_chan, 2 * m, 3, 1, 1, bias=True, padding_mode='circular')
        self.l1 = Conv2d(2 * m, m, 1, 1, 0, bias=True)
        self.l2 = Conv2d(m, n_chan, 1, 1, 0, bias=True)
        self.layers = [self.l0, self.l2, self.l2]
        self.apply(init_weights)
        self.n_passes = np.random.randint(1, 5)
        self.weights = self.get_init_weights()

    def forward(self, x):
        x = torch.Tensor(x).unsqueeze(0)
        for _ in range(max(1, int(min(200, self.n_passes)))):
            x = self.l0(x)
            x = torch.nn.functional.relu(x)
            x = self.l1(x)
            x = torch.nn.functional.relu(x)
            x = self.l2(x)
            x = torch.sigmoid(x)

#       for _ in range(max(1, int(self.n_passes/2))):

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

    def get_init_weights(self):
        weights = []
        #   n_par = 0
        for lyr in self.layers:
            #       n_par += np.prod(lyr.weight.shape)
            #       n_par += np.prod(lyr.bias.shape)
            weights.append(lyr.weight.view(-1).detach().numpy())
            weights.append(lyr.bias.view(-1).detach().numpy())
        init_weights = np.hstack(weights)

        return init_weights

class CAGenome():
    def __init__(self, n_tiles, map_width, seed):
        self.nn = NeuralCA(n_tiles)
#       self.seed = seed.reshape(1, *seed.shape)
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
        self.multi_hot = self.nn(self.seed).squeeze(0).detach().numpy()
        self.map_arr = self.multi_hot.argmax(axis=0)

    # def configure_crossover(self, parent0, parent2, config):
    #    super().configure_crossover(parent0, parent2, config)
    #    mults_0, mults_2 = parent1.atk_mults, parent2.atk_mults
    #    self.atk_mults, _ = mate_atk_mults(mults_0, mults_2, single_offspring=True)

    def mutate(self):

        noise = np.random.random(self.nn.weights.shape) * self.epsilon - (self.epsilon / 2)
        n_mut = max(1, int(self.rng.exponential(scale=1.0, size=1)))
        x = np.argwhere(noise)
        idxs = np.random.choice(x.shape[0], n_mut, replace=False)
        x = x[idxs]
        weights = self.nn.weights
        weights[x] = noise[x]
#       mask = (np.random.random(size=(self.nn.weights.shape)) < 5/noise.size) * 1
#       mutation = np.multiply(mask, noise)
#       weights = self.nn.weights + mutation
        self.nn.n_passes += int(np.random.uniform(0, 2.5))
        self.nn.set_weights(weights)
        self.atk_mults = mutate_atk_mults(self.atk_mults)
        self.age = -1

    def clone(self):
        child = copy.deepcopy(self)
        child.age = -1

        return child

    def get_iterable(self):
        return np.hstack((self.nn.weights, [self.nn.n_passes]))


class DefaultGenome(neat.genome.DefaultGenome, Genome):
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

        cppn = neat.nn.FeedForwardNetwork.create(self, self.neat_config)
        #       if self.config.NET_RENDER:
        #          with open('nmmo_cppn.pkl', 'wb') a
        multi_hot = np.zeros((self.n_tiles, self.map_width, self.map_height), dtype=np.float)
        map_arr = np.zeros((self.map_width, self.map_height), dtype=np.uint8)

        for x in range(self.map_width):
            for y in range(self.map_height):
                # a decent scale for NMMO
                x_i, y_i = x * 2 / self.map_width - 1, y * 2 / self.map_width - 1
                x_i, y_i = x_i * 2, y_i * 2
                v = cppn.activate((x_i, y_i))

           #    if self.config.THRESHOLD:
           #        # use NMMO's threshold logic
           #        assert len(v) == 1
           #        v = v[0]
           #        v = self.map_generator.material_evo(self.config, v)
           #    else:
                # CPPN has output channel for each tile type; take argmax over channels
                # also a spawn-point tile
                assert len(v) == self.n_tiles
                multi_hot[:, x, y] = v
                # Shuffle before selecting argmax to prevent bias for certain tile types in case of ties
                v = np.array(v)
               #v = np.random.choice(np.flatnonzero(v == v.max()))
                v = np.argmax(v)
                map_arr[x, y] = v
        self.map_arr = map_arr
        self.multi_hot = multi_hot

        return map_arr, multi_hot


class PatternGenome(Genome):
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

#       self.update_features()
        return self.paint_map()

#   def update_features(self):
#       self.features = [0, 0]
#       for p in self.patterns:
#           if isinstance(p, (Line, Rectangle, RectanglePerimeter)):
#               self.features[0] += 1
#           elif isinstance(p, (CirclePerimeter, Circle, Gaussian)):
#               self.features[1] += 1

    def get_iterable(self):
        # each pattern has: type, intensity, p1, p2, p3, p4
        it = np.zeros(shape=())

        return []

#       #FIXME: hack
#       return [self.__hash__]

    def mutate(self):
        super().mutate()
        n_patterns = len(self.patterns)
        n_add = max(0, int(self.rng.exponential(scale=1.0, size=1)))
        n_add = min(self.max_patterns - n_patterns, n_add)
        n_del = max(0, int(self.rng.exponential(scale=1.0, size=1)))
        n_del = min(n_patterns - 1, int(self.rng.exponential(scale=1.0, size=1)))
        n_mut = max(1, int(self.rng.exponential(scale=1.0, size=1)))
#       print('n_add: {}, n_mut: {}, n_del: {}'.format(n_add, n_mut, n_del))
        mutees = np.random.choice(self.patterns, n_mut)
        for m in mutees:
            m.mutate()
        for i in range(n_del):
            self.patterns.pop(np.random.randint(n_patterns-i))
        new_types = np.random.choice(PRIMITIVE_TYPES, n_add)
        [self.patterns.append(n.generate(n,
                                         tile_i=np.random.randint(0, self.n_tiles-1),
                                         intensity=np.random.random(),
                                         n_tiles=self.n_tiles,
                                         map_width=self.map_width)) for n in new_types]
        self.multi_hot = None
#       self.flat_map = None
#       self.update_features()
#       print('added {} patterns, mutated {}, deleted {}'.format(n_add, n_mut, n_del))


    def paint_map(self):
#       if hasattr(self, 'flat_map') and self.flat_map is not None:
#           return self.flat_map, self.multi_hot
        multi_hot = np.zeros((self.n_tiles, self.map_width, self.map_width))
        multi_hot[self.default_tile, :, :] = 1e-10

        for p in self.patterns:
            p.paint(multi_hot)
        map_arr = np.argmax(multi_hot, axis=0)
        self.map_arr, self.multi_hot = map_arr, multi_hot

#       return map_arr, multi_hot

class TileFlipGenome(Genome):
    def __init__(self, n_tiles, map_width):
        super().__init__(n_tiles, map_width)
        self.map_arr = np.random.randint(0, n_tiles, (map_width, map_width))
        self.rng = default_rng()

    def mutate(self):
        map_width = self.map_width
        n_muts = max(1, int(self.rng.exponential(scale=1.0, size=1)))
        new = np.random.randint(0, self.n_tiles, (n_muts))
        idxs = np.argwhere(np.zeros((map_width, map_width)) == 0)
        pos_idxs = np.random.choice(idxs.shape[0], n_muts, replace=False)
        mut_idxs = idxs[pos_idxs]
        self.map_arr[mut_idxs[:,0], mut_idxs[:,1]] = new

    def gen_map(self):
        pass

    def clone(self):
        child = TileFlipGenome(self.n_tiles, self.map_width)
        child.map_arr = self.map_arr.copy()

        return child

    def get_iterable(self):
        return self.map_arr.reshape(-1)


class LSystemGenome(Genome):
    def __init__(self, n_tiles, map_width):
        super().__init__(n_tiles, map_width)
        self.map_width = map_width
        self.n_tiles = n_tiles
        self.axiom_width = axiom_width = 2
        self.n_expansions = 5  # how many (stochastic) expansion rules per tile type
        self.axiom = np.random.randint(0, n_tiles, (axiom_width, axiom_width), dtype=np.uint8)
        #FIXME: not currently guaranteeing mutation will have an effect, since we may change only one expansion rule,
        # which then may not be used when expanding.
        self.expansions = dict([
            (i, np.repeat(np.random.randint(0, n_tiles, (1, 2, 2)), self.n_expansions, axis=0)) for i in range(n_tiles)
        ])

    def mutate(self):
        super().mutate()
        rnd = np.random.random()
        n_ax_mut = max(0, int(np.random.normal(0, 1)))
        if n_ax_mut == 0:
            min_rule_mut = 1
        else:
            min_rule_mut = 0
        n_rule_cells = len(self.expansions) * self.n_expansions * 2 * 2
        n_rule_mut = max(min_rule_mut, int(np.random.normal(0.5, 2)))
        for _ in range(n_ax_mut):
            x, y = (np.random.randint(0, self.axiom_width, (2)))
            # make sure to mutate to new tile type
            self.axiom[x, y] = (self.axiom[x, y] + np.random.randint(self.n_tiles)) % self.n_tiles
        for _ in range(n_rule_mut):
            rule_id = np.random.randint(len(self.expansions))
            expansion_id = np.random.randint(self.n_expansions)
            x, y = (np.random.randint(0, 2, (2)))
#           print(self.expansions.values())
            self.expansions[rule_id][expansion_id][x, y] = (self.expansions[rule_id][expansion_id][x, y] + np.random.randint(self.n_tiles)) % self.n_tiles
#           print(self.expansions.values())

#       self.axiom = np.where(np.random.random(self.axiom.shape) < 1/8, np.random.randint(0, self.n_tiles, (self.axiom_width, self.axiom_width)), self.axiom)
#       [self.expansions.update({k: np.where(np.random.random(expansion.shape) < 1.5/n_rule_cells, np.random.randint(0, self.n_tiles, (2, 2)), expansion)})
#        for (k, expansion) in self.expansions.items()]

    def gen_map(self):
        np.random.seed(420)
        map_arr = self.axiom.copy()
        while map_arr.shape[0] < self.map_width:
            new_arr = np.empty((map_arr.shape[0] * 2, map_arr.shape[1] * 2), dtype=np.uint8)
            for i in range(map_arr.shape[0]):
                for j in range(map_arr.shape[1]):
                    new_arr[2*i:2*i+2, 2*j:2*j+2] = self.expansions[map_arr[i, j]][np.random.randint(self.n_expansions)]
            map_arr = new_arr
        if (map_arr == self.map_arr).all():
            T()
        elif self.map_arr is not None:
            print((map_arr!=self.map_arr).sum())
#           print(self.expansions.values())
        self.map_arr = map_arr

#class EvoFitness(Fitness):
#    def __init__(self, val):
#       super().__init__(self, val)
#        self.values = None
##       self.valid = False
#
#    def dominates(self, fitness):
#        #       assert len(self.values) == 0 and len(fitness.values) == 1
#        assert len(self.values) == 1 == len(fitness.values)
#        my_fit = self.values[-1]
#        their_fit = fitness.values[-1]
#
#        return my_fit > their_fit


class EvoIndividual(Individual):
    def __eq__(self, other):
        # For qdpy when adding to container and checking against existing elites
        return self is other

    def __init__(self, iterable, rank, evolver):
        super().__init__(iterable=iterable)
        self.iterable = iterable
        self.rank = rank
        self.fitness = Fitness([0])
#       self.fitness = EvoFitness([0])
        # invalidate fitness
        self.fitness.delValues()
#       self.features = Features([0])
#       self.features.delValues()
#       self.reset()
#       self.fitness = Fitness()
        self.idx = rank
        self.n_tiles = len(evolver.TILE_PROBS)
        self.SPAWN_IDX = evolver.SPAWN_IDX
        self.FOOD_IDX = evolver.mats.FOREST.value.index
        self.WATER_IDX = evolver.mats.WATER.value.index
        self.NENT = evolver.config.NENT
        self.TERRAIN_BORDER = evolver.config.TERRAIN_BORDER
        if evolver.ALL_GENOMES:
            rnd = np.random.random()
            if rnd < 1/5:
                self.chromosome = DefaultGenome(self.idx, evolver.neat_config, self.n_tiles, evolver.map_width)
            elif rnd < 2/5:
                self.chromosome = PatternGenome(self.n_tiles, evolver.map_width,
                                                evolver.mats.MaterialEnum.GRASS.value.index)
            elif rnd < 3/5:
                self.chromosome = LSystemGenome(self.n_tiles, evolver.map_width)
            elif rnd < 4/5:
                self.chromosome = TileFlipGenome(self.n_tiles, evolver.map_width)
            else:
#               seed = np.random.random((self.n_tiles, evolver.map_width, evolver.map_width))
                seed = None
                self.chromosome = CAGenome(self.n_tiles, evolver.map_width)
        if evolver.CPPN:
            #FIXME: yeesh
            self.chromosome = DefaultGenome(self.idx, evolver.neat_config, self.n_tiles, evolver.map_width)
#           self.chromosome = evolver.chromosomes[self.idx]
        elif evolver.CA:
#           seed = np.random.random((self.n_tiles, evolver.map_width, evolver.map_width))
            seed = None
            self.chromosome = CAGenome(self.n_tiles, evolver.map_width, seed)
        elif evolver.LSYSTEM:
            self.chromosome = LSystemGenome(self.n_tiles, evolver.map_width)
        elif evolver.TILE_FLIP:
            self.chromosome = TileFlipGenome(self.n_tiles, evolver.map_width)
        elif evolver.PRIMITIVES:
            self.chromosome = PatternGenome(self.n_tiles, evolver.map_width,
                                            enums.MaterialEnum.GRASS.value.index)
        self.chromosome.gen_map()
        self.validate_map()
        self.score_hists = []
#       self.feature_hists = {}
        self.age = -1
        self.ALPs = []
    #     self.map_arr = None
    #     self.multi_hot = None
    #     self.atk_mults = None

    def clone(self, evolver):
        child = copy.deepcopy(self)
#       child = EvoIndividual(self.iterable[:], self.rank, evolver)
#       assert child is not self
#       child.fitness.delValues()
#       child.features.delValues()
#       child.fitness.values = None
#       child.fitness.valid = False
        child.score_hists = []
#       child.feature_hists = {}
        child.age = -1
#       child_chrom = self.chromosome.clone()
#       assert child_chrom is not self.chromosome
        child.idx = None
#       child.chromosome = child_chrom
#       assert child is not self
        child.ALPs = []

        return child

    def mutate(self):
        self.chromosome.mutate()
        self.chromosome.gen_map()
        self.validate_map()
        self.iterable = self.chromosome.get_iterable()
        self.features = []

    def validate_map(self):
 #      if not hasattr(self.chromosome, 'multi_hot'):
 #          multi_hot = None
 #      else:
 #          multi_hot = self.chromosome.multi_hot
        map_arr = self.chromosome.map_arr
        self.add_border(map_arr, None)
        spawn_idxs = map_arr == self.SPAWN_IDX
        food_idxs = map_arr == self.FOOD_IDX
        water_idxs = map_arr == self.WATER_IDX
        spawn_points = np.vstack(np.where(spawn_idxs)).transpose()
        n_spawns = len(spawn_points)
        n_food = (1 * food_idxs).sum()
        n_water = (1 * water_idxs).sum()
        if n_spawns < self.NENT or n_food < self.NENT or n_water < self.NENT:
            self.valid_map = False
        else:
            self.valid_map = True

#       if n_spawns >= self.NENT:
#           return map_arr
#       n_new_spawns = self.NENT - n_spawns

#       #     if multi_hot is not None:
#       #        spawn_idxs = k_largest_index_argsort(
#       #              multi_hot[enums.Material.SPAWN.value.index, :, :],
#       #              n_new_spawns)
#       #    #   map_arr[spawn_idxs[:, 0], spawn_idxs[:, 1]] = enums.Material.SPAWN.value.index
#       #     else:
#       border = self.TERRAIN_BORDER
#       #     spawn_idxs = np.random.randint(border, self.map_width - border, (2, n_new_spawns))
#       #     map_arr[spawn_idxs[0], spawn_idxs[1]] = self.SPAWN_IDX
#       b = self.TERRAIN_BORDER
#       all_idxs = np.vstack(np.where(map_arr[b:-b,b:-b]!=-1)).T
#       spawn_idxs = all_idxs[np.random.choice(all_idxs.shape[0], n_new_spawns + 1, replace=False)] + b
#       map_arr[spawn_idxs[:,0], spawn_idxs[:,1]] = self.SPAWN_IDX
#       while not (map_arr == self.SPAWN_IDX).sum() >= self.NENT:
#           print('Insufficient spawn points')
#           map_arr[np.random.choice(all_idxs[all_idxs.shape[0]], 1)] = self.SPAWN_IDX


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