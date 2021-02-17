import neat
import neat.nn
import torch
from torch.nn import Conv2d
import copy
import numpy as np
from forge.blade.lib import enums
from evolution.paint_terrain import Line, Rectangle, RectanglePerimeter, Circle, CirclePerimeter, Gaussian
from pdb import set_trace as T

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
    def __init__(self):
        self.atk_mults = gen_atk_mults()

    def clone(self):
        child = copy.deepcopy(self)

        return child

    def mutate(self):
        self.atk_mults = mutate_atk_mults(self.atk_mults)

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(-1.01)

    if type(m) == torch.nn.Conv1d:
        torch.nn.init.orthogonal_(m.weight)

class NeuralCA(torch.nn.Module):
    def __init__(self, n_chan):
        m = 15
        super().__init__()
        self.l0 = Conv2d(n_chan, 2 * m, 3, 1, 1, bias=True, padding_mode='circular')
        self.l1 = Conv2d(2 * m, m, 1, 1, 0, bias=True)
        self.l2 = Conv2d(m, n_chan, 1, 1, 0, bias=True)
        self.layers = [self.l0, self.l2, self.l2]
        self.apply(init_weights)
        self.n_passes = 1
        self.weights = self.get_init_weights()

    def forward(self, x):
        x = torch.Tensor(x).unsqueeze(0)
        for _ in range(max(1, int(self.n_passes))):
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
        self.n_passes = max(1, weights[n_el])
#       print('Neural CAL n_passes', self.n_passes)

    def get_init_weights(self):
        weights = []
        #   n_par = 0
        for lyr in self.layers:
            #       n_par += np.prod(lyr.weight.shape)
            #       n_par += np.prod(lyr.bias.shape)
            weights.append(lyr.weight.view(-1).detach().numpy())
            weights.append(lyr.bias.view(-1).detach().numpy())
        weights.append(self.n_passes)
        init_weights = np.hstack(weights)

        return init_weights

class CAGenome():
    def __init__(self, n_tiles, seed):
        self.nn = NeuralCA(n_tiles)
#       self.seed = seed.reshape(1, *seed.shape)
#       self.seed = np.random.randint(0, n_tiles, (n_tiles, map_width, map_width))
        self.seed = seed
        self.gen_map()
        self.atk_mults = gen_atk_mults()
        self.age = -1

    def gen_map(self):
        self.multi_hot = self.nn(self.seed).squeeze(0).detach().numpy()
        self.map_arr = self.multi_hot.argmax(axis=0)

    # def configure_crossover(self, parent0, parent2, config):
    #    super().configure_crossover(parent0, parent2, config)
    #    mults_0, mults_2 = parent1.atk_mults, parent2.atk_mults
    #    self.atk_mults, _ = mate_atk_mults(mults_0, mults_2, single_offspring=True)

    def mutate(self):
        noise = np.random.random(self.nn.weights.shape) * 2 - 1
        mask = (np.random.random(size=(self.nn.weights.shape)) < 0.1) * 1
        mutation = np.multiply(mask, noise)
        weights = self.nn.weights + mutation
        self.nn.set_weights(weights)
        self.gen_map()
        self.atk_mults = mutate_atk_mults(self.atk_mults)
        self.age = -1

    def clone(self):
        child = copy.deepcopy(self)
        child.age = -1

        return child


class DefaultGenome(neat.genome.DefaultGenome):
    ''' A wrapper class for a NEAT genome, which smuggles in other evolvable params for NMMO,
    beyond the map.'''
    def __init__(self, key, neat_config, map_width, n_tiles):
        super().__init__(key)
        self.configure_new(neat_config.genome_config)
        self.map_arr = None
        self.multi_hot = None
        self.atk_mults = gen_atk_mults()
        self.age = 0
        self.neat_config = neat_config
        self.map_width = self.map_height = map_width
        self.n_tiles = n_tiles
        self.gen_map()

    def configure_crossover(self, parent1, parent2, config):
        super().configure_crossover(parent1, parent2, config)
        mults_1, mults_2 = parent1.atk_mults, parent2.atk_mults
        self.atk_mults, _ = mate_atk_mults(mults_1, mults_2, single_offspring=True)

    def mutate(self):
        super().mutate(self.neat_config.genome_config)
        self.atk_mults = mutate_atk_mults(self.atk_mults)
        self.age = 0
        self.gen_map()

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
                v = np.random.choice(np.flatnonzero(v == v.max()))
                #              v = np.argmax(v)
                map_arr[x, y] = v
#       map_arr = self.validate_spawns(map_arr, multi_hot)
        self.map_arr = map_arr
        self.multi_hot = multi_hot

        return map_arr, multi_hot

class Fitness():
    def __init__(self):
        self.values = None
        self.valid = False

    def dominates(self, fitness):
#       assert len(self.values) == 0 and len(fitness.values) == 1
        assert len(self.values) == 1 == len(fitness.values)
        my_fit = self.values[-1]
        their_fit = fitness.values[-1]

        return my_fit > their_fit


class PatternGenome(Genome):
    def __init__(self, map_width, n_tiles, max_patterns, default_tile):
        super().__init__()
        self.map_width = map_width
        self.n_tiles = n_tiles
        self.max_patterns = max_patterns
        self.default_tile = default_tile
        self.pattern_templates = [Line, Rectangle, RectanglePerimeter, Gaussian, Circle, CirclePerimeter]
        #     self.pattern_templates = [Gaussian]
        self.weights =  [2/3,  1/3]
        # some MAP-Elites dimensions
        self.features = None
        self.flat_map = None
        self.multi_hot = None
        self.gen_map()

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
        self.patterns = np.random.choice(self.pattern_templates,
                                         np.random.randint(self.max_patterns), self.weights).tolist()
        self.features = [0, 0]  # Number of lines, circle perimeters

        for i, p in enumerate(self.patterns):
            #if p in [Line, Rectangle]:
            p = p.generate(p, self.n_tiles, self.map_width)
            self.patterns[i] = p
        self.update_features()

        return self.paint_map()

    def update_features(self):
        self.features = [0, 0]
        for p in self.patterns:
            if isinstance(p, (Line, Rectangle, RectanglePerimeter)):
                self.features[0] += 1
            elif isinstance(p, (CirclePerimeter, Circle, Gaussian)):
                self.features[1] += 1

    def mutate(self):
        super().mutate()
        n_patterns = len(self.patterns)
        if n_patterns == 0:
            add_ptrn = True
        else:
            add_ptrn = np.random.randint(0, min(5, self.max_patterns - n_patterns))
            for p in np.random.choice(self.patterns, np.random.randint(0, max(1, int(n_patterns//5)))):
                p.mutate()

        for i in range(0, min(5, n_patterns)):
            self.patterns.pop(np.random.randint(n_patterns-i))
        if add_ptrn and n_patterns < self.max_patterns:
            p = np.random.choice(self.pattern_templates)
            p = p.generate(p, self.n_tiles, self.map_width)
            self.patterns.append(p)
        self.flat_map = None
        self.multi_hot = None
        self.gen_map()
        #     self.update_features()

        return self.paint_map()


    def paint_map(self):
        if hasattr(self, 'flat_map') and self.flat_map is not None:
            return self.flat_map, self.multi_hot
        multi_hot = np.zeros((self.n_tiles, self.map_width, self.map_width))
        multi_hot[self.default_tile, :, :] = 1e-10

        for p in self.patterns:
            p.paint(multi_hot)
        map_arr = np.argmax(multi_hot, axis=0)
        self.map_arr, self.multi_hot = map_arr, multi_hot

        return map_arr, multi_hot


class Individual():
    def __init__(self, rank, evolver, mysterious_class=None):
        self.fitness = Fitness()
        self.idx = rank
        self.n_tiles = len(evolver.TILE_PROBS)
        self.SPAWN_IDX = evolver.SPAWN_IDX
        self.NENT = evolver.config.NENT
        self.TERRAIN_BORDER = evolver.config.TERRAIN_BORDER
        if evolver.ALL_GENOMES:
            rnd = np.random.random()
            if rnd < 1/3:
                self.chromosome = DefaultGenome(self.idx, evolver.neat_config, evolver.map_width, self.n_tiles)
            elif rnd < 2/3:
                self.chromosome = PatternGenome(evolver.map_width, self.n_tiles, evolver.max_primitives,
                                                enums.Material.GRASS.value.index)
            else:
                seed = np.random.random((self.n_tiles, evolver.map_width, evolver.map_width))
                self.chromosome = CAGenome(self.n_tiles, seed)
        if evolver.CPPN:
            #FIXME: yeesh
            self.chromosome = DefaultGenome(self.idx, evolver.neat_config, evolver.map_width, self.n_tiles)
#           self.chromosome = evolver.chromosomes[self.idx]
        elif evolver.CA:
            seed = np.random.random((self.n_tiles, evolver.map_width, evolver.map_width))
            self.chromosome = CAGenome(self.n_tiles, seed)
        self.validate_spawns()
        self.score_hists = []
        self.feature_hists = {}
        self.age = -1
        self.ALPs = []
    #     self.map_arr = None
    #     self.multi_hot = None
    #     self.atk_mults = None

    def clone(self):
        child = copy.deepcopy(self)
        child.features = None
        child.fitness.values = None
        child.fitness.valid = False
        child.score_hists = []
        child.feature_hists = {}
        child.age = -1
        child_chrom = self.chromosome.clone()
        child.idx = None
        child.chromosome = child_chrom
        child.ALPs = []

        return child

    def mutate(self):
        self.chromosome.mutate()
        self.validate_spawns()

    def validate_spawns(self):
        map_arr, multi_hot = self.chromosome.map_arr, self.chromosome.multi_hot
        self.add_border(map_arr, multi_hot)
        idxs = map_arr == self.SPAWN_IDX
        spawn_points = np.vstack(np.where(idxs)).transpose()
        n_spawns = len(spawn_points)

        if n_spawns >= self.NENT:
            return map_arr
        n_new_spawns = self.NENT - n_spawns

        #     if multi_hot is not None:
        #        spawn_idxs = k_largest_index_argsort(
        #              multi_hot[enums.Material.SPAWN.value.index, :, :],
        #              n_new_spawns)
        #    #   map_arr[spawn_idxs[:, 0], spawn_idxs[:, 1]] = enums.Material.SPAWN.value.index
        #     else:
        border = self.TERRAIN_BORDER
        #     spawn_idxs = np.random.randint(border, self.map_width - border, (2, n_new_spawns))
        #     map_arr[spawn_idxs[0], spawn_idxs[1]] = self.SPAWN_IDX
        b = self.TERRAIN_BORDER
        all_idxs = np.vstack(np.where(map_arr[b:-b,b:-b]!=-1)).T
        spawn_idxs = all_idxs[np.random.choice(all_idxs.shape[0], n_new_spawns + 1, replace=False)] + b
        map_arr[spawn_idxs[:,0], spawn_idxs[:,1]] = self.SPAWN_IDX
        if not (map_arr == self.SPAWN_IDX).sum() >= self.NENT:
            print('Insufficient spawn points')
            T()


    def add_border(self, map_arr, multi_hot=None):
        b = self.TERRAIN_BORDER
        # the border must be lava
        map_arr[0:b, :]= enums.Material.LAVA.value.index
        map_arr[:, 0:b]= enums.Material.LAVA.value.index
        map_arr[-b:, :]= enums.Material.LAVA.value.index
        map_arr[:, -b:]= enums.Material.LAVA.value.index

        if multi_hot is not None:
            multi_hot[:, 0:b, :]= -1
            multi_hot[:, :, 0:b]= -1
            multi_hot[:, -b:, :]= -1
            multi_hot[:, :, -b:]= -1
