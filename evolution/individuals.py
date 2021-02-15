import neat
import neat.nn
import torch
from torch.nn import Conv2d
import copy
import numpy as np
from forge.blade.lib import enums
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
        mask = np.random.randint(0, 2, (self.nn.weights.shape))
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
    evolver = None
    ''' A wrapper class for a NEAT genome, which smuggles in other evolvable params for NMMO,
    beyond the map.'''
    def __init__(self, key):
        super().__init__(key)
        self.map_arr = None
        self.multi_hot = None
        self.atk_mults = gen_atk_mults()
        self.age = 0

    def configure_crossover(self, parent1, parent2, config):
        super().configure_crossover(parent1, parent2, config)
        mults_1, mults_2 = parent1.atk_mults, parent2.atk_mults
        self.atk_mults, _ = mate_atk_mults(mults_1, mults_2, single_offspring=True)

    def mutate(self, config):
        super().mutate(config)
        self.atk_mults = mutate_atk_mults(self.atk_mults)
        self.age = 0

    def clone(self):
        child = copy.deepcopy(self)
        child.age = 0
        child.map_arr = None
        child.multi_hot = None

        return child

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


class Individual():
    def __init__(self, rank, evolver, mysterious_class=None):
        self.fitness = Fitness()
        self.idx = rank
        self.n_tiles = len(evolver.TILE_PROBS)
        self.SPAWN_IDX = evolver.SPAWN_IDX
        self.NENT = evolver.config.NENT
        self.TERRAIN_BORDER = evolver.config.TERRAIN_BORDER
        if evolver.CPPN:
            #FIXME: yeesh
            self.chromosome = evolver.chromosomes[self.idx]
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
