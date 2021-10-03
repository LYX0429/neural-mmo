from pdb import set_trace as TT
#import ray

# Deprecated, we now ask each remote env for its stats directly and so no longer need to go through global actors
#@ray.remote
#class Counter:
#   ''' When using rllib trainer to train and simulate on evolved maps, this global object will be
#   responsible for providing unique indexes to parallel environments.'''
#   def __init__(self, config):
#      self.count = 0
#   def get(self):
#      self.count += 1
#
#      if self.count == config.N_EVO_MAPS:
#          self.count = 0
#
#      return self.count
#   def set(self, i):
#      self.count = i - 1

#@ray.remote
#class Stats:
#   def __init__(self, config):
#      self.stats = {}
#      self.mults = {}
#      self.headers = ['hunting', 'fishing', 'constitution', 'range', 'mage', 'melee', 'defense', 'mining', 'woodcutting', #'wilderness']
#      self.config = config
#   def add(self, stats, mapIdx):
#      if config.RENDER:
#         print(self.headers)
#         print(stats)
#         print(calc_differential_entropy(stats))
#
#         return
#
#      if mapIdx not in self.stats:
#         self.stats[mapIdx] = [stats]
#      else:
#         self.stats[mapIdx].append(stats)
#   def get(self):
#      return self.stats
#   def reset(self):
#      self.stats = {}
#   def get_headers(self, headers=None):
#      if not headers:
#         return self.headers
#
#      if not self.headers:
#         self.headers = headers
#
#      return self.headers
#   def add_mults(self, g_hash, mults):
#      self.mults[g_hash] = mults
#   def get_mults(self, g_hash):
#     return self.mults[g_hash]

def get_eval_map_inds(map_archive, n_inds):
   """Given an archive of individuals corrsponding to map generators, select individuals to use for generating maps
    to evaluate."""
   assert n_inds % 2 == 0
   inds = map_archive['container']
   inds_lst = [i for i in inds]
   # NOTE: assuming single-objective
   inds_lst.sort(key= lambda x: x.fitness.getValues()[0])
   fit_inds, inds_lst = inds_lst[:n_inds//2], inds_lst[n_inds//2:]
   inds_lst.sort(key= lambda x: x.age, reverse=True)
   old_inds = inds_lst[:n_inds//2]
   return fit_inds + old_inds

# FIXME: avoid this redundancy!
gen_objective_names = [
   'MapTestText',
   'Lifespans',
   'L2',
   'Hull',
   'Differential',
   'Sum',
   'Discrete',
   'FarNearestNeighbor',
   'CloseNearestNeighbor',
   'AdversityDiversity',
   'AdversityDiversityTrgs',
   'InvL2',
]
genome_names = [
   'Baseline',
   'RiverBottleneckBaseline',
   'ResourceNichesBaseline',
   'BottleneckedResourceNichesBaseline',
   'Simplex',
   'NCA',
   'TileFlip',
   'CPPN',
   'Primitives',
   'L-System',
   'All',
]

def get_exp_shorthand(exp_name):
   '''A disgusting function for getting a shorthand for experiment names.'''
   # TODO: Do this in a nicer way, using dictionaries of experiment settings directly (?)
   exp_shorthand = ''
   found_genome_name = False
   for genome_name in genome_names:
#     if genome_name in exp_name:
      if 'gene-'+genome_name+'_' in exp_name:
         exp_shorthand += genome_name
         assert not found_genome_name
         found_genome_name = True
   assert found_genome_name
   found_obj_name = False
   for obj_name in gen_objective_names:
      if 'fit-'+obj_name+'_' in exp_name:
         exp_shorthand += ' ' + obj_name
         assert not found_obj_name
         found_obj_name = True
   assert found_obj_name
   if 'PAIRED' in exp_name:
      exp_shorthand += ' PAIRED'
   return exp_shorthand