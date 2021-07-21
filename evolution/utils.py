@ray.remote
class Counter:
   ''' When using rllib trainer to train and simulate on evolved maps, this global object will be
   responsible for providing unique indexes to parallel environments.'''
   def __init__(self, config):
      self.count = 0
   def get(self):
      self.count += 1

      if self.count == config.N_EVO_MAPS:
          self.count = 0

      return self.count
   def set(self, i):
      self.count = i - 1

@ray.remote
class Stats:
   def __init__(self, config):
      self.stats = {}
      self.mults = {}
      self.headers = ['hunting', 'fishing', 'constitution', 'range', 'mage', 'melee', 'defense', 'mining', 'woodcutting', 'wilderness']
      self.config = config
   def add(self, stats, mapIdx):
      if config.RENDER:
         print(self.headers)
         print(stats)
         print(calc_differential_entropy(stats))

         return

      if mapIdx not in self.stats:
         self.stats[mapIdx] = [stats]
      else:
         self.stats[mapIdx].append(stats)
   def get(self):
      return self.stats
   def reset(self):
      self.stats = {}
   def get_headers(self, headers=None):
      if not headers:
         return self.headers

      if not self.headers:
         self.headers = headers

      return self.headers
   def add_mults(self, g_hash, mults):
      self.mults[g_hash] = mults
   def get_mults(self, g_hash):
      return self.mults[g_hash]
