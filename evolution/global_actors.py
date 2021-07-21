import ray

@ray.remote
class Counter:
    ''' When using rllib trainer to train and simulate on evolved maps, this global object will be
    responsible for providing unique indexes to parallel environments.'''
    def __init__(self, config):
        self.count = 0
        self.idxs = None

    def get(self):

        if not self.idxs:
            # Then we are doing inference and have set the idx directly

            return self.count
        idx = self.idxs[self.count % len(self.idxs)]
        self.count += 1

        return idx

    def set(self, i):
        # For inference
        self.count = i

    def set_idxs(self, idxs):
        self.count = 0
        self.idxs = idxs

@ray.remote
class Stats:
   def __init__(self, config):
      self.stats = {}
      self.mults = {}
      self.spawn_points = {}
      self.config = config

   def add(self, stats, mapIdx):
      if self.config.RENDER:
         return

      if mapIdx not in self.stats or 'skills' not in self.stats[mapIdx]:
         self.stats[mapIdx] = stats
      else:
         for (k, v) in stats.items():
             if k in self.stats:
                 self.stats[k].append(v)
             else:
                 self.stats[k] = [stats[k]]
   def get(self):
      return self.stats
   def reset(self):
      self.stats = {}
   def add_mults(self, g_hash, mults):
      self.mults[g_hash] = mults
   def get_mults(self, g_hash):
      if g_hash not in self.mults:
         return None

      return self.mults[g_hash]
   def add_spawn_points(self, g_hash, spawn_points):
      self.spawn_points[g_hash] = spawn_points
   def get_spawn_points(self, g_hash):
      return self.spawn_points[g_hash]
