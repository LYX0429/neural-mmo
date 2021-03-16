from pdb import set_trace as T
import numpy as np

from forge.blade.lib import material
from forge.blade.io.stimulus import Static

class Tile:
   def __init__(self, config, realm, r, c):
      self.config = config
      self.realm  = realm

      self.serialized = 'R{}-C{}'.format(r, c)

      self.r     = Static.Tile.R(realm.dataframe, self.serial, r)
      self.c     = Static.Tile.C(realm.dataframe, self.serial, c)
      self.nEnts = Static.Tile.NEnts(realm.dataframe, self.serial)
      self.index = Static.Tile.Index(realm.dataframe, self.serial, 0)

      realm.dataframe.init(Static.Tile, self.serial, (r, c))

   @property
   def serial(self):
      return self.serialized

   @property
   def repr(self):
      return ((self.r, self.c))

   @property
   def pos(self):
      return self.r.val, self.c.val

   @property
   def habitable(self):
      return self.mat in material.Habitable

   @property
   def vacant(self):
      return len(self.ents) == 0 and self.habitable

   @property
   def occupied(self):
      return not self.vacant

   @property
   def impassible(self):
      return self.mat in material.Impassible

   @property
   def lava(self):
      return self.mat == material.Lava

   def reset(self, mat, config):
      self.state    = mat(config)
      self.mat      = mat(config)

      self.depleted = False
      self.tex      = mat.tex
      self.ents     = {}

      self.nEnts.update(0)
      self.index.update(self.state.index)
 
   def addEnt(self, ent):
      assert ent.entID not in self.ents
      self.ents[ent.entID] = ent

   def delEnt(self, entID):
      assert entID in self.ents
      del self.ents[entID]

   def step(self):
      if not self.depleted or np.random.rand() >= self.mat.respawn:
         return

      self.depleted = False
      self.state    = self.mat

      self.index.update(self.state.index)

   def harvest(self, deplete=True):
      err1 = '{} is depleted'.format(self.state)
      err2 = '{} not harvestable'.format(self.state)

      assert not self.depleted, err1
      assert self.state in material.Harvestable, err2

      if deplete:
         self.depleted = True
         self.state    = self.mat.deplete(self.config)
         self.index.update(self.state.index)

      return self.mat.harvest()
