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

   @property
   def static(self):
      '''No updates needed'''
      assert self.capacity <= self.mat.capacity
      return self.capacity == self.mat.capacity

   def reset(self, mat, config):
      self.state  = mat(config)
      self.mat    = mat(config)

      self.capacity = self.mat.capacity
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
      if (not self.static and 
            np.random.rand() < self.mat.respawn):
         self.capacity += 1

      if self.static:
         self.state = self.mat
         self.index.update(self.state.index)

   def harvest(self):
      if self.capacity == 0:
         return False
      elif self.capacity >= 1:
         self.state = self.mat.degen(self.config)
         self.index.update(self.state.index)
      self.capacity -= 1
      return True
      return self.mat.dropTable.roll()

   def terraform(self, config, mat):
       super().__init__(Stimulus.Tile, config)
       self.mat = mat()
       self.state = mat()
      #print(self.__class__)
      #print(dir(self))
      #print(dir(self.mat))
      #if self.mat.harvestable:
      #    print('harvesting tile!')
      #    for i in range(1):
      #        self.harvest()
