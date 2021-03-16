from pdb import set_trace as T
import numpy as np

from forge.blade import core
from forge.blade.lib import enums
from forge.blade.lib import material
from forge.blade.lib import material

import os
import time

def loadTiled(tiles, fPath, materials, config, map_arr=None):
   if map_arr is not None:
      idxMap = map_arr
   else:
      idxMap = np.load(fPath)
   assert tiles.shape == idxMap.shape
   for r, row in enumerate(idxMap):
      for c, idx in enumerate(row):
#        mat  = materials[idx]
#        tile = tiles[r, c]

#        tile.mat      = mat()
#        tile.ents     = {}

#        tile.state    = mat()
#        tile.capacity = tile.mat.capacity
#        tile.tex      = mat.tex

#        tile.nEnts.update(0)
#        tile.index.update(tile.state.index)

          mat = materials[idx]
          tile = tiles[r, c]
          tile.reset(mat, config)
   return tiles
#class Map:
#   def __init__(self, realm, config):
#      sz              = config.TERRAIN_SIZE
#      self.shape      = (sz, sz)
#      self.config     = config
#      self.map_arr = None
from forge.blade.lib import material


class Map:
   '''Map object representing a list of tiles
   
   Also tracks a sparse list of tile updates
   '''
   def __init__(self, config, realm):
      self.config = config
      self.map_arr = None

      sz          = config.TERRAIN_SIZE
      self.tiles  = np.zeros((sz, sz), dtype=object)

      for r in range(sz):
         for c in range(sz):
           #self.tiles[r, c] = core.Tile(realm, config, enums.Grass, r, c, 'grass')
            self.tiles[r, c] = core.Tile(config, realm, r, c)


   def set_map(self, realm, idx, map_arr):
      self.idx = idx
      self.map_arr = map_arr
#     self.reset(self.idx, realm, map_arr=map_arr)

   def reset(self, realm, idx, map_arr=None):
#     self.idx = idx
      materials = {mat.index: mat for mat in material.All}
#     materials = {mat.index: mat for mat in material.All}
      fName     = self.config.ROOT + str(idx) + '/map.npy'#+ self.config.PATH_MAP_SUFFIX
      if map_arr is not None:
         self.map_arr = map_arr
      if self.map_arr is not None:
         self.tiles = loadTiled(self.tiles, fName,  materials, self.config, map_arr=self.map_arr)
      # shittily loading vanilla map from the hard drive like a pleb
      else:
#        loadTiled(self.tiles, fName, materials)

#         materials = {mat.index: mat for mat in material.All}
#         fPath  = os.path.join(self.config.PATH_MAPS,
#               self.config.PATH_MAP_SUFFIX.format(idx))
          self.tiles = loadTiled(self.tiles, fName, materials, self.config)


      self.updateList = set()

   def harvest(self, r, c):
      self.updateList.add(self.tiles[r, c])
      return self.tiles[r, c].harvest()

   def inds(self):
      return np.array([[j.index.val for j in i] for i in self.tiles])

   @property
   def packet(self):
       '''Packet of degenerate resource states'''
       missingResources = []
       for e in self.updateList:
           missingResources.append(e.pos)
       return missingResources

   @property
   def repr(self):
      '''Flat matrix of tile material indices'''
#     for row in self.tiles:
#         for t in row:
#             if not hasattr(t, 'mat'):
#                 T()
      rep = [[t.mat.index if 0 <t.mat.index < 16 else 1 for t in row] for row in self.tiles]
#     rep = [[t.mat.index for t in row] for row in self.tiles]


#     rep = [[t.index.val + 1 for t in row] for row in self.tiles]
#     rep = [[0 for t in row] for row in self.tiles]
#     rep = [[np.random.randint(5, 7) for t in row] for row in self.tiles]
      return rep

#   def reset(self, realm, idx):
#      '''Reuse the current tile objects to load a new map'''
#      self.updateList = set()
#
#      materials = {mat.index: mat for mat in material.All}
#      fPath  = os.path.join(self.config.PATH_MAPS,
#            self.config.PATH_MAP_SUFFIX.format(idx))
#      for r, row in enumerate(np.load(fPath)):
#         for c, idx in enumerate(row):
#            mat  = materials[idx]
#            tile = self.tiles[r, c]
#            tile.reset(mat, self.config)
#=======
#   def reset(self, realm, idx):
#      '''Reuse the current tile objects to load a new map'''
#      self.updateList = set()
#
#      materials = {mat.index: mat for mat in material.All}
#      fPath  = os.path.join(self.config.PATH_MAPS,
#            self.config.PATH_MAP_SUFFIX.format(idx))
#      for r, row in enumerate(np.load(fPath)):
#         for c, idx in enumerate(row):
#            mat  = materials[idx]
#            tile = self.tiles[r, c]
#            tile.reset(mat, self.config)
#>>>>>>> 17f0ddfd1c21ba37d2a5bb44eca6fe7a18aba382

   def step(self):
      '''Evaluate updatable tiles'''
      for e in self.updateList.copy():
         if not e.depleted:
            self.updateList.remove(e)
         e.step()

#<<<<<<< HEAD
#   def harvest(self, r, c):
#      '''Called by actions that harvest a resource tile'''
#      self.updateList.add(self.tiles[r, c])
#      return self.tiles[r, c].harvest()
#=======
   def harvest(self, r, c, deplete=True):
      '''Called by actions that harvest a resource tile'''
      if deplete:
         self.updateList.add(self.tiles[r, c])
      return self.tiles[r, c].harvest(deplete)
#>>>>>>> 17f0ddfd1c21ba37d2a5bb44eca6fe7a18aba382
