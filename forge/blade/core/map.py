from pdb import set_trace as T
import numpy as np

from forge.blade import core
from forge.blade.lib import enums
from forge.blade.lib import material

import os
#<<<<<<< HEAD
import time

def loadTiled(tiles, fPath, materials, config, map_arr=None):
   if map_arr is not None:
      idxMap = map_arr
   else:
      idxMap = np.load(fPath)
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
#=======

class Map:
   '''Map object representing a list of tiles
   
   Also tracks a sparse list of tile updates
   '''
   def __init__(self, config, realm):
      self.config = config
      self.map_arr = None

      sz          = config.TERRAIN_SIZE
      self.tiles  = np.zeros((sz, sz), dtype=object)
#>>>>>>> 1473e2bf0dd54f0ab2dbf0d05f6dbb144bdd1989

      for r in range(sz):
         for c in range(sz):
#<<<<<<< HEAD
           #self.tiles[r, c] = core.Tile(realm, config, enums.Grass, r, c, 'grass')
            self.tiles[r, c] = core.Tile(config, realm, r, c)


   def set_map(self, realm, idx, map_arr):
      self.idx = idx
      self.map_arr = map_arr
#     self.reset(self.idx, realm, map_arr=map_arr)

   def reset(self, realm, idx, map_arr=None):
#     self.idx = idx
      materials = dict((mat.value.index, mat.value) for mat in enums.MaterialEnum)
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
#=======
#           self.tiles[r, c] = core.Tile(config, realm, r, c)
#>>>>>>> 1473e2bf0dd54f0ab2dbf0d05f6dbb144bdd1989

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
      return [[t.mat.index for t in row] for row in self.tiles]

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

   def step(self):
      '''Evaluate updatable tiles'''
      for e in self.updateList.copy():
         if e.static:
            self.updateList.remove(e)
         e.step()

   def harvest(self, r, c):
      '''Called by actions that harvest a resource tile'''
      self.updateList.add(self.tiles[r, c])
      return self.tiles[r, c].harvest()
