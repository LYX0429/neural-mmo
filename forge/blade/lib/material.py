from pdb import set_trace as T
from forge.blade.systems import droptable
from forge.blade import item
from enum import Enum

class Material:
   capacity = 0

   def __init__(self, config):
      pass

   def __eq__(self, mtl):
      return self.index == mtl.index

   def __equals__(self, mtl):
      return self == mtl

class Lava(Material):
   tex      = 'lava'
   index    = 0

class Water(Material):
   tex      = 'water'
   index    = 1

   def __init__(self, config):
      self.deplete = __class__
      self.respawn  = 1.0

   def harvest(self):
      return droptable.Empty()

class Grass(Material):
   tex      = 'grass'
   index    = 2

class Scrub(Material):
   tex     = 'scrub'
   index   = 3

class Forest(Material):
   tex     = 'forest'
   index   = 4

   deplete = Scrub
   def __init__(self, config):
      self.respawn  = config.FOREST_RESPAWN

   def harvest(self):
      return droptable.Empty()

class Stone(Material):
   tex     = 'stone'
   index   = 5

class Slag(Material):
   tex     = 'slag'
   index   = 6

class Ore(Material):
   tex     = 'ore'
   index   = 7

   deplete = Stone
   def __init__(self, config):
      self.respawn  = config.ORE_RESPAWN

   def harvest(self):
      return droptable.Ammunition(item.Scrap)

class Stump(Material):
   tex     = 'stump'
   index   = 8

class Tree(Material):
   tex     = 'tree'
   index   = 9

   deplete = Stump
   def __init__(self, config):
      self.respawn  = config.TREE_RESPAWN

   def harvest(self):
      return droptable.Ammunition(item.Shaving) 

class Fragment(Material):
   tex     = 'fragment'
   index   = 10

class Crystal(Material):
   tex     = 'crystal'
   index   = 11

   deplete = Fragment
   def __init__(self, config):
      self.respawn  = config.CRYSTAL_RESPAWN

   def harvest(self):
      return droptable.Ammunition(item.Shard) 

class Weeds(Material):
   tex     = 'weeds'
   index   = 12

class Herb(Material):
   tex     = 'herb'
   index   = 13

   deplete = Weeds
   def __init__(self, config):
      self.respawn  = config.HERB_RESPAWN

   def harvest(self):
      return droptable.Consumable(item.Potion) 

class Ocean(Material):
   tex     = 'ocean'
   index   = 14

class Fish(Material):
   tex     = 'fish'
   index   = 15

   deplete = Ocean
   def __init__(self, config):
      self.respawn  = config.FISH_RESPAWN

   def harvest(self):
      return droptable.Consumable(item.Food)

class Spawn(Material):
   index = 16
   tex = 'spawn'


class Meta(type):
   def __iter__(self):
      yield from self.materials

   def __contains__(self, mtl):
      if isinstance(mtl, Material):
         mtl = type(mtl)
      return mtl in self.materials

class All(metaclass=Meta):
   materials = {
      Lava, Water, Grass, Scrub, Forest,
      Stone, Slag, Ore, Stump, Tree,
      Fragment, Crystal, Weeds, Herb, Ocean, Fish, Spawn}

class Impassible(metaclass=Meta):
   materials = {Lava, Water, Stone, Ocean, Fish}

class Habitable(metaclass=Meta):
   materials = {Grass, Scrub, Forest, Ore, Tree, Crystal, Weeds, Herb, Spawn}

class Harvestable(metaclass=Meta):
   materials = {Water, Forest, Ore, Tree, Crystal, Herb, Fish}


class MaterialEnum(Enum):
   SPAWN = Spawn
   FOREST = Forest
   WATER = Water
   LAVA = Lava