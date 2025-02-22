from pdb import set_trace as T

class Material:
   harvestable = False
   capacity    = 1
   def __init__(self, config):
      pass

   def __eq__(self, mtl):
      return self.index == mtl.index

   def __equals__(self, mtl):
      return self == mtl

class Lava(Material):
   tex   = 'lava'
   index = 0

class Water(Material):
   tex   = 'water'
   index = 1

class Grass(Material):
   tex   = 'grass'
   index = 2

class Scrub(Material):
   tex = 'scrub'
   index = 3

class Forest(Material):
   tex   = 'forest'
   index = 4

   harvestable = True
   degen       = Scrub

   def __init__(self, config):
      self.capacity = config.FOREST_CAPACITY
      self.respawn  = config.FOREST_RESPAWN
      #self.dropTable = DropTable.DropTable()

class Spawn(Material):
   index = 8
   tex = 'spawn'


class Stone(Material):
   tex   = 'stone'
   index = 5

#class Orerock(Material):
#   tex   = 'iron_ore'
#   index = 6
#
#   harvestable = True
#   degen       = Stone
#
#   def __init__(self, config):
#      self.capacity = config.OREROCK_CAPACITY
#      self.respawn  = config.OREROCK_RESPAWN
#      #self.dropTable = systems.DropTable()
#      #self.dropTable.add(ore.Copper, 1)


class Orerock(Material):
   index = 6
   degen = Grass
   tex = 'iron_ore'
   capacity = 1
   respawn = 0.025
   def __init__(self, config):
      super().__init__(config)
      self.harvestable = True
      #self.dropTable = systems.DropTable()
      #self.dropTable.add(ore.Copper, 1)

class Tree(Material):
   index = 7
   degen = Forest
   tex = 'tree'
   capacity = 1
   respawn = 0.025
   def __init__(self, config):
      super().__init__(config)
      self.harvestable = True

class Meta(type):
   def __iter__(self):
      yield from self.materials

   def __contains__(self, mtl):
      if isinstance(mtl, Material):
         mtl = type(mtl)
      return mtl in self.materials

class All(metaclass=Meta):
   materials = {Lava, Water, Grass, Scrub, Forest, Stone, Orerock, Tree, Spawn}

class Impassible(metaclass=Meta):
   materials = {Lava, Stone}

class Habitable(metaclass=Meta):
   materials = {Grass, Scrub, Forest, Spawn, Orerock, Tree}


