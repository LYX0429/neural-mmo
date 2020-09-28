
from forge.blade.lib.utils import staticproperty

from forge.blade.lib import utils, enums
from forge.blade.io.action.node import Node, NodeType
from forge.blade.io.action.static import Direction, Fixed
from forge.blade.io.stimulus.static import Stimulus
from forge.blade.core.tile import Tile

from ray.rllib.utils.spaces.flexdict import FlexDict
from ray.rllib.utils.spaces.repeated import Repeated
from collections import defaultdict
import gym

import numpy as np
np.set_printoptions(threshold=np.inf, linewidth=200)

class Action(Node):
    nodeType = NodeType.SELECTION

    @staticproperty
    def edges():
#       print('Action edges pcg')
        #return [Move, Attack, Exchange, Skill]

        return [Move, Terraform]

    @staticproperty
    def n():
        return len(Action.arguments)

    def args(stim, entity, config):
        return Static.edges
    #Called upon module import (see bottom of file)
    #Sets up serialization domain
    def hook():
        idx = 0
        arguments = []

        for action in Action.edges:
            for args in action.edges:
                if not 'edges' in args.__dict__:
                    continue

                for arg in args.edges:
                    arguments.append(arg)
                    arg.serial = tuple([idx])
                    arg.idx = idx
                    idx += 1
        Action.arguments = arguments

class Terraform(Node):
    nodeType = NodeType.SELECTION
    @staticproperty
    def n():
       #print('len terraform args', Terraform.arguments)
        return len(Terraform.arguments)

    @staticproperty
    def edges():
        return [Terrain]

    @staticproperty
    def leaf():
        return True

    def call(world, entity, terr):
       #print(dir(world.env))
       #print('world', world.env.tiles)

       #print('pcg world ', world.env.np())
       #print(dir(entity))
        x, y = trg_pos = entity.base.pos
        trg_tile = world.env.tiles[trg_pos]
       #print('terraform target tile', trg_pos, trg_tile)
        if trg_tile.terraformable:
            trg_tile.terraform(world.config, terr.terrain_type)
           #print('terraforming tile', x, y)
       #    if trg_tile.mat.harvestable:
       #       #print('pcg world ', world.env.np())
       #        print('harvesting tile', trg_tile.r, trg_tile.c)
       #       #print('world.env object ', world.env)
       #       #world.env.step()
       #        world.env.harvest(trg_tile.r, trg_tile.c)
       #       #print('pcg world ', world.env.np())
       #print('terraform terrain', terr)
       #tile = Tile(world.config, terr.terrain_type, x, y, 1, None)
       #world.env.tiles[trg_pos] =  tile
#       print(dir(world.env.tiles))
        return 1

class Terrain(Node):
    argType = Fixed

    @staticproperty
    def edges():
        return [Water, Grass, Scrub, Stone, Forest, Lava]#Orerock

    def args(stim, entity, config):
        return Terrain.edges

#   @staticmethod
#   def n():
#       return len(enums.Material)

#   def args(stim, entity, config):
#       print('enums.Material', enums.Material)
#       raise Exception
#       return enums.Material

class Water(Node):
    terrain_type = enums.Water

class Grass(Node):
    terrain_type = enums.Grass

class Scrub(Node):
    terrain_type = enums.Scrub

class Lava(Node):
    terrain_type = enums.Lava

class Forest(Node):
    terrain_type = enums.Forest

class Stone(Node):
    terrain_type = enums.Stone

class Orerock(Node):
    terrain_type = enums.Orerock

class Move(Node):
    priority = 0
    nodeType = NodeType.SELECTION
    def call(world, entity, direction):
        r, c = entity.base.pos
        entity.history.lastPos = (r, c)
        rDelta, cDelta = direction.delta
        rNew, cNew = r+rDelta, c+cDelta
       #if world.env.tiles[rNew, cNew].state.index in enums.IMPASSIBLE:
       #    return
        if not utils.inBounds(rNew, cNew, world.shape):
            return
        if entity.status.freeze > 0:
           return

#       print('entity in move call', entity)

#       entity.base.pos = (rNew, cNew)
        entity.base.r.update(rNew)
        entity.base.c.update(cNew)
        entID = entity.entID

        r, c = entity.history.lastPos
        # Not necessary since we've replaced the tile (and everything on it??)?
        world.env.tiles[r, c].delEnt(entID)

        r, c = entity.base.pos
        world.env.tiles[r, c].addEnt(entID, entity)

    @staticproperty
    def edges():
       return [Direction]

    @staticproperty
    def leaf():
       return True


#Neural MMO observation space
def observationSpace(config):
   obs = FlexDict({})

   for entity in sorted(Stimulus.values()):
      attrDict = FlexDict({})

      for attr in sorted(entity.values()):
         attrDict[attr] = attr(config).space
      n           = entity.N(config)
      obs[entity] = Repeated(attrDict, max_len=n)

   return obs

#Neural MMO action space
def actionSpace(config):
   atns = FlexDict(defaultdict(FlexDict))

   for atn in sorted(Action.edges):
      for arg in sorted(atn.edges):
         n              = arg.N(config)
         atns[atn][arg] = gym.spaces.Discrete(n)

   return atns

Action.hook()
