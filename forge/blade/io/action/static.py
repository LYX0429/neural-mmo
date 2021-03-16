from pdb import set_trace as T
import numpy as np

from forge.blade import item
from forge.blade.lib import utils, material
from forge.blade.lib.utils import staticproperty
from forge.blade.systems import combat
from forge.blade.io.node import Node, NodeType
from forge.blade.io.stimulus import Static
import forge

class Fixed:
   pass

#ActionRoot
class Action(Node):
   nodeType = NodeType.SELECTION

   @staticproperty
   def edges():
      return [Move, Attack, Buy, InventoryAction]

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

class Move(Node):
   priority = 1
   nodeType = NodeType.SELECTION
   def call(env, entity, direction):
      r, c  = entity.pos
      entID = entity.entID
      entity.history.lastPos = (r, c)
      rDelta, cDelta = direction.delta
      rNew, cNew = r+rDelta, c+cDelta

      trg_pos = rNew, cNew
      tile = env.map.tiles[trg_pos]
      dest_idx = tile.mat.index
      #One agent per cell
      tile = env.map.tiles[rNew, cNew] 
      if tile.occupied and not tile.lava:
         return
      if entity.status.freeze > 0:
         return
#     #FIXME: hack. Check if tile is degenerated and thus passible.
#     hack_passible = tile.depleted
#     if dest_idx == material.Tree.index and not hack_passible:
#        return Woodcut.call(env, entity, trg_pos)
#     if dest_idx == material.Orerock.index:
#        if not hack_passible:
#           return Mine.call(env, entity, trg_pos)
#        else:
#           pass

#     if not utils.inBounds(rNew, cNew, env.shape):
      if not utils.inBounds(rNew, cNew, env.map.inds().shape):
         return

      if not tile.habitable and not tile.lava:
         return

      env.dataframe.move(Static.Entity, entID, (r, c), (rNew, cNew))
      entity.base.r.update(rNew)
      entity.base.c.update(cNew)

      env.map.tiles[r, c].delEnt(entID)
      env.map.tiles[rNew, cNew].addEnt(entity)

      if env.map.tiles[rNew, cNew].lava:
         entity.receiveDamage(None, entity.resources.health.val)

   @staticproperty
   def edges():
      return [Direction]

   @staticproperty
   def leaf():
      return True

class Direction(Node):
   argType = Fixed

   @staticproperty
   def edges():
      return [North, South, East, West]

   def args(stim, entity, config):
      return Direction.edges

class North(Node):
   delta = (-1, 0)

class South(Node):
   delta = (1, 0)

class East(Node):
   delta = (0, 1)

class West(Node):
   delta = (0, -1)


class Attack(Node):
   priority = 0
   nodeType = NodeType.SELECTION
   @staticproperty
   def n():
      return 3

   @staticproperty
   def edges():
      return [Style, Target]

   @staticproperty
   def leaf():
      return True

   def inRange(entity, stim, config, N):
      R, C = stim.shape
      R, C = R//2, C//2

      rets = set([entity])
      for r in range(R-N, R+N+1):
         for c in range(C-N, C+N+1):
            for e in stim[r, c].ents.values():
               if not config.WILDERNESS:
                  rets.add(e)
                  continue

               minWilderness = min(entity.status.wilderness.val, e.status.wilderness.val)
               selfLevel     = combat.level(entity.skills)
               targLevel     = combat.level(e.skills)
               if abs(selfLevel - targLevel) <= minWilderness:
                  rets.add(e)

      rets = list(rets)
      return rets

   def l1(pos, cent):
      r, c = pos
      rCent, cCent = cent
      return abs(r - rCent) + abs(c - cCent)

   def call(env, entity, style, targ):
      if targ is None or entity.entID == targ.entID:
         return

      #Can't attack if either party is immune
      if entity.status.immune > 0 or targ.status.immune > 0:
         return

      #Check wilderness level
      wilderness = min(entity.status.wilderness, targ.status.wilderness)
      selfLevel  = combat.level(entity.skills)
      targLevel  = combat.level(targ.skills)

      if (env.config.WILDERNESS and abs(selfLevel - targLevel) > wilderness
            and entity.isPlayer and targ.isPlayer):
         return

      #Check attack range
      rng     = style.attackRange(env.config)
      start   = np.array(entity.base.pos)
      end     = np.array(targ.base.pos)
      dif     = np.max(np.abs(start - end))

      #Can't attack same cell or out of range
      if dif == 0 or dif > rng:
         return 
      
      #Execute attack
      entity.history.attack = {}
      entity.history.attack['target'] = targ.entID
      entity.history.attack['style'] = style.__name__
      targ.attacker = entity

      dmg = combat.attack(entity, targ, style.skill)
      if style.freeze and dmg > 0:
         targ.status.freeze.update(env.config.FREEZE_TIME)

      return dmg

class Style(Node):
   argType = Fixed

   @staticproperty
   def edges():
      return [Melee, Range, Mage]

   def args(stim, entity, config):
      return Style.edges

class Target(Node):
   argType  = 'Entity'

   @classmethod
   def N(cls, config):
      #return config.WINDOW ** 2
      return config.N_AGENT_OBS

   def args(stim, entity, config):
      #Should pass max range?
      return Attack.inRange(entity, stim, config, None)

   @classmethod
   def gameObjects(cls, realm, entity, val):
      return [realm.entity(targ) for targ in entity.targets]

class Melee(Node):
   nodeType = NodeType.ACTION
   index = 0
   freeze=False

   def attackRange(config):
      return config.MELEE_RANGE

   def skill(entity):
      return entity.skills.melee

class Range(Node):
   nodeType = NodeType.ACTION
   index = 1
   freeze=False

   def attackRange(config):
      return config.RANGE_RANGE

   def skill(entity):
      return entity.skills.range

class Mage(Node):
   nodeType = NodeType.ACTION
   index = 2
   freeze=True

   def attackRange(config):
      return config.MAGE_RANGE

   def skill(entity):
      return entity.skills.mage

class Buy(Node):
   priority = -1 
   nodeType = NodeType.SELECTION

   @staticproperty
   def edges():
      return [ItemType]

   @staticproperty
   def leaf():
      return True

#<<<<<<< HEAD
#class Harvest(Node):
#   priority = 2
#   nodeType = NodeType.SELECTION
#   @staticproperty
#   def edges():
#     #return [Fish, Mine]
#      return [Mine, Woodcut]
#
#   def args(stim, entity, config):
#      return Harvest.edges
#
#class Fish(Node):
#   nodeType = NodeType.ACTION
#
#class Mine(Node):
#   nodeType = NodeType.SELECTION
#
##  def update(self, realm, entity):
#   def call(env, entity, trg):
#      if not env.map.harvest(*trg):
#         return
#      ore = entity.resources.ore
#      mining = entity.skills.mining
#
#      restore = np.floor(mining.level * 1)
#      ore.increment(restore)
#
#      scale = entity.config.XP_SCALE
#      # we don't use RESOURCE_RESTORE to affect inventory, but factor it into experience for the
#      # sake of diversity calculations
#      entity.skills.mining.exp += scale * restore * 10 
#
#   @staticproperty
#   def edges():
#      return [Direction]
#
#  #@staticproperty
#  #def leaf():
#  #   return True
#
#class Woodcut(Node):
#   nodeType = NodeType.SELECTION
#
#   def call(env, entity, trg):
#      if not env.map.harvest(*trg):
#         return
#      wood = entity.resources.wood
#      woodcutting = entity.skills.woodcutting
#
#      restore = np.floor(woodcutting.level * 1)
#      wood.increment(restore)
#
#      scale = entity.config.XP_SCALE
#      entity.skills.woodcutting.exp += scale * restore * 10
#
#   @staticproperty
#   def edges():
#      return [Direction]
#
#   @staticproperty
#   def leaf():
#      return True
#=======
   def call(env, entity, item):
      if not item:
         return
      
      return env.exchange.buy(entity, item, 0, 99)

class ItemType(Node):
   argType = Fixed
   @staticproperty
   def edges():
      return [item.Hat, item.Top, item.Bottom, item.Weapon,
              item.Scrap, item.Shaving, item.Shard,
              item.Food, item.Potion]

   def args(stim, entity, config):
      return Item.edges
#>>>>>>> 17f0ddfd1c21ba37d2a5bb44eca6fe7a18aba382

class InventoryAction(Node):
   priority = -2 
   nodeType = NodeType.SELECTION

   @staticproperty
   def edges():
      return [InventoryActionType, Item]

   @staticproperty
   def leaf():
      return True

   def call(env, entity, actionType, item):
      assert actionType in (Discard, Use, Sell)

      if item is None:
         return

      if actionType == Discard:
         return
         return entity.inventory.consumables.remove(item)
      if actionType == Sell:
         return env.exchange.sell(entity, item)

      if not entity.inventory.consumables.__contains__(type(item)):
         return

      item.use(entity)
      entity.inventory.consumables.remove(item)
      return True

class Item(Node):
   argType  = 'Item'

   @classmethod
   def N(cls, config):
      return config.N_AMMUNITION + config.N_CONSUMABLES + config.N_LOOT + 1

   def args(stim, entity, config):
      return entity.items

   @classmethod
   def gameObjects(cls, realm, entity, val):
      return entity.inventory.items[5:]

class InventoryActionType(Node):
   priority = -3 
   argType  = Fixed

   @staticproperty
   def edges():
      return [Discard, Sell, Use]

   def args(env, entity, item):
      return InventoryActionType.edges

class Discard(Node):
   nodeType = NodeType.ACTION

class Sell(Node):
   nodeType = NodeType.ACTION

class Use(Node):
   nodeType = NodeType.ACTION

class Message:
   pass

class Reproduce:
   pass

class BecomeSkynet:
   pass

Action.hook()
