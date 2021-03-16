from pdb import set_trace as T
import numpy as np

from forge.blade.lib import utils, material
from forge.blade.lib.utils import staticproperty
from forge.blade.io.node import Node, NodeType
from forge.blade.systems import combat
from forge.blade.io.stimulus import Static
import forge

class Fixed:
   pass

#ActionRoot
class Action(Node):
   nodeType = NodeType.SELECTION

   @staticproperty
   def edges():
      #return [Move, Attack, Exchange, Skill]
      return [Move, Attack]
      #return [Move]

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
#<<<<<<< HEAD
#      if len(env.map.tiles[trg_pos].ents) != 0:
#         return
#      if tile.impassible:
#=======
      tile = env.map.tiles[rNew, cNew]
      if tile.occupied and not tile.lava:
#>>>>>>> 1473e2bf0dd54f0ab2dbf0d05f6dbb144bdd1989
         return
      if entity.status.freeze > 0:
         return
      #FIXME: hack. Check if tile is degenerated and thus passible.
      hack_passible = tile.capacity == 0
      if dest_idx == material.Tree.index and not hack_passible:
         return Woodcut.call(env, entity, trg_pos)
      if dest_idx == material.Orerock.index:
         if not hack_passible:
            return Mine.call(env, entity, trg_pos)
         else:
            pass

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
      #Can't attack if either party is immune
      if entity.status.immune > 0 or targ.status.immune > 0:
         return

      #Check if self targeted
      if entity.entID == targ.entID:
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
   argType = None
   #argType = Player 

   @classmethod
   def N(cls, config):
      #return config.WINDOW ** 2
      return config.N_AGENT_OBS

   def args(stim, entity, config):
      #Should pass max range?
      return Attack.inRange(entity, stim, config, None)

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

class Reproduce:
   pass

class Skill(Node):
   nodeType = NodeType.SELECTION
   @staticproperty
   def edges():
     #return [Harvest, Process]
      return [Harvest]

   def args(stim, entity, config):
      return Skill.edges

class Harvest(Node):
   priority = 2
   nodeType = NodeType.SELECTION
   @staticproperty
   def edges():
     #return [Fish, Mine]
      return [Mine, Woodcut]

   def args(stim, entity, config):
      return Harvest.edges

class Fish(Node):
   nodeType = NodeType.ACTION

class Mine(Node):
   nodeType = NodeType.SELECTION

#  def update(self, realm, entity):
   def call(env, entity, trg):
      if not env.map.harvest(*trg):
         return
      ore = entity.resources.ore
      mining = entity.skills.mining

      restore = np.floor(mining.level * 1)
      ore.increment(restore)

      scale = entity.config.XP_SCALE
      # we don't use RESOURCE_RESTORE to affect inventory, but factor it into experience for the
      # sake of diversity calculations
      entity.skills.mining.exp += scale * restore * 10 

   @staticproperty
   def edges():
      return [Direction]

  #@staticproperty
  #def leaf():
  #   return True

class Woodcut(Node):
   nodeType = NodeType.SELECTION

   def call(env, entity, trg):
      if not env.map.harvest(*trg):
         return
      wood = entity.resources.wood
      woodcutting = entity.skills.woodcutting

      restore = np.floor(woodcutting.level * 1)
      wood.increment(restore)

      scale = entity.config.XP_SCALE
      entity.skills.woodcutting.exp += scale * restore * 10

   @staticproperty
   def edges():
      return [Direction]

   @staticproperty
   def leaf():
      return True

class Process(Node):
   nodeType = NodeType.SELECTION
   @staticproperty
   def edges():
      return [Cook, Smith]

   def args(stim, entity, config):
      return Process.edges

class Cook(Node):
   nodeType = NodeType.ACTION

class Smith(Node):
   nodeType = NodeType.ACTION

class Exchange(Node):
   nodeType = NodeType.SELECTION
   @staticproperty
   def edges():
      return [Buy, Sell, CancelOffer]

   def args(stim, entity, config):
      return Exchange.edges

class Buy(Node):
   nodeType = NodeType.ACTION

class Sell(Node):
   nodeType = NodeType.ACTION

class CancelOffer(Node):
   nodeType = NodeType.ACTION

class Message:
   pass

class BecomeSkynet:
   pass

Action.hook()
