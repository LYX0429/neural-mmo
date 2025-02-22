import numpy as np
from pdb import set_trace as T

from forge.blade.systems import ai, equipment
from forge.blade.lib import material

from forge.blade.systems.skill import Skills
from forge.blade.systems.inventory import Inventory
from forge.blade.systems.achievement import Diary
from forge.blade.entity import entity
from forge.blade.io.stimulus import Static
from forge.blade.io import action

class Player(entity.Entity):
   def __init__(self, realm, pos, iden, pop, pop_name='', name='', color=None):
      super().__init__(realm, pos, iden, name, color, pop)
      self.pop    = pop
      self.pop_name = pop_name

      #Scripted hooks
      self.target = None
      self.food   = None
      self.water  = None
      self.vision = 7

      #Submodules
      self.skills     = Skills(self)
      self.achievements = Diary(realm.config)
      #self.inventory = Inventory(dataframe)
      #self.chat      = Chat(dataframe)

      #Update immune
      mmul = realm.config.IMMUNE_MUL
      madd = realm.config.IMMUNE_ADD
      mmax = realm.config.IMMUNE_MAX
      immune = min(mmul*len(realm.players) + madd, mmax)
      self.status.immune.update(immune)

      self.dataframe.init(Static.Entity, self.entID, self.pos)
      self.explored = set()
      self.actions_matched = 0

   @property
   def serial(self):
      return self.population, self.entID

   @property
   def isPlayer(self) -> bool:
      return True

   @property
   def population(self):
      return self.pop

   def applyDamage(self, dmg, style):
      self.resources.food.increment(dmg)
      self.resources.water.increment(dmg)
      self.skills.applyDamage(dmg, style)
      
   def receiveDamage(self, source, dmg):
      if not super().receiveDamage(source, dmg):
         if source:
            source.history.playerKills += 1
         return

      self.resources.food.decrement(dmg)
      self.resources.water.decrement(dmg)
      self.skills.receiveDamage(dmg)

   def receiveLoot(self, loadout):
      if loadout.chestplate.level > self.loadout.chestplate.level:
         self.loadout.chestplate = loadout.chestplate
      if loadout.platelegs.level > self.loadout.platelegs.level:
         self.loadout.platelegs = loadout.platelegs

   def packet(self):
      data = super().packet()

      data['entID']    = self.entID
      data['annID']    = self.population

      data['base']     = self.base.packet()
      data['resource'] = self.resources.packet()
      data['skills']   = self.skills.packet()

      return data
  
   def update(self, realm, actions):
      '''Post-action update. Do not include history'''
#     print('player {} position {} health {}'.format(self.entID, self.pos, self.resources.health.val))
      super().update(realm, actions)

      if not self.alive:
         return
      if hasattr(realm, 'target_action_sequence') and len(actions) > 0 and self.entID in actions:
         if self.pos[1] - self.history.lastPos[1] == 1:
            self.actions_matched += 1

      self.resources.update(realm, self, actions)
      self.skills.update(realm, self, actions)
      self.explored.add(self.pos)
      self.achievements.update(realm, self)

#   def act(self, world, atnArgs):
#      #Right now we only support one arg. So *args for future update
#      atn, args = atnArgs
#      args = args.values()
#      atn.call(world, self, *args)
#
#   @property
#   def isPlayer(self) -> bool:
#      return True
