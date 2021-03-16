import numpy as np
from pdb import set_trace as T

from forge.blade.systems import ai
from forge.blade.lib import material

from forge.blade.systems.skill import Skills
from forge.blade.systems.inventory import Inventory
from forge.blade.entity import entity
from forge.blade.io.stimulus import Static
from forge.blade.io import action

class Player(entity.Entity):
   def __init__(self, realm, pos, iden, pop, name='', color=None):
      super().__init__(realm, pos, iden, name, color, pop)
      self.pop    = pop

      #Scripted hooks
      self.target = None
      self.food   = None
      self.water  = None
      self.initialized = False
      self.combat      = False
      self.forage      = False
      self.resource    = None
      self.downtime    = None

      #Logs
      self.buys   = 0
      self.sells  = 0

      #Submodules
      self.skills     = Skills(self)
      self.inventory  = Inventory(realm, self)
      #self.chat      = Chat(dataframe)

      #Update immune
      mmul = realm.config.IMMUNE_MUL
      madd = realm.config.IMMUNE_ADD
      mmax = realm.config.IMMUNE_MAX
      immune = min(mmul*len(realm.players) + madd, mmax)
      self.status.immune.update(immune)

      self.dataframe.init(Static.Entity, self.entID, self.pos)
#     self.exploration_grid = np.zeros(realm.shape, dtype=np.bool)
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
         return 

      self.resources.food.decrement(dmg)
      self.resources.water.decrement(dmg)
      self.skills.receiveDamage(dmg)

   def packet(self):
      data = super().packet()

      data['entID']     = self.entID
      data['annID']     = self.population

      data['base']      = self.base.packet()
      data['resource']  = self.resources.packet()
      data['skills']    = self.skills.packet()
      data['inventory'] = self.inventory.packet()

      return data
  
   def update(self, realm, actions):
      '''Post-action update. Do not include history'''
      super().update(realm, actions)

      if not self.alive:
         return
      if hasattr(realm, 'target_action_sequence') and len(actions) > 0 and self.entID in actions:
#        act = actions[self.entID][action.static.Move][action.static.Direction]
#        if act == realm.target_action_sequence[realm.tick]:
         if self.pos[1] - self.history.lastPos[1] == 1:
            self.actions_matched += 1

      self.resources.update(realm, self, actions)
      self.skills.update(realm, self, actions)
      #self.inventory.update(world, actions)
      self.explored.add(self.pos)
