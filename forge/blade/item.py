from pdb import set_trace as T
import random

from forge.blade.io.stimulus import Static
from forge.blade.lib.enums import Tier

class Item:
   INSTANCE_ID = 1
   def __init__(self, realm, level,
         capacity=0, quantity=0, tradable=True,
         offense=0, defense=0, minDmg=0, maxDmg=0,
         restore=0):
      self.instanceID   = self.INSTANCE_ID
      Item.INSTANCE_ID += 1

      self.config   = realm.config
      self.realm    = realm  

      self.index    = Static.Item.Index(realm.dataframe, self.instanceID, self.ITEM_ID)
      self.level    = Static.Item.Level(realm.dataframe, self.instanceID, level)
      self.capacity = Static.Item.Capacity(realm.dataframe, self.instanceID, capacity)
      self.quantity = Static.Item.Quantity(realm.dataframe, self.instanceID, quantity)
      self.tradable = Static.Item.Tradable(realm.dataframe, self.instanceID, tradable)
      self.offense  = Static.Item.Offense(realm.dataframe, self.instanceID, offense)
      self.defense  = Static.Item.Defense(realm.dataframe, self.instanceID, defense)
      self.minDmg   = Static.Item.MinDmg(realm.dataframe, self.instanceID, minDmg)
      self.maxDmg   = Static.Item.MaxDmg(realm.dataframe, self.instanceID, maxDmg)
      self.restore  = Static.Item.Restore(realm.dataframe, self.instanceID, restore)

      realm.dataframe.init(Static.Item, self.instanceID, None)

   @property
   def packet(self):
      return {'item':     self.__class__.__name__,
              'level':    self.level.val,
              'capacity': self.capacity.val,
              'quantity': self.quantity.val,
              'offense':  self.offense.val,
              'defense':  self.defense.val,
              'minDmg':   self.minDmg.val,
              'maxDmg':   self.maxDmg.val,
              'restore':  self.restore.val}
 
   def use(self, entity):
      return

class Stack(Item):
   def use(self, entity):
      assert self.quantity > 0

      self.quantity -= 1
      if self.quantity > 0:
         return

      entity.inventory.remove(self)

class Gold(Item):
   ITEM_ID = 1
   def __init__(self, realm, **kwargs):
      super().__init__(realm, level=0, tradable=False, **kwargs)

class Equipment(Item):
   @property
   def packet(self):
     packet = {'color': self.color.packet()}
     return {**packet, **super().packet}

   @property
   def color(self):
     if self.level == 0:
        return Tier.BLACK
     if self.level < 10:
        return Tier.WOOD
     elif self.level < 20:
        return Tier.BRONZE
     elif self.level < 40:
        return Tier.SILVER
     elif self.level < 60:
        return Tier.GOLD
     elif self.level < 80:
        return Tier.PLATINUM
     else:
        return Tier.DIAMOND

class Offensive(Equipment):
   def __init__(self, realm, level, **kwargs):
      offense = realm.config.EQUIPMENT_OFFENSE(level)
      super().__init__(realm, level, offense=offense, **kwargs)

class Defensive(Equipment):
   def __init__(self, realm, level, **kwargs):
      defense = realm.config.EQUIPMENT_DEFENSE(level)
      super().__init__(realm, level, defense=defense, **kwargs)

class Hat(Defensive):
   ITEM_ID = 2

class Top(Defensive):
   ITEM_ID = 3

class Bottom(Defensive):
   ITEM_ID = 4

class Weapon(Offensive):
   ITEM_ID = 5

class Ammunition(Stack):
   def __init__(self, realm, level, **kwargs):
      minDmg, maxDmg = realm.config.DAMAGE_AMMUNITION(level)
      super().__init__(realm, level, minDmg=minDmg, maxDmg=maxDmg, **kwargs)

   def damage(self):
      return random.randint(self.minDmg.val, self.maxDmg.val)
  
class Scrap(Ammunition):
   ITEM_ID = 6

class Shaving(Ammunition):
   ITEM_ID = 7

class Shard(Ammunition):
   ITEM_ID = 8

class Consumable(Item):
   def __init__(self, realm, level, **kwargs):
      restore = realm.config.RESTORE(level)
      super().__init__(realm, level, restore=restore, **kwargs)

class Food(Consumable):
   ITEM_ID = 9
   def use(self, entity):
      entity.resources.food.increment(self.restore.val)
      entity.resources.water.increment(self.restore.val)

class Potion(Consumable):
   ITEM_ID = 10
   def use(self, entity):
      entity.resources.health.increment(self.restore.val)
 
