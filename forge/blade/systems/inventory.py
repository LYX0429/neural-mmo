from pdb import set_trace as T
import numpy as np

import inspect

from forge.blade import Item
from forge.blade.systems import skill as Skill

class Pouch:
   def __init__(self, capacity):
      self.capacity = capacity
      self.items    = []

   @property
   def space(self):
      return self.capacity - len(self.items)

   @property
   def packet(self):
      return [item.packet for item in self.items]

   def add(self, item):
      space = self.space
      err = '{} out of space for {}'
      assert space, err.format(self, item) 
      self.items.append(item)

   def remove(self, item):
      items = self.__contains__(item)
      err = '{} does not contain {}'
      assert items, err.format(self, item) 
      self.items.remove(items[0])

   def use(self, item):
      items = self.__contains__(item)

      if not items:
         return False

      item = items[0]
      if not item.use():
         return False

      return item

   def __iter__(self):
      for item in self.items:
         yield item

   def __contains__(self, item):
      items = []
      for itm in self.items:
         if inspect.isclass(item) and type(itm) == item:
            items.append(itm)
         elif not inspect.isclass(item) and isinstance(item, type(itm)):
            items.append(itm)
      return items

class Loadout:
   def __init__(self, realm, hat=0, top=0, bottom=0, weapon=0):
      self.hat = self.top = self.bottom = self.weapon = None
      if hat != 0:
         self.hat    = Item.Hat(realm, hat)
      if top != 0:
         self.top    = Item.Top(realm, top)
      if bottom != 0:
         self.bottom = Item.Bottom(realm, bottom)
      if weapon != 0:
         self.weapon = Item.Weapon(realm, weapon)

   def __contains__(self, item):
      return (item == self.hat or
              item == self.top or
              item == self.bottom or
              item == self.weapon)

   @property
   def packet(self):
      packet = {}

#     packet['hat']    = self.hat.packet
#     packet['top']    = self.top.packet
#     packet['bottom'] = self.bottom.packet
#     packet['weapon'] = self.weapon.packet

      return packet

   @property
   def items(self):
      items = [self.hat, self.top, self.bottom, self.weapon]
      return [e for e in items if e is not None]

   @property
   def levels(self):
      return [e.level.val for e in self.items]

   @property
   def defense(self):
      if not (items := self.items):
         return 0
      return round(np.mean([e.defense.val for e in items]))

   @property
   def offense(self):
      if not (items := self.items):
         return 0
      return round(np.mean([e.offense.val for e in items]))

   @property
   def level(self):
      if not (levels := self.levels):
         return 0
      return np.mean(levels)
         
   def remove(self, item):
      if item == self.hat:
         self.hat = None
      elif item == self.top:
         self.top = None
      elif item == self.bottom:
         self.bottom = None
      elif item == self.weapon:
         self.weapon = None
      else:
         system.exit(0, "{} not in inv".format(item))

class Inventory:
   def __init__(self, realm, entity):
      config           = realm.config
      self.realm       = realm
      self.entity      = entity
      self.config      = config

      self.gold        = Item.Gold(realm)
      self.equipment   = Loadout(realm)
      self.ammunition  = Pouch(config.N_AMMUNITION)
      self.consumables = Pouch(config.N_CONSUMABLES)
      self.loot        = Pouch(config.N_LOOT)

      self.pouches = [self.equipment, self.ammunition,
                      self.consumables, self.loot]

   def packet(self):
      data                = {}

      data['gold']        = self.gold.packet
      data['ammunition']  = self.ammunition.packet
      data['consumables'] = self.consumables.packet
      data['loot']        = self.loot.packet

      return data

   @property
   def items(self):
      return (self.equipment.items + [self.gold] + self.ammunition.items
            + self.consumables.items + self.loot.items)

   @property
   def dataframeKeys(self):
      return [e.instanceID for e in self.items[5:]]

   def __contains__(self, item):
      for pouch in self.pouches:
         if ret := pouch.__contains__(item):
            return ret
      return False

   def remove(self, item):
      for pouch in self.pouches:
         if item in pouch:
            pouch.remove(item)
            return
      
      assert False, "{} not in inv".format(item)

   def receivePurchase(self, item):
      if isinstance(item, Item.Hat):
         if self.equipment.hat:
            self.realm.exchange.sell(self.entity, self.equipment.hat)
         self.equipment.hat = item
      elif isinstance(item, Item.Top):
         if self.equipment.top:
            self.realm.exchange.sell(self.entity, self.equipment.top)
         self.equipment.top = item
      elif isinstance(item, Item.Bottom):
         if self.equipment.bottom:
            self.realm.exchange.sell(self.entity, self.equipment.bottom)
         self.equipment.bottom = item
      elif isinstance(item, Item.Weapon):
         if self.equipment.weapon:
            self.realm.exchange.sell(self.entity, self.equipment.weapon)
         self.equipment.weapon = item
      elif isinstance(item, Item.Ammunition) and self.ammunition.space:
         self.ammunition.add(item)
      elif isinstance(item, Item.Consumable) and self.consumables.space:
         self.consumables.add(item)
 
   def receiveLoot(self, items):
      if type(items) != list:
         items = [items]

      for item in items:
         #msg = 'Received Drop: Level {} {}'
         #print(msg.format(item.level, item.__class__.__name__))
         if isinstance(item, Item.Gold):
            self.gold.quantity += item.quantity.val
         elif self.loot.space:
            self.loot.add(item)
