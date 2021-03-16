from pdb import set_trace as T
import numpy as np

from forge.blade.systems.ai import behavior, move, attack, utils
from forge.blade.systems import skill
from forge.blade.lib import material
from forge.blade.io.action import static as Action
from forge.blade import item as Item
from forge.blade import systems

def passive(realm, entity):
   behavior.update(entity)
   actions = {}

   behavior.meander(realm, actions, entity)

   return actions

def neutral(realm, entity):
   behavior.update(entity)
   actions = {}

   if not entity.attacker:
      behavior.meander(realm, actions, entity)
   else:
      entity.target = entity.attacker
      behavior.hunt(realm, actions, entity)

   return actions

def hostile(realm, entity):
   behavior.update(entity)
   actions = {}

   # This is probably slow
   if not entity.target:
      entity.target = utils.closestTarget(entity, realm.map.tiles,
                                          rng=entity.vision)

   if not entity.target:
      behavior.meander(realm, actions, entity)
   else:
      behavior.hunt(realm, actions, entity)

   return actions

def forage(realm, entity, explore=True, forage=behavior.forageDijkstra):
   return baseline(realm, entity, explore, forage, combat=False)

def combat(realm, entity, explore=True, forage=behavior.forageDijkstra):
   return baseline(realm, entity, explore, forage, combat=True)


def baseline(realm, entity, explore, forage, combat):
   harvest = behavior.harvestDijkstra
   config  = realm.config

   #Initialize
   if not entity.initialized:
      entity.initialized = True

      rng = np.random.rand()
      if rng < 0.12:
         entity.forage   = skill.Fishing
         entity.resource = material.Fish
      elif rng < 0.24:
         entity.forage   = skill.Hunting
         entity.resource = material.Herb
      elif rng < 0.36:
         entity.forage   = skill.Prospecting
         entity.resource = material.Ore
      elif rng < 0.48:
         entity.forage   = skill.Carving
         entity.resource = material.Tree
      elif rng < 0.60:
         entity.forage   = skill.Alchemy
         entity.resource = material.Crystal

      rng = np.random.rand()
      if rng < 0.33:
         entity.skills.style = Action.Melee
      elif rng < 0.66:
         entity.skills.style = Action.Range
      else:
         entity.skills.style = Action.Mage
         
   behavior.update(entity)
   actions = {}

   #Baseline only considers nearest entity
   entity.target = utils.closestTarget(entity,
         realm.map.tiles, rng=config.STIM)

   #Define agent behavior during downtime
   if explore:
      downtime = behavior.explore
   else:
      downtime = forage

   #Consume items
   potion = entity.inventory.consumables.__contains__(Item.Potion)
   if potion and entity.resources.health <= entity.resources.health.max/2:
     potion = potion[0]
     potion.use(entity)
     entity.inventory.consumables.remove(potion)

   food = entity.inventory.consumables.__contains__(Item.Food)
   if food and (entity.resources.food == 0 or entity.resources.water == 0):
      food = food[0]
      food.use(entity)
      entity.inventory.consumables.remove(food)
   
   #Exchange Buy/Sell
   for item in entity.inventory.loot:
      realm.exchange.sell(entity, item)

   for item in {Item.Food, Item.Potion}:
      if item not in entity.inventory.consumables:
         realm.exchange.buy(entity, item, 0, 99)

   ammunition = entity.inventory.ammunition
   if entity.forage:
      for item in ammunition.items:
         realm.exchange.sell(entity, item)
   else:
      if entity.skills.style == Action.Melee and Item.Scrap not in ammunition:
         realm.exchange.buy(entity, Item.Scrap, 0, 99)
      elif entity.skills.style == Action.Range and Item.Shaving not in ammunition:
         realm.exchange.buy(entity, Item.Shaving, 0, 99)
      elif entity.skills.style == Action.Mage and Item.Shard not in ammunition:
         realm.exchange.buy(entity, Item.Shard, 0, 99)

      items       = entity.inventory.equipment
      equipTypes  = [Item.Hat, Item.Top, Item.Bottom, Item.Weapon]
      equipment   = [items.hat, items.top, items.bottom, items.weapon]
      equipLevels = [e.level.val if e is not None else 0 for e in equipment]
      idx = np.random.choice(np.flatnonzero(np.array(equipLevels) == np.max(equipLevels)))
      realm.exchange.buy(entity, equipTypes[idx], equipLevels[idx], 99)

   #Forage if low on resources
   min_level = 7
   if (entity.resources.food <= min_level
         or entity.resources.water <= min_level):
      forage(realm, actions, entity)
   elif entity.attacker and combat:
      entity.target = entity.attacker
      behavior.evade(realm, actions, entity)
      behavior.attack(realm, actions, entity)
   elif not entity.forage and entity.target and combat:
      downtime(realm, actions, entity)
      entLvl  = systems.combat.level(entity.skills)
      targLvl = systems.combat.level(entity.target.skills)
      if targLvl <=  entLvl <= 5 or entLvl >= targLvl+3:
         behavior.hunt(realm, actions, entity)
      else:
         downtime(realm, actions, entity)
   elif entity.forage:
      success = harvest(realm, actions, entity)
      if not success:
         downtime(realm, actions, entity)
   else:
      downtime(realm, actions, entity)

   return actions

