#Main world definition. Defines and manages entity handlers,
#Defines behavior of the world under all circumstances and handles
#interaction by agents. Also defines an accurate stimulus that
#encapsulates the world as seen by a particular agent

from pdb import set_trace as TT
import random
import numpy as np

from collections import defaultdict, Mapping
from typing import Dict, Callable

from forge.blade import core
from forge.blade.lib.enums import Palette
from forge.blade.entity.npc import NPC
from forge.blade.entity import Player
from forge import trinity

from forge.blade.io.stimulus import Static

def prioritized(entities: Dict, merged: Dict):
   '''Sort actions into merged according to priority'''
   for idx, actions in entities.items():
      for atn, args in actions.items():
         merged[atn.priority].append((idx, (atn, args.values())))
   return merged

class EntityGroup(Mapping):
   def __init__(self, config, realm):
      self.dataframe = realm.dataframe
      self.config    = config

      self.entities  = {}
      self.dead      = {}

   def __len__(self):
      return len(self.entities)

   def __contains__(self, e):
      return e in self.entities

   def __getitem__(self, key):
      return self.entities[key]
   
   def __iter__(self):
      yield from self.entities

   def items(self):
      return self.entities.items()

   @property
   def packet(self):
      corporeal = {**self.entities, **self.dead}
      return {k: v.packet() for k, v in corporeal.items()}

   def reset(self):
      for entID, ent in self.entities.items():
         self.dataframe.remove(Static.Entity, entID, ent.pos)

      self.entities = {}
      self.dead     = {}

   def add(iden, entity):
      assert iden not in self.entities
      self.entities[iden] = entity

   def remove(iden):
      assert iden in self.entities
      del self.entities[iden] 

   def spawn(self, entity):
      pos, entID = entity.pos, entity.entID
      self.realm.map.tiles[pos].addEnt(entity)
      self.entities[entID] = entity
 
   def cull(self):
      self.dead = {}
      for entID in list(self.entities):
         player = self.entities[entID]
         if not player.alive:
            r, c  = player.base.pos
            entID = player.entID
            self.dead[entID] = player

            self.realm.map.tiles[r, c].delEnt(entID)
            del self.entities[entID]
            self.realm.dataframe.remove(Static.Entity, entID, player.pos)

      return self.dead

   def update(self, actions):
      for entID, entity in self.entities.items():
         entity.update(self.realm, actions)

class NPCManager(EntityGroup):
   def __init__(self, config, realm):
      super().__init__(config, realm)
      self.realm = realm
      self.idx   = -1
 
   def spawn(self):
      for _ in range(self.config.NPC_SPAWN_ATTEMPTS):
         if len(self.entities) >= self.config.NMOB:
            break

         r, c = np.random.randint(0, self.config.TERRAIN_SIZE, 2).tolist()
         if self.realm.map.tiles[r, c].occupied:
            continue

         npc = NPC.spawn(self.realm, (r, c), self.idx)
         if npc:
            super().spawn(npc)
            self.idx -= 1

   def actions(self, realm):
      actions = {}
      for idx, entity in self.entities.items():
         actions[idx] = entity.decide(realm)
      return actions
       
class PlayerManager(EntityGroup):
   def __init__(self, config, realm, identify: Callable):
      super().__init__(config, realm)
      if config.MULTI_MODEL_NAMES is not None:
         self.models = config.MULTI_MODEL_NAMES
      else:
         if config.PAIRED:
            if not config.EVALUATE:
               self.models = ['pro', 'ant']
               np.random.shuffle(self.models)
            else:
               self.models = ['pro']
         elif config.MODEL is None:
            self.models = [config.GENOME]
         else:
            self.models = [config.MODEL]
      self.max_pop = config.MAX_POP
      self.identify = identify
      self.realm    = realm

      # self.palette = Palette(config.NPOP)
      # self.idx = 1
      self.palette = Palette(config.NPOP, multi_evo=config.MULTI_MODEL_NAMES is not None, paired=config.PAIRED)
      self.reset_pop_counts()

   def reset(self):
      np.random.shuffle(self.models)
      return super().reset()

   def spawn(self):
      for _ in range(self.config.PLAYER_SPAWN_ATTEMPTS):
         if len(self.entities) >= self.config.NENT:
            break

         pop = self.idx % self.config.NPOP
         name = self.models[pop]
         if self.max_pop is not None and self.pop_counts[name] >= self.max_pop:
            # print('Not spawning: reached max pop in population {}'.format(name))
            return
         if self.config.EVO_MAP:# and not self.config.EVALUATE:
            r, c = self.evo_spawn()
         else:
            r, c   = self.config.SPAWN()
         if self.realm.map.tiles[r, c].occupied:
            continue

         if self.config.EVALUATE or self.config.PAIRED:
            pop_name = name
            color = self.palette.colors[name.split(' ')[0]]  # hack: we only color-code by map-generator, so here we're
                                                         # extracting the name of the map-generator from the short-hand
                                                         # this is atrocious
            player    = Player(self.realm, (r, c), self.idx, pop, pop_name=pop_name, name=name+'_', color=color)
         else:
            # pop, name = self.identify()
            pop_name = name
            color     = self.palette.colors[pop]
            player    = Player(self.realm, (r, c), self.idx, pop, pop_name=pop_name, name=name, color=color)

         super().spawn(player)
         self.pop_counts[name] += 1
         self.idx += 1

   def reset_pop_counts(self):
      self.idx = 1
      self.pop_counts = {m: 0 for m in self.models}

   def evo_spawn(self):
      assert len(self.realm.spawn_points) != 0
      r, c = random.choice(self.realm.spawn_points)
      r, c = int(r), int(c)
      return r, c

class Realm:
   '''Top-level world object'''
   def __init__(self, config, identify: Callable):
      self.config   = config
      self.identify = identify

      #Load the world file
      self.dataframe = trinity.Dataframe(config)
      self.map       = core.Map(config, self)

      #Entity handlers
      self.players  = PlayerManager(config, self, identify)
      self.npcs     = NPCManager(config, self)

   def reset(self, idx):
      '''Reset the environment and load the specified map

      Args:
         idx: Map index to load
      '''
      # Shuffle models so that we don't always spawn the same one first
      self.map.reset(self, idx)
      self.players.reset()
      self.npcs.reset()
      self.tick = 0
 
   def packet(self):
      '''Client packet'''
      return {'environment': self.map.repr,
              'resource': self.map.packet,
              'player': self.players.packet,
              'npc': self.npcs.packet}

   @property
   def population(self):
      '''Number of player agents'''
      return len(self.players.entities)

   def entity(self, entID):
      '''Get entity by ID'''
      if entID < 0:
         return self.npcs[entID]
      else:
         return self.players[entID]

   def set_map(self, idx, map_arr):
      self.map.set_map(self, idx, map_arr)

   def step(self, actions):
      '''Run game logic for one tick
      
      Args:
         actions: Dict of agent actions
      '''

      #Prioritize actions
      npcActions = self.npcs.actions(self)
      merged     = defaultdict(list)
      prioritized(actions, merged)
      prioritized(npcActions, merged)

      #Update entities and perform actions
      self.players.update(actions)
      self.npcs.update(npcActions)

      #Execute actions
      for priority in sorted(merged):
         for entID, (atn, args) in merged[priority]:
            ent = self.entity(entID)
            atn.call(self, ent, *args)

      #Cull dead agents and spawn new ones
      dead = self.players.cull()
      self.npcs.cull()

      self.npcs.spawn()
      self.players.spawn()

      while len(self.players.entities) == 0:
         self.players.spawn()

      #Update map
      self.map.step()
      self.tick += 1

      return dead
