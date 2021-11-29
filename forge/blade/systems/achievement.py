from pdb import set_trace as T

class Diary:
   def __init__(self, config):
      self.achievements = []
      self.achievements = [Get_food, Get_water, Surviving_1, Surviving_2, Surviving_3, Exploration_1,
                           Exploration_2, Exploration_3, Skill_up_woodcutting_1, Skill_up_woodcutting_2,
                           Skill_up_woodcutting_3, Skill_up_mining_1, Skill_up_mining_2, Skill_up_mining_3]
      self.achievements = [a() for a in self.achievements]

      index = {'Get_food': 0, 'Get_water': 1, 'Surviving_1': 2, 'Surviving_2': 3, 'Surviving_3': 4,
               'Exploration_1': 5, 'Exploration_2': 6, 'Exploration_3': 7, 'Skill_up_woodcutting_1': 8,
               'Skill_up_woodcutting_2': 9, 'Skill_up_woodcutting_3': 10, 'Skill_up_mining_1': 11,
               'Skill_up_mining_2': 12, 'Skill_up_mining_3': 13, }

      self.achievements[index['Get_food']].activate = True
      self.achievements[index['Get_food']].prerequisite = 0
      self.achievements[index['Get_food']].value = 0
      self.achievements[index['Get_food']].next = [index['Surviving_1'], index['Exploration_1']]

      self.achievements[index['Get_water']].activate = True
      self.achievements[index['Get_water']].prerequisite = 0
      self.achievements[index['Get_water']].value = 0
      self.achievements[index['Get_water']].next = [index['Surviving_1'], index['Exploration_1']]

      self.achievements[index['Surviving_1']].prerequisite = 2
      self.achievements[index['Surviving_1']].value = 1
      self.achievements[index['Surviving_1']].next = [index['Surviving_2'], index['Exploration_1'],
                                                      index['Skill_up_woodcutting_1'], index['Skill_up_mining_1']]

      self.achievements[index['Surviving_2']].prerequisite = 1
      self.achievements[index['Surviving_2']].value = 1
      self.achievements[index['Surviving_2']].next = [index['Surviving_3'], index['Exploration_2']]

      self.achievements[index['Surviving_3']].prerequisite = 1
      self.achievements[index['Surviving_3']].value = 1
      self.achievements[index['Surviving_3']].next = [index['Exploration_3']]

      self.achievements[index['Exploration_1']].prerequisite = 3
      self.achievements[index['Exploration_1']].value = 5
      self.achievements[index['Exploration_1']].next = [index['Exploration_2']]

      self.achievements[index['Exploration_2']].prerequisite = 2
      self.achievements[index['Exploration_2']].value = 10
      self.achievements[index['Exploration_2']].next = [index['Exploration_3']]

      self.achievements[index['Exploration_3']].prerequisite = 2
      self.achievements[index['Exploration_3']].value = 15
      self.achievements[index['Exploration_3']].next = []

      self.achievements[index['Skill_up_woodcutting_1']].prerequisite = 1
      self.achievements[index['Skill_up_woodcutting_1']].value = 5
      self.achievements[index['Skill_up_woodcutting_1']].next = [index['Skill_up_woodcutting_2']]

      self.achievements[index['Skill_up_woodcutting_2']].prerequisite = 1
      self.achievements[index['Skill_up_woodcutting_2']].value = 10
      self.achievements[index['Skill_up_woodcutting_2']].next = [index['Skill_up_woodcutting_3']]

      self.achievements[index['Skill_up_woodcutting_3']].prerequisite = 1
      self.achievements[index['Skill_up_woodcutting_3']].value = 15
      self.achievements[index['Skill_up_woodcutting_3']].next = []

      self.achievements[index['Skill_up_mining_1']].prerequisite = 1
      self.achievements[index['Skill_up_mining_1']].value = 5
      self.achievements[index['Skill_up_mining_1']].next = [index['Skill_up_mining_2']]

      self.achievements[index['Skill_up_mining_2']].prerequisite = 1
      self.achievements[index['Skill_up_mining_2']].value = 10
      self.achievements[index['Skill_up_mining_2']].next = [index['Skill_up_mining_3']]

      self.achievements[index['Skill_up_mining_3']].prerequisite = 1
      self.achievements[index['Skill_up_mining_3']].value = 15
      self.achievements[index['Skill_up_mining_3']].next = []

   @property
   def stats(self):
      return [a.stats for a in self.achievements]

   def score(self, aggregate=True):
      score = [a.score() for a in self.achievements]
      if score and aggregate:
         return sum(score)
      return score

   def update(self, realm, entity, aggregate=True, dry=False):
      scores = []
      for a in self.achievements:
         score, unlock = a.update(realm, entity, dry)
         scores.append(score)
         if unlock:
            for i in a.next:
               self.achievements[i].prerequisite -= 1
               if self.achievements[i].prerequisite == 0:
                  self.achievements[i].activate = True

      if scores and aggregate:
         return sum(scores)
      return scores


class Achievement:
   def __init__(self):
      self.activate = False
      self.complete = False
      self.next = []
      self.prerequisite = 1
      self.value = 0

   @property
   def stats(self):
      return self.__class__.__name__, self.complete

   def score(self, complete=None):
      if not complete:
         complete = self.complete
      if complete:
         return self.value
      return 0

   def update(self, value, dry):
      if not self.activate:
         return 0, False

      #Progress to score conversion
      old = self.score(self.complete)
      new = self.score(value)

      unlock = False
      if not dry:
         if not self.complete and value:
            unlock = True
         self.complete = value
      return new - old, unlock

class Exploration_1(Achievement):
   def __init__(self):
      super().__init__()

   def update(self, realm, entity, dry):
      if entity.history.exploration >= 10:
         return super().update(True, dry)
      else:
         return super().update(False, dry)

class Exploration_2(Achievement):
   def __init__(self):
      super().__init__()

   def update(self, realm, entity, dry):
      if entity.history.exploration >= 20:
         return super().update(True, dry)
      else:
         return super().update(False, dry)

class Exploration_3(Achievement):
   def __init__(self):
      super().__init__()

   def update(self, realm, entity, dry):
      if entity.history.exploration >= 30:
         return super().update(True, dry)
      else:
         return super().update(False, dry)


class Get_food(Achievement):
   def __init__(self):
      super().__init__()
      self.food = 99999

   def update(self, realm, entity, dry):
      old = self.food
      new = entity.resources.food.val
      if dry:
         self.food = new
      if new > old:
         return super().update(True, dry)
      else:
         return super().update(False, dry)

class Get_water(Achievement):
   def __init__(self):
      super().__init__()
      self.water = 99999

   def update(self, realm, entity, dry):
      old = self.water
      new = entity.resources.water.val
      if dry:
         self.water = new
      if new > old:
         return super().update(True, dry)
      else:
         return super().update(False, dry)

class Surviving_1(Achievement):
   def __init__(self):
      super().__init__()

   def update(self, realm, entity, dry):
      if entity.history.timeAlive.val >= 5:
         return super().update(True, dry)
      else:
         return super().update(False, dry)

class Surviving_2(Achievement):
   def __init__(self):
      super().__init__()

   def update(self, realm, entity, dry):
      if entity.history.timeAlive.val >= 15:
         return super().update(True, dry)
      else:
         return super().update(False, dry)

class Surviving_3(Achievement):
   def __init__(self):
      super().__init__()

   def update(self, realm, entity, dry):
      if entity.history.timeAlive.val >= 30:
         return super().update(True, dry)
      else:
         return super().update(False, dry)

class Skill_up_hunting_1(Achievement):
   def __init__(self):
      super().__init__()

   def update(self, realm, entity, dry):
      if entity.skills.hunting.level >= 11:
         return super().update(True, dry)
      else:
         return super().update(False, dry)

class Skill_up_hunting_2(Achievement):
   def __init__(self):
      super().__init__()

   def update(self, realm, entity, dry):
      if entity.skills.hunting.level >= 13:
         return super().update(True, dry)
      else:
         return super().update(False, dry)

class Skill_up_hunting_3(Achievement):
   def __init__(self):
      super().__init__()

   def update(self, realm, entity, dry):
      if entity.skills.hunting.level >= 15:
         return super().update(True, dry)
      else:
         return super().update(False, dry)

class Skill_up_fishing_1(Achievement):
   def __init__(self):
      super().__init__()

   def update(self, realm, entity, dry):
      if entity.skills.fishing.level >= 11:
         return super().update(True, dry)
      else:
         return super().update(False, dry)

class Skill_up_woodcutting_1(Achievement):
   def __init__(self):
      super().__init__()

   def update(self, realm, entity, dry):
      if entity.skills.woodcutting.level >= 11:
         return super().update(True, dry)
      else:
         return super().update(False, dry)

class Skill_up_woodcutting_2(Achievement):
   def __init__(self):
      super().__init__()

   def update(self, realm, entity, dry):
      if entity.skills.woodcutting.level >= 13:
         return super().update(True, dry)
      else:
         return super().update(False, dry)

class Skill_up_woodcutting_3(Achievement):
   def __init__(self):
      super().__init__()

   def update(self, realm, entity, dry):
      if entity.skills.woodcutting.level >= 15:
         return super().update(True, dry)
      else:
         return super().update(False, dry)

class Skill_up_mining_1(Achievement):
   def __init__(self):
      super().__init__()

   def update(self, realm, entity, dry):
      if entity.skills.mining.level >= 11:
         return super().update(True, dry)
      else:
         return super().update(False, dry)

class Skill_up_mining_2(Achievement):
   def __init__(self):
      super().__init__()

   def update(self, realm, entity, dry):
      if entity.skills.mining.level >= 13:
         return super().update(True, dry)
      else:
         return super().update(False, dry)

class Skill_up_mining_3(Achievement):
   def __init__(self):
      super().__init__()

   def update(self, realm, entity, dry):
      if entity.skills.mining.level >= 15:
         return super().update(True, dry)
      else:
         return super().update(False, dry)
