from pdb import set_trace as T

class Tier:
   EASY   = 4
   NORMAL = 10
   HARD   = 25

class Diary:
   def __init__(self, config):
      self.achievements = []

      if config.game_system_enabled('Achievement'):
         self.achievements = [Exploration]

      self.achievements = [a(config) for a in self.achievements]

   @property
   def stats(self):
      return [a.stats for a in self.achievements]

   def score(self, aggregate=True):
      score = [a.score() for a in self.achievements]
      if score and aggregate:
         return sum(score)
      return score

   def update(self, realm, entity, aggregate=True, dry=False):
      scores = [a.update(realm, entity, dry) for a in self.achievements]
      if scores and aggregate:
         return sum(scores)
      return scores


class Achievement:
   def __init__(self, easy=None, normal=None, hard=None):
      self.progress = 0

      self.easy     = easy
      self.normal   = normal
      self.hard     = hard

   @property
   def stats(self):
      return self.__class__.__name__, self.progress

   def score(self, progress=None):
      if not progress:
         progress = self.progress
      if self.hard and progress >= self.hard:
         return Tier.HARD
      elif self.normal and progress >= self.normal:
         return Tier.NORMAL
      elif self.easy and progress >= self.easy:
         return Tier.EASY
      return 0

   def update(self, value, dry):
      if value <= self.progress:
         return 0

      #Progress to score conversion
      old = self.score(self.progress)
      new = self.score(value)

      if not dry:
         self.progress = value

      return new - old

class Exploration(Achievement):
   def __init__(self, config):
      super().__init__(easy   = config.EXPLORATION_EASY,
                       normal = config.EXPLORATION_NORMAL,
                       hard   = config.EXPLORATION_HARD)

   def update(self, realm, entity, dry):
      return super().update(entity.history.exploration, dry)
