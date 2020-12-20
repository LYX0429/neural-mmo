from pdb import set_trace as T
import numpy as np
from skimage.draw import line, rectangle
from scipy.stats import multivariate_normal

class PaintPattern():
   def __init__(self, tile_i, intensity, map_width):
      self.map_width = map_width
      self.intensity = intensity
      self.tile_i = tile_i

class EndpointPattern(PaintPattern):
   def generate(pattern_class, n_tiles, map_width):
      p = pattern_class(
            np.random.randint(0, map_width, 2),
            np.random.randint(0, map_width, 2),
            np.random.randint(0, n_tiles),
            np.random.random(),
            map_width,
            )

      return p

   def __init__(self, pos_0, pos_1, tile_i, intensity, map_width):
      super().__init__(tile_i, intensity, map_width)
      self.x_0, self.y_0 = pos_0
      self.x_1, self.y_1 = pos_1

   def mutate(self):
      self.x_0 = self.mutate_endpoint(self.x_0)
      self.y_0 = self.mutate_endpoint(self.y_0)
      self.x_1 = self.mutate_endpoint(self.x_1)
      self.y_1 = self.mutate_endpoint(self.y_1)

   def mutate_endpoint(self, e):
      rand = np.random.random()

      if rand < 0.3:
         e += np.random.randint(10)
         e = min(max(0, e), self.map_width-1)
      elif rand < 0.35:
         e = np.random.randint(self.map_width)

      return e

class Distribution(PaintPattern):
   def generate(pattern_class, n_tiles, map_width):
      p = pattern_class(
            mean=np.random.randint(0, map_width, 2),
            std_devs=np.random.randint(0, map_width, 2),
            tile_i=np.random.randint(0, n_tiles),
            intensity=np.random.random(),
            map_width=map_width,
            )

      return p

   def __init__(self, mean, std_devs, tile_i, intensity, map_width):
      super().__init__(tile_i, intensity, map_width)
      self.mean = mean
      self.std_devs = std_devs

   def mutate(self):
      self.mean = np.clip(0, self.map_width-1, self.mean + np.random.randint(0, 10, len(self.mean)))
      self.std_devs = np.clip(0, self.map_width-1, self.std_devs + np.random.randint(0, 10, len(self.std_devs)))


class Line(EndpointPattern):
   def __init__(self, *args):
      super(Line, self).__init__(*args)

   def paint(self, map_arr):
      y_0, y_1 = self.y_0, self.y_1
      x_0, x_1 = self.x_0, self.x_1
      tile_i, intensity = self.tile_i, self.intensity
      rr, cc = line(x_0, y_0, x_1, y_1)
      map_arr[tile_i, rr, cc] += intensity

class Rectangle(EndpointPattern):
   def __init__(self, *args):
      super(Rectangle, self).__init__(*args)

   def paint(self, map_arr):
      y_0, y_1 = self.y_0, self.y_1
      x_0, x_1 = self.x_0, self.x_1
      tile_i, intensity = self.tile_i, self.intensity
      rr, cc = rectangle((x_0, y_0), (x_1, y_1))
      map_arr[tile_i, rr, cc] += intensity

class Gaussian(Distribution):
   def __init__(self, mean, std_devs, tile_i, intensity, map_width):
      super().__init__(mean, std_devs, tile_i, intensity, map_width)

   def paint(self, map_arr):
      mean, std_devs = self.mean, self.std_devs
      tile_i, intensity = self.tile_i, self.intensity
      dist = multivariate_normal(mean, self.std_devs, allow_singular=True)
      x, y = np.indices(map_arr[tile_i].shape)
      pos = np.dstack((x, y))
      map_arr[tile_i] += dist.pdf(pos) * intensity

class Chromosome():
   def __init__(self, map_width, n_tiles, max_patterns, default_tile):
      self.map_width = map_width
      self.n_tiles = n_tiles
      self.max_patterns = max_patterns
      self.default_tile = default_tile
      self.pattern_templates = [Line, Rectangle, Gaussian]
#     self.pattern_templates = [Gaussian]
      self.weights =  [2/3,  1/3]

#  def init_endpoint_pattern(self, p):
#     p = p(
#           np.random.randint(0, self.map_width, 2),
#           np.random.randint(0, self.map_width, 2),
#           np.random.randint(0, self.n_tiles),
#           np.random.random(),
#           self.map_width,
#           )

#     return p

   def generate(self):
      self.patterns = np.random.choice(self.pattern_templates, 
            np.random.randint(self.max_patterns), self.weights).tolist()

      for i, p in enumerate(self.patterns):
        #if p in [Line, Rectangle]:
         p = p.generate(p, self.n_tiles, self.map_width)
         self.patterns[i] = p
        #else:
        #   raise Exception

      return self.paint_map()

   def mutate(self):
      for p in self.patterns:
         rand = np.random.random()

         if rand < 0.3:
            p.mutate()

      n_patterns = len(self.patterns)
      if np.random.random() < 0.3 and n_patterns > 0:
         self.patterns.pop(np.random.randint(n_patterns))
      if np.random.random() < 0.3 and n_patterns < self.max_patterns:
         p = np.random.choice(self.pattern_templates)
         p = p.generate(p, self.n_tiles, self.map_width)
         self.patterns.append(p)

      return self.paint_map()


   def paint_map(self):
      multi_hot = np.zeros((self.n_tiles, self.map_width, self.map_width))
      multi_hot[self.default_tile, :, :] = 1e-10

      for p in self.patterns:
         p.paint(multi_hot)
      flat_map = np.argmax(multi_hot, axis=0)

      return flat_map, multi_hot

