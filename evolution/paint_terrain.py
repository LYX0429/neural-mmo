from pdb import set_trace as T
import numpy as np
from skimage.draw import line, rectangle, rectangle_perimeter, circle, circle_perimeter
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
            np.random.random() * 2 - 1,
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
         e += np.random.randint(-10, 10)
         e = min(max(0, e), self.map_width-1)
      elif rand < 0.35:
         e = np.random.randint(self.map_width)

      return e

class Distribution(PaintPattern):
   def generate(pattern_class, n_tiles, map_width):
      p = pattern_class(
            mean=np.random.randint(0, map_width, 2),
            std_devs=np.random.randint(1, map_width, 2),
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
      self.mean = np.clip(self.mean + np.random.randint(-10, 10, len(self.mean)), 0, self.map_width-1)
      self.std_devs = np.clip(self.std_devs + np.random.randint(-10, 10, len(self.std_devs)), 1, self.map_width-1)

class CircleType(PaintPattern):
   def generate(pattern_class, n_tiles, map_width):
      p = pattern_class(
            r=np.random.randint(0, map_width),
            c=np.random.randint(0, map_width),
            radius=np.random.randint(1, map_width/2), 
            tile_i=np.random.randint(0, n_tiles),
            intensity=np.random.random(),
            map_width=map_width,
            )

      return p

   def __init__(self, r, c, radius, tile_i, intensity, map_width):
      super().__init__(tile_i, intensity, map_width)
      self.r = r
      self.c = c
      self.radius = radius

   def mutate(self):
      self.r = np.clip(self.r + np.random.randint(-10, 10), 0, self.map_width-1)
      self.c = np.clip(self.c + np.random.randint(-10, 10), 0, self.map_width-1)
      self.radius = np.clip(self.radius + np.random.randint(-10, 10), 1, self.map_width-1)

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

class RectanglePerimeter(EndpointPattern):
   def __init__(self, *args):
      super(RectanglePerimeter, self).__init__(*args)

   def paint(self, map_arr):
      y_0, y_1 = self.y_0, self.y_1
      x_0, x_1 = self.x_0, self.x_1
      tile_i, intensity = self.tile_i, self.intensity
      rr, cc = rectangle_perimeter((x_0, y_0), (x_1, y_1), shape=(self.map_width, self.map_width))
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

class Circle(CircleType):
   def __init__(self, r, c, radius, tile_i, intensity, map_width):
      super(Circle, self).__init__(r, c, radius, tile_i, intensity, map_width)

   def paint(self, map_arr):
#     rr, cc = disk((self.r, self.c), self.radius, shape=(self.map_width, self.map_width))
      rr, cc = circle(self.r, self.c, self.radius, shape=(self.map_width, self.map_width))
      map_arr[self.tile_i, rr, cc] += self.intensity


class CirclePerimeter(CircleType):
   def __init__(self, r, c, radius, tile_i, intensity, map_width):
      super(CirclePerimeter, self).__init__(r, c, radius, tile_i, intensity, map_width)

   def paint(self, map_arr):
      rr, cc = circle_perimeter(self.r, self.c, self.radius, shape=(self.map_width, self.map_width))
      map_arr[self.tile_i, rr, cc] += self.intensity

