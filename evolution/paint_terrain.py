from pdb import set_trace as T
import numpy as np
from skimage.draw import line, rectangle, rectangle_perimeter, circle, circle_perimeter
from scipy.stats import multivariate_normal


def gen_from_iterable(self, it):
   pattern_type, tile_type, intensity = it[0], it[1], it[2]

   pattern_class = PRIMITIVE_TYPES[pattern_type]
   return pattern_class(it[3], it[4], it[5], it[6], tile_type, intensity, self.map_width)

class PaintPattern():
   def __init__(self, tile_i, intensity, n_tiles, map_width):
      self.map_width = map_width
      self.n_tiles = n_tiles
      self.intensity = intensity
      self.tile_type = tile_i

   def mutate(self):
      if np.random.random() < 0.3:
         self.intensity += np.random.normal(0, 0.5)
      if np.random.random() < 0.3:
         self.tile_type = np.random.randint(0, self.n_tiles)


   def get_iterable(self):
      return np.array([self.pattern_type, self.tile_type, self.intensity])



class EndpointPattern(PaintPattern):
   def generate(pattern_class, tile_i, intensity, n_tiles, map_width):
      p = pattern_class(
            pos_0=np.random.randint(0, map_width, 2),
            pos_1=np.random.randint(0, map_width, 2),
            tile_i=tile_i,
            intensity=intensity,
            n_tiles = n_tiles,
            map_width=map_width,
            )

      return p

   def __init__(self, pos_0, pos_1, tile_i, intensity, n_tiles, map_width):
      super().__init__(tile_i, intensity, n_tiles, map_width)
      self.x_0, self.y_0 = pos_0
      self.x_1, self.y_1 = pos_1

   def mutate(self):
      super().mutate()
      pt = np.random.choice(['x_0', 'x_1', 'y_1', 'y_1'])
      setattr(self, pt, self.mutate_endpoint(getattr(self, pt)))

   def mutate_endpoint(self, e):
      rand = np.random.random()

#     if rand < 0.3:
      e += int(np.random.normal(0, 3))
      e = min(max(0, e), self.map_width-1)
#     elif rand < 0.35:
#        e = np.random.randint(self.map_width)

      return e

   def get_iterable(self):
      it = super().get_iterable
      it = np.hstack((it, np.array([self.x_0, self.x_1, self.y_0, self.y_1])))

      return it

class Distribution(PaintPattern):
   def generate(pattern_class, tile_i, intensity, n_tiles, map_width):
      p = pattern_class(
            mean=np.random.randint(0, map_width, 2),
            std_devs=np.random.randint(1, map_width, 2),
            tile_i=tile_i,
            intensity=intensity,
            n_tiles = n_tiles,
            map_width=map_width,
            )

      return p

   def __init__(self, mean, std_devs, tile_i, intensity, n_tiles, map_width):
      super().__init__(tile_i, intensity, map_width, n_tiles)
      self.mean = mean
      self.std_devs = std_devs

   def mutate(self):
      super().mutate()
      self.mean = np.clip(self.mean + np.random.normal(0, 5, len(self.mean)), 0, self.map_width-1)
      self.std_devs = np.clip(self.std_devs + np.random.normal(0, 5, len(self.std_devs)), 1, self.map_width-1)

   def get_iterable(self):
      it = super().get_iterable
      it = np.hstack((it, np.array([self.mean[0], self.mean[1], self.std_devs[0], self.std_devs[1]])))

      return it

   def from_iterable(self, it):
      self.mean[0], self.mean[1], self.std_devs[0], self.std_devs[1] = it

class CircleType(PaintPattern):
   def generate(pattern_class, tile_i, intensity, n_tiles, map_width):
      p = pattern_class(
            r=np.random.randint(0, map_width),
            c=np.random.randint(0, map_width),
            radius=np.random.randint(1, map_width/2),
#           extr=None,  # extra bit
            tile_i=tile_i,
            intensity=intensity,
            n_tiles = n_tiles,
            map_width=map_width,
            )

      return p

   def __init__(self, r, c, radius,
                #extr,
                tile_i, intensity, n_tiles, map_width):
      super().__init__(tile_i, intensity, n_tiles, map_width)
      self.r = r
      self.c = c
      self.radius = radius

   def mutate(self):
      super().mutate()
      self.r = np.clip(self.r + int(np.random.normal(0, 5)), 0, self.map_width-1)
      self.c = np.clip(self.c + int(np.random.normal(0, 5)), 0, self.map_width-1)
      self.radius = np.clip(self.radius + int(np.random.normal(0, 5)), 1, self.map_width-1)

class Line(EndpointPattern):
   pattern_type = 0
   def __init__(self, *args, **kwargs):
      super(Line, self).__init__(*args, **kwargs)

   def paint(self, map_arr):
      y_0, y_1 = self.y_0, self.y_1
      x_0, x_1 = self.x_0, self.x_1
      tile_i, intensity = self.tile_type, self.intensity
      rr, cc = line(x_0, y_0, x_1, y_1)
      map_arr[tile_i, rr, cc] += intensity

class RectanglePerimeter(EndpointPattern):
   pattern_type = 1
   def __init__(self, *args, **kwargs):
      super(RectanglePerimeter, self).__init__(*args, **kwargs)

   def paint(self, map_arr):
      y_0, y_1 = self.y_0, self.y_1
      x_0, x_1 = self.x_0, self.x_1
      tile_i, intensity = self.tile_type, self.intensity
      rr, cc = rectangle_perimeter((x_0, y_0), (x_1, y_1), shape=(self.map_width, self.map_width))
      map_arr[tile_i, rr, cc] += intensity

class Rectangle(EndpointPattern):
   pattern_type = 2
   def __init__(self, *args, **kwargs):
      super(Rectangle, self).__init__(*args, **kwargs)

   def paint(self, map_arr):
      y_0, y_1 = self.y_0, self.y_1
      x_0, x_1 = self.x_0, self.x_1
      tile_i, intensity = self.tile_type, self.intensity
      rr, cc = rectangle((x_0, y_0), (x_1, y_1))
      map_arr[tile_i, rr, cc] += intensity

class Circle(CircleType):
   pattern_type = 3
   def __init__(self, r, c, radius, tile_i, intensity, n_tiles, map_width):
      super(Circle, self).__init__(r, c, radius, tile_i, intensity, n_tiles, map_width)

   def paint(self, map_arr):
      #     rr, cc = disk((self.r, self.c), self.radius, shape=(self.map_width, self.map_width))
      rr, cc = circle(self.r, self.c, self.radius, shape=(self.map_width, self.map_width))
      map_arr[self.tile_type, rr, cc] += self.intensity


class CirclePerimeter(CircleType):
   pattern_type = 5
   def __init__(self, r, c, radius, tile_i, intensity, map_width, n_tiles):
      super(CirclePerimeter, self).__init__(r=r, c=c, radius=radius, tile_i=tile_i,
                                            intensity=intensity, n_tiles=n_tiles,
                                            map_width=map_width)

   def paint(self, map_arr):
      rr, cc = circle_perimeter(self.r, self.c, self.radius, shape=(self.map_width, self.map_width))
      map_arr[self.tile_type, rr, cc] += self.intensity

class Gaussian(Distribution):
   pattern_type = 6
   def __init__(self, mean, std_devs, tile_i, intensity, map_width, n_tiles):
      super().__init__(mean, std_devs, tile_i, intensity, map_width, n_tiles)

   def paint(self, map_arr):
      mean, std_devs = self.mean, self.std_devs
      tile_i, intensity = self.tile_type, self.intensity
      dist = multivariate_normal(mean, self.std_devs, allow_singular=True)
      x, y = np.indices(map_arr[tile_i].shape)
      pos = np.dstack((x, y))
      map_arr[tile_i] += dist.pdf(pos) * intensity


PRIMITIVE_TYPES = [Line, Rectangle, RectanglePerimeter, Circle, CirclePerimeter, Gaussian]
PRIMITIVE_IDS = dict([(i, val) for (i, val) in enumerate(PRIMITIVE_TYPES)])

