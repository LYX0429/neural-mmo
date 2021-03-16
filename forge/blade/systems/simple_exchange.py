from pdb import set_trace as T

from collections import defaultdict, deque
import math

class Offer:
   def __init__(self, seller, item):
      self.seller = seller
      self.item   = item

   '''
   def __lt__(self, offer):
      return self.price < offer.price
      
   def __le__(self, offer):
      return self.price <= offer.price

   def __eq__(self, offer):
      return self.price == offer.price

   def __ne__(self, offer):
      return self.price != offer.price

   def __gt__(self, offer):
      return self.price > offer.price

   def __ge__(self, offer):
      return self.price >= offer.price
   '''

#Why is the api so weird...
class Queue(deque):
   def __init__(self):
      super().__init__()
      self.price = None

   def push(self, x):
      self.appendleft(x)
   
   def peek(self):
      if len(self) > 0:
         return self[-1]
      return None

class Listing:
   def __init__(self):
      self.offers = defaultdict(Queue)
      self.alpha  = 0.01
      self.step()

   def step(self):
      self.volume = 0

   def available(self, level=None):
      if level is None:
         return self.level()
   
      offers = self.offers[level]
      if len(offers) == 0:
         return None

      return offers

   def supply(self):
      total = 0
      for level in range(99, 0, -1):
         total += len(self.offers[level])
      return total

   def level(self):
      '''Highest level on market'''
      for level in range(99, 0, -1):
         if len(self.offers[level]) == 0:
            continue
         return level

   def price(self):
      '''Highest level price on market'''
      for level in range(99, 0, -1):
         if len(offers := self.offers[level]) == 0:
            continue
         return offers.price

   def minPrice(self, minLevel, maxLevel):
      price = None
      for level in range(maxLevel, minLevel, -1):
         offers = self.offers[level]
         if not hasattr(offers, 'price'):
            continue

         if offers.price and (price is None or offers.price < price):
            price = offers.price

      return price

   def maxPrice(self, minLevel, maxLevel):
      price = None
      for level in range(maxLevel, minLevel, -1):
         offers = self.offers[level]
         if not hasattr(offers, 'price'):
            continue

         if offers.price and (price is None or offers.price > price):
            price = offers.price

      return price

   def value(self):
      '''Total value'''
      total = 0
      for level in range(99, 0, -1):
         offers = self.offers[level]
         if n := len(offers):
            total += n * offers.price
      return total

   def buy(self, buyer, minLevel, maxLevel):
      bound = self.minPrice(minLevel, maxLevel)
      if not bound or bound > buyer.inventory.gold.quantity:
         return

      for level in range(maxLevel, minLevel, -1):
         offers = self.offers[level]
         if not offers.price:
            continue

         adjusted     = (1 + self.alpha) * offers.price 
         price        = math.ceil(adjusted)

         if buyer.inventory.gold.quantity < price:
            continue

         offers.price = price

         if not offers:
            continue

         offer        = offers.pop()
         seller       = offer.seller
         item         = offer.item

         buyer.inventory.receivePurchase(item)
         #print('Buy {}: {}'.format(item.__name__, price))
 
         seller.inventory.gold.quantity += price
         buyer.inventory.gold.quantity  -= price

         buyer.buys   += 1
         seller.sells += 1
         self.volume  += 1

         return price

      return False
         
   def sell(self, seller, item):
      level    = item.level.val
      offers   = self.offers[level]

      adjusted = 1
      if offers:
         adjusted = (1 - self.alpha) * offers.price
      else:
         minPrice = self.minPrice(0, level)
         if minPrice:
            adjusted = minPrice

      price        = max(math.floor(adjusted), 1)
      offers.price = price

      #print('Sell {}: {}'.format(item.__class__.__name__, price))

      if price == 1 and len(offers):
         seller.inventory.gold.quantity += 1
      else:
         offer = Offer(seller, item)
         self.offers[level].push(offer)

      seller.inventory.remove(item)

class Exchange:
   def __init__(self):
      self.items  = defaultdict(Listing)

   def step(self):
      for item, listing in self.items.items():
         listing.step()

   def available(self, item):
      return self.items[item].available()

   def buy(self, buyer, item, minLevel, maxLevel):
      return self.items[item].buy(buyer, minLevel, maxLevel)
    
   def sell(self, seller, item):
      self.items[type(item)].sell(seller, item)
