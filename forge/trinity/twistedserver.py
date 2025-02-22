from pdb import set_trace as T
import numpy as np

from signal import signal, SIGINT
import sys, os, json, pickle, time
import ray

from twisted.internet import reactor
from twisted.internet.task import LoopingCall
from twisted.python import log
from twisted.web.server import Site
from twisted.web.static import File

from autobahn.twisted.websocket import WebSocketServerFactory, \
    WebSocketServerProtocol
from autobahn.twisted.resource import WebSocketResource

def sign(x):
    return int(np.sign(x))

def softmax(x):
    exp = np.exp(x)
    return exp / np.sum(exp)

def move(orig, targ):
    ro, co = orig
    rt, ct = targ
    dr = rt - ro
    dc = ct - co
    if abs(dr) > abs(dc):
        return ro + sign(dr), co
    elif abs(dc) > abs(dr):
        return ro, co + sign(dc)
    else:
        return ro + sign(dr), co + sign(dc)

class GodswordServerProtocol(WebSocketServerProtocol):
#<<<<<<< HEAD
#   def __init__(self):
#       super().__init__()
#       print("Created a server")
#       self.frame  = 0
#
#       #"connected" is already used by WSSP
#       self.isConnected = False
#       self.pos = [512, 512]
#       self.cmd = None
#
#       self.WINDOW = 128
#
#   def onOpen(self):
#       print("Opened connection to server")
#
#   def onClose(self, wasClean, code=None, reason=None):
#       self.isConnected = False
#       print('Connection closed')
#
#   def connectionMade(self):
#       super().connectionMade()
#       self.factory.clientConnectionMade(self)
#
#   def connectionLost(self, reason):
#       super().connectionLost(reason)
#       self.factory.clientConnectionLost(self)
#
#   #Not used without player interaction
#   def onMessage(self, packet, isBinary):
#       print("Server packet", packet)
#       packet    = packet.decode()
#       _, packet = packet.split(';') #Strip headeer
#       r, c, cmd = packet.split(' ') #Split camera coords
#       if len(cmd) == 0:
#           cmd = None
#
#       self.pos = [int(r), int(c)]
#       self.cmd = cmd
#
#       self.isConnected = True
#
#   def onConnect(self, request):
#       print("WebSocket connection request: {}".format(request))
#       realm = self.factory.realm
#       self.realm = realm
#       self.frame += 1
#
#       data = self.serverPacket()
#       self.sendUpdate()
#
#   def serverPacket(self):
#       data = self.realm.clientData()
#       return data
#
#   def sendUpdate(self):
#       data = self.serverPacket()
#
#       packet               = {}
#       packet['resource']   = data['resource']
#       packet['player']     = data['player']
#       packet['npc']        = data['npc']
#       packet['pos']        = data['pos']
#       packet['wilderness'] = data['wilderness']
#
#       config = data['config']
#
#       print('Is Connected? : {}'.format(self.isConnected))
#       if not self.isConnected:
#          packet['map']    = data['environment']
#          packet['border'] = config.TERRAIN_BORDER
#          packet['size']   = config.TERRAIN_SIZE
#
#       if 'overlay' in data:
#          packet['overlay'] = data['overlay']
#          print('SENDING OVERLAY: ', len(packet['overlay']))
#
#       packet = json.dumps(packet).encode('utf8')
#       self.sendMessage(packet, False)
#=======
    def __init__(self):
        super().__init__()
        print("Created a server")
        self.frame  = 0

        #"connected" is already used by WSSP
        self.isConnected = False
        self.pos = [512, 512]
        self.cmd = None

        self.WINDOW = 128

    def onOpen(self):
        print("Opened connection to server")

    def onClose(self, wasClean, code=None, reason=None):
        self.isConnected = False
        print('Connection closed')

    def connectionMade(self):
        super().connectionMade()
        self.factory.clientConnectionMade(self)

    def connectionLost(self, reason):
        super().connectionLost(reason)
        self.factory.clientConnectionLost(self)

    #Not used without player interaction
    def onMessage(self, packet, isBinary):
        print("Server packet", packet)
        packet    = packet.decode()
        _, packet = packet.split(';') #Strip headeer
        r, c, cmd = packet.split(' ') #Split camera coords
        if len(cmd) == 0:
            cmd = None

        self.pos = [int(r), int(c)]
        self.cmd = cmd

        self.isConnected = True

    def onConnect(self, request):
        print("WebSocket connection request: {}".format(request))
        realm = self.factory.realm
        self.realm = realm
        self.frame += 1

        data = self.serverPacket()
        self.sendUpdate()

    def serverPacket(self):
        data = self.realm.packet
        return data

    def sendUpdate(self):
        data = self.serverPacket()

        packet               = {}
        packet['resource']   = data['resource']
        packet['player']     = data['player']
        packet['npc']        = data['npc']
        packet['pos']        = data['pos']
        packet['wilderness'] = data['wilderness']

        config = data['config']

        print('Is Connected? : {}'.format(self.isConnected))
        if not self.isConnected:
            packet['map']    = data['environment']
            packet['border'] = config.TERRAIN_BORDER
            packet['size']   = config.TERRAIN_SIZE

        if 'overlay' in data:
           packet['overlay'] = data['overlay']
           print('SENDING OVERLAY: ', len(packet['overlay']))

        packet = json.dumps(packet).encode('utf8')
        self.sendMessage(packet, False)
#>>>>>>> 1473e2bf0dd54f0ab2dbf0d05f6dbb144bdd1989

class WSServerFactory(WebSocketServerFactory):
    def __init__(self, ip, realm, step):
        super().__init__(ip)
        self.realm, self.step = realm, step
        self.time = time.time()
        self.clients = []

        self.pos = [40, 40]
        self.cmd = None
        self.tickRate = 0.6
        self.tick = 0

        self.step(self.pos, self.cmd)
        lc = LoopingCall(self.announce) #If this misses a tick, it waits for a whole cycle
        lc.start(self.tickRate)

    def announce(self):
        self.tick += 1
        uptime = np.round(self.tickRate*self.tick, 1)
        print('Wall Clock: ', time.time() - self.time, 'Uptime: ', uptime, ', Tick: ', self.tick)
        self.time = time.time()

        for client in self.clients:
            client.sendUpdate()
            if client.pos is not None:
                self.pos = client.pos
                self.cmd = client.cmd

        self.step(self.pos, self.cmd)

    def clientConnectionMade(self, client):
        self.clients.append(client)

    def clientConnectionLost(self, client):
        self.clients.remove(client)

class Application:
   def __init__(self, realm, step):
      signal(SIGINT, self.kill)
      self.realm = realm
      log.startLogging(sys.stdout)
      port = 8080

      factory          = WSServerFactory(u'ws://localhost:' + str(port), realm, step)
      factory.protocol = GodswordServerProtocol 
      resource         = WebSocketResource(factory)

      # We server static files under "/" and our WebSocket 
      # server under "/ws" (note that Twisted uses bytes 
      # for URIs) under one Twisted Web Site
      root = File(".")
      root.putChild(b"ws", resource)
      site = Site(root)

      reactor.listenTCP(port, site)
      reactor.run()

   def kill(*args):
      print("Killed by user")
      reactor.stop()
      os._exit(0)

