from tank import Tank
from bullet import Bullet
from obstacle import Obstacle
from base import Base
from flag import Flag
import random
import math
import gameConsts


# This AI is probably be too difficult for human player to beat

class AI():
  def __init__(self, color, gameObjects):
    self.color = color
    self.myTanks = []
    self.enemyTanks = []
    self.myFlag = None
    self.enemyFlags = []
    self.myBase = None
    self.enemyBases = []
    for go in gameObjects:
      if (isinstance(go, Tank)):
        if go.color == self.color:
          self.myTanks.append(go)
        else:
          self.enemyTanks.append(go)
      elif (isinstance(go, Flag)):
        if go.color == self.color:
          self.myFlag = go
        else:
          self.enemyFlags.append(go)
      elif (isinstance(go, Base)):
        if go.color == self.color:
          self.myBase = go
        else:
          self.enemyBases.append(go)

  def control(self):
    for t in self.myTanks:
      if (t.respawn):
        continue
      if (isinstance(t.flag, Flag)):
        t.setDestination(self.myBase.position)
        continue

      # find closest tank
      tankTarget = {'dist': math.inf, 'enemyTank': None}
      for e in self.enemyTanks:
        dist = math.hypot(e.position[0] - t.position[0], e.position[1] - t.position[1])
        if (not e.respawn and dist < tankTarget['dist']):
          tankTarget['enemyTank'] = e
          tankTarget['dist'] = dist

      random.shuffle(self.enemyFlags)
      attackMode = True
      for f in self.enemyFlags:
        if(f.pickedUpBy is None):
          t.setDestination(f.position)
          attackMode = False
          continue


      if((tankTarget['enemyTank'] is not None and (dist < gameConsts.SIGHT_ENEMY_RANGE or attackMode))):
        t.setDestination(tankTarget['enemyTank'].position)

        if(tankTarget['dist'] < gameConsts.FIRE_ENEMY_RANGE):
          if(t.fired == 0):
            t.fire()
        continue
