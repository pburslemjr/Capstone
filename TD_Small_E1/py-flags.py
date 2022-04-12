import math
import pygame as pg
from tank import Tank
from bullet import Bullet
from obstacle import Obstacle
from base import Base
from flag import Flag
from scoreboard import Scoreboard
from artificialIntelligence import AI
import gameConsts

pg.init()
screen = gameConsts.screen
selectLast = -gameConsts.SELECT_ADD_TIME # negative to allow immediate select
selectedTanks = set()
scoreboard = Scoreboard(gameConsts.players)

bg = pg.image.load(gameConsts.MAP_BACKGROUND)
bg = pg.transform.scale(bg, (gameConsts.BACKGROUND_SIZE, gameConsts.BACKGROUND_SIZE))


def selectTank(tank):
  global selectLast
  global selectedTanks
  gameTime = pg.time.get_ticks()
  if gameTime - selectLast > gameConsts.SELECT_ADD_TIME:
    selectLast = gameTime
    for t in selectedTanks:
      t.unselect()
    selectedTanks.clear()
  selectedTanks.add(tank)
  tank.select()
              
gameObjects = []
def addToGameObject(go):
  gameObjects.append(go)

def main():
    
    selectableTanks = []
    allTanks = []

    for o in gameConsts.obstacles:
      gameObjects.append(Obstacle((o['x'], o['y']), o['size']))

    # after tanks puts them on top
    for p in gameConsts.players:
      gameObjects.append(Base(p['color'], (p['base']['x'], p['base']['y']), p['base']['size']))
    
    for p in gameConsts.players:
      tankNum = 1 # TODO: replace with dictionary index?
      for t in p['tanks']:
        tank = Tank(color = p['color'], position=(t['position']['x'], t['position']['y'],), number = tankNum, addToGameObject = addToGameObject)
        gameObjects.append(tank)
        tankNum = tankNum + 1
        allTanks.append(tank)
        if p['human']:
          selectableTanks.append(tank)

    # append flags after tanks so they are on top
    for p in gameConsts.players:
      gameObjects.append(Flag(p['color'], (p['base']['x'], p['base']['y']), gameConsts.FLAG_SIZE))

    allAIPlayers = []
    for p in gameConsts.players:
      if not p['human']:
        allAIPlayers.append(AI(p['color'], gameObjects))



    clock = pg.time.Clock()
    done = False
    scoreboardNextTick = 0
    aiNextTick = 0

    while not done:
      clock.tick(gameConsts.FPS)
      currentTick = pg.time.get_ticks()
      for event in pg.event.get():
          if event.type == pg.QUIT:
              done = True
          if event.type == pg.MOUSEBUTTONDOWN:
            pos = pg.mouse.get_pos()
            for t in selectedTanks:
              t.setDestination(pos)
          elif event.type == pg.KEYDOWN:
            if event.key == pg.K_SPACE:
              for t in selectedTanks:
                t.fire()
            if event.key == pg.K_1: # TODO: map K_1 to index of selectableTank array
              selectTank(selectableTanks[0])
            if event.key == pg.K_2:
              selectTank(selectableTanks[1])
            if event.key == pg.K_3:
              selectTank(selectableTanks[2])
      # screen.fill(gameConsts.BACKGROUND_COLOR) # TODO: replace with grass
      for x in range(0, gameConsts.MAP_WIDTH, gameConsts.BACKGROUND_SIZE):
        for y in range(0, gameConsts.MAP_WIDTH, gameConsts.BACKGROUND_SIZE):
          screen.blit(bg, (x,y))

      if (currentTick > aiNextTick ):
        aiNextTick = currentTick + gameConsts.AI_UPDATE_TIMEOUT
        for ai in allAIPlayers:
          ai.control()

      scoreboard.update()
      for obj1 in gameObjects:
        for obj2 in gameObjects:
          if obj1 is obj2: continue
          if not(obj1.top > obj2.bottom or obj1.right < obj2.left or obj1.bottom < obj2.top or obj1.left > obj2.right):
            handleHit(obj1, obj2, gameObjects)
        if(obj1.markedForTermination):
          gameObjects.remove(obj1)
          del obj1
          continue
        if (isinstance(obj1, Tank)):
          checkWalls(obj1)
          if obj1.respawn:
            obj1.checkRespawn(pg.time.get_ticks())
            continue
        if (isinstance(obj1, Flag)):
          if obj1.pickedUp and obj1.pickedUpBy.respawn:
            obj1.pickedUpBy.setFlag(None) 
            obj1.dropped()

        obj1.update()
        obj1.getSprite().draw(screen)
        
      if (currentTick > scoreboardNextTick ):
        scoreboardNextTick = currentTick + gameConsts.ONE_SECOND
        for t in allTanks:
          if (isinstance(t.flag, Flag)):
            scoreboard.updateScore(t.color, gameConsts.POINTS_CARRYING_FLAG)


      pg.display.flip()

def handleHit(o1, o2, gameObjects):
  o1Tank = isinstance(o1, Tank)
  o2Tank = isinstance(o2, Tank)
  if(isinstance(o1, Bullet) and isinstance(o2, Obstacle)): o1.terminate()
  if(o1Tank and isinstance(o2, Obstacle) or o1Tank and o2Tank):
    if (o1.ghost and o2Tank or o2Tank and o2.ghost): # recently respawn tanks can drive through other tanks
      return
    xDiff = o2.position[0] - o1.position[0]
    yDiff = o2.position[1] - o1.position[1]
    if abs(abs(xDiff) - abs(yDiff)) < 3: # ignore exact corner collision. work around
      return
    angle = (math.atan2(-(xDiff), (yDiff)) * 180 / math.pi) % 360
    # print(angle)
    if(angle <= 225 and angle > 135):
      o1.preventMovement('up')
    elif(angle > 225 and angle <= 315):
      o1.preventMovement('right')
    elif (angle <= 45 or angle > 315):
      o1.preventMovement('down')
    else: #(angle > 45 and angle <= 135):
      o1.preventMovement('left')
  if(isinstance(o1, Bullet) and o2Tank and (o1.color != o2.color or gameConsts.FRIENDLY_FIRE)):
    o1.terminate()
    if isinstance(o2.flag, Flag):
      o2.flag.dropped()
      o2.setFlag(None)
    o2.setRespawn(pg.time.get_ticks())
  if(o1Tank and isinstance(o2, Flag) and (o1.color != o2.color)):
    o1.setFlag(o2)
    o2.setPickedUp(o1)
  if(o1Tank and isinstance(o2, Base) and isinstance(o1.flag, Flag) and o1.color == o2.color):
    scoreboard.updateScore(o1.color, gameConsts.POINTS_RETURNING_FLAG)
    o1.flag.respawn()
    o1.flag.dropped()
    o1.setFlag(None)





def checkWalls(obj):
  if(obj.top <= 0):
    obj.preventMovement('up')
  elif(obj.bottom >= gameConsts.MAP_HEIGHT):
    obj.preventMovement('down')
  if(obj.right >= gameConsts.MAP_WIDTH):
    obj.preventMovement('right')
  elif(obj.left <= 0):
    obj.preventMovement('left')

if __name__ == '__main__':
    main()
    pg.quit()
