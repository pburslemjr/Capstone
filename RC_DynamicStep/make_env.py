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
import gym


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

selectableTanks = []
allAIPlayers = []
allTanks = []

allObstacles = []

allBases = []

allFlags = []

allBullets = []

def reset():
    

	#Add Obstacles
	for o in gameConsts.obstacles:
		obs = Obstacle((o['x'], o['y']), o['size'])
		gameObjects.append(obs)
		allObstacles.append(obs)

	# Add Bases before tanks so that tanks are on top
	for p in gameConsts.players:
		base = Base(p['color'], (p['base']['x'], p['base']['y']), p['base']['size'])
		gameObjects.append(base)
		allBases.append(base)
    
	#Add Tanks
	tankNum = 1 # TODO: replace with dictionary index?
	for p in gameConsts.players:
		for t in p['tanks']:
			tank = Tank(color = p['color'], position=(t['position']['x'], t['position']['y'],), number = tankNum, addToGameObject = addToGameObject)
			gameObjects.append(tank)
			tankNum = tankNum + 1
			allTanks.append(tank)
			if p['human']:
				selectableTanks.append(tank)
			if not p['human']:
				allAIPlayers.append(AI(p['color'], gameObjects))
	

	#Add flags after tanks so they are on top
	for p in gameConsts.players:
		flag = Flag(p['color'], (p['base']['x'], p['base']['y']), gameConsts.FLAG_SIZE)
		gameObjects.append(flag)
		allFlags.append(flag)

	#Display background
	# screen.fill(gameConsts.BACKGROUND_COLOR) # TODO: replace with grass
	for x in range(0, gameConsts.MAP_WIDTH, gameConsts.BACKGROUND_SIZE):
		for y in range(0, gameConsts.MAP_WIDTH, gameConsts.BACKGROUND_SIZE):
			screen.blit(bg, (x,y))


	scoreboard.update()
	#Place objects on top of background
	for obj1 in gameObjects:
		for obj2 in gameObjects:
			if obj1 is obj2:
				continue
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
       
 
	pg.display.flip()

	return [observation()]


def step(actions):
	
	local_rewards = [0]*len(selectableTanks)
	global_rewards = 0.0
	C = 0.05

	# screen.fill(gameConsts.BACKGROUND_COLOR) # TODO: replace with grass
	for x in range(0, gameConsts.MAP_WIDTH, gameConsts.BACKGROUND_SIZE):
		for y in range(0, gameConsts.MAP_WIDTH, gameConsts.BACKGROUND_SIZE):
			screen.blit(bg, (x,y))

	done = False
	
	for i,a in enumerate(actions):
		selectableTanks[i].speed = a[0] * gameConsts.TANK_MAX_SPEED
		selectableTanks[i].angleSpeed = a[1] * gameConsts.TANK_MAX_ROTATION
		if(a[2] == 1):	
			allBullets.append(selectableTanks[i].fire())


	for ai in allAIPlayers:
		ai.control()


	scoreboard.update()
	for obj1 in gameObjects:
		for obj2 in gameObjects:
			if obj1 is obj2:
				continue
			if not(obj1.top > obj2.bottom or obj1.right < obj2.left or obj1.bottom < obj2.top or obj1.left > obj2.right):
				local_rewards, global_rewards = handleHit(obj1, obj2, gameObjects)
		if(obj1.markedForTermination):
			gameObjects.remove(obj1)
			del obj1
			continue
		if (isinstance(obj1, Tank)):
			if(isinstance(obj1.flag, Flag) and obj1.tank_num <= len(selectableTanks)):
				local_rewards[obj1.tank_num] = local_rewards[obj1.tank_num] + (C *  get_dist(allBases[1].position[0], allBases[1].position[1], allBases[0].position[0], allBases[0].position[1]) / get_dist(obj1.position[0], obj1.position[1], allBases[0].position[0], allBases[0].position[1]))
			elif(isinstance(obj1.flag, Flag) and obj1.tank_num > len(selectableTanks)):
				global_rewards = global_rewards - (C *  get_dist(allBases[1].position[0], allBases[1].position[1], allBases[0].position[0], allBases[0].position[1]) / get_dist(obj1.position[0], obj1.position[1], allBases[0].position[0], allBases[0].position[1]))

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
       
 
	for t in allTanks:
		if (isinstance(t.flag, Flag)):
			scoreboard.updateScore(t.color, gameConsts.POINTS_CARRYING_FLAG)

	pg.display.flip()

	return [observation(), reward(local_rewards, global_rewards), 0, []]


def observation():
	obs = [0]*100
	i = 0
	vals = 3

	for obj in allTanks:
		obs[(i*3)] = obj.position[0]
		obs[(i*3)+1] = obj.position[1]
		obs[(i*3)+2] = obj.angle
		i = i+1
	
	for obj in allObstacles:
		obs[(i*3)] = obj.position[0]
		obs[(i*3)+1] = obj.position[1]
		obs[(i*3)+2] = obj.size
		i = i+1

	for obj in allBases:
		obs[(i*3)] = obj.position[0]
		obs[(i*3)+1] = obj.position[1]
		obs[(i*3)+2] = obj.size
		i = i+1
			
	for obj in allFlags:
		obs[(i*3)] = obj.position[0]
		obs[(i*3)+1] = obj.position[1]
		obs[(i*3)+2] = obj.size
		i = i+1

	for obj in allBullets:
		obs[(i*3)] = obj.position[0]
		obs[(i*3)+1] = obj.position[1]
		obs[(i*3)+2] = obj.angle
		i = i+1

	return obs


def reward(local_reward, global_reward):
	r = []
	for i in range(len(local_reward)):
		r.append(local_reward[i] + global_reward)
	return r



def handleHit(o1, o2, gameObjects):
	local_rewards = [0]*len(selectableTanks)
	global_rewards = 0.0
	o1Tank = isinstance(o1, Tank)
	o2Tank = isinstance(o2, Tank)
	if(isinstance(o1, Bullet) and isinstance(o2, Obstacle)):
		allBullets.remove(o1)
		o1.terminate()
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

		#Local Rewards
		if(o1.color != o2.color):
			if(isinstance(o2.flag, Flag)):
				local_rewards[o1.tank_num] = local_rewards[o1.tank_num] + 20.0
			else:
				local_rewards[o1.tank_num] = local_rewards[o1.tank_num] + 10.0
		else:
			local_rewards[o1.tank_num] = local_rewards[o1.tank_num] - 20.0
		
		if(isinstance(o2.flag, Flag)):
			local_rewards[o2.tank_num] = local_rewards[o2.tank_num] - 20.0
		else:
			local_rewards[o2.tank_num] = local_rewards[o2.tank_num] - 10.0

		allBullets.remove(o1)
		o1.terminate()

		if isinstance(o2.flag, Flag):
			o2.flag.dropped()
			o2.setFlag(None)
		o2.setRespawn(pg.time.get_ticks())
	if(o1Tank and isinstance(o2, Flag) and (o1.color != o2.color)):
		if(selectableTanks[0].color == o2.color):
			global_rewards = global_rewards - 50
		else:
			global_rewards = global_rewards + 50

		o1.setFlag(o2)
		o2.setPickedUp(o1)
	if(o1Tank and isinstance(o2, Base) and isinstance(o1.flag, Flag) and o1.color == o2.color):
		scoreboard.updateScore(o1.color, gameConsts.POINTS_RETURNING_FLAG)
		o1.flag.respawn()
		o1.flag.dropped()
		o1.setFlag(None)
		if(selectableTanks[0].color == o1.color):
			global_rewards = global_rewards + 100
		else:
			global_rewards = global_rewards - 100			

	return [local_rewards, global_rewards]



def checkWalls(obj):
	if(obj.top <= 0):
		obj.preventMovement('up')
	elif(obj.bottom >= gameConsts.MAP_HEIGHT):
		obj.preventMovement('down')
	if(obj.right >= gameConsts.MAP_WIDTH):
		obj.preventMovement('right')
	elif(obj.left <= 0):
		obj.preventMovement('left')


def get_dist(source_x, source_y, neighbor_x, neighbor_y):
	return math.sqrt(pow(source_x - neighbor_x, 2) + pow(source_y - neighbor_y, 2))


'''if __name__ == '__main__':
    main()
    pg.quit()'''
