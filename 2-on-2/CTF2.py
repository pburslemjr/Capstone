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
import numpy as np
import gym
from gym import spaces
from pygame.math import Vector2
import random
random.seed(1)
np.random.seed(1)

class CTF(gym.Env):

	def __init__(self):
		super(CTF, self).__init__()

		self.obs_size = 4*(3+7+(2*3)+(2*3)+(4*3)+2)	#Tank, enemy tank, flag, base, obstacle
		self.episode=0
		#self.action_space = spaces.Box(low=np.array([[-1, -1, 0], [-1, -1, 0], [-1, -1, 0]]) , high=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]), dtype=np.float32)

		self.action_space = spaces.Box(low=np.array([-1, -1, -1]) , high=np.array([1, 1, 1]), dtype=np.float32)

		self.obs_limits_low = [0]*self.obs_size
		self.obs_limits_high = [1000]*self.obs_size
		self.obs = [0]*self.obs_size
		self.eps = 0.3
		self.gen_eps = 0.1#0.1

		self.ep_len = 20480
		self.team_1_score = 0.0
		self.team_2_score = 0.0
		self.tot_score = 0.0

		i=0

		self.obs_limits_low[i] = 0.0
		self.obs_limits_high[i] = 1.0
		i=i+1
		self.obs_limits_low[i] = 0.0
		self.obs_limits_high[i] = 1.0
		i=i+1
		self.obs_limits_low[i] = 0.0
		self.obs_limits_high[i] = 100.0
		i=i+1

		while(i<self.obs_size):
			if((i % (self.obs_size/4)) < 6*(len(gameConsts.players)-1)):
				self.obs_limits_low[i] = 0.0
				self.obs_limits_high[i] = gameConsts.MAP_WIDTH + gameConsts.TANK_SIZE[0]
				i=i+1
				if(i >= self.obs_size):
					break
				self.obs_limits_low[i] = 0.0
				self.obs_limits_high[i] = gameConsts.MAP_HEIGHT + gameConsts.TANK_SIZE[0]
				i=i+1
				if(i >= self.obs_size):
					break
				self.obs_limits_low[i] = -180.0
				self.obs_limits_high[i] = 180.0
				i=i+1
				if(i >= self.obs_size):
					break
				self.obs_limits_low[i] = -180.0
				self.obs_limits_high[i] = 180.0
				i=i+1
				if(i >= self.obs_size):
					break
				self.obs_limits_low[i] = 0.0
				self.obs_limits_high[i] = 1.0
				i=i+1
				if(i >= self.obs_size):
					break
				self.obs_limits_low[i] = 0.0
				self.obs_limits_high[i] = 1.0
				i=i+1
				if(i >= self.obs_size):
					break
				self.obs_limits_low[i] = 0.0
				self.obs_limits_high[i] = 100.0
				i=i+1
			elif((i % (self.obs_size/4)) > (self.obs_size/4)-3):
				self.obs_limits_low[i] = 0.0
				self.obs_limits_high[i] = self.get_dist(0.0, 0.0, gameConsts.MAP_WIDTH, gameConsts.MAP_HEIGHT)
				i=i+1
				if(i >= self.obs_size):
					break
				self.obs_limits_low[i] = 0
				self.obs_limits_high[i] = self.get_dist(0.0, 0.0, gameConsts.MAP_WIDTH, gameConsts.MAP_HEIGHT)
				i=i+1
			else:
				self.obs_limits_low[i] = 0.0
				self.obs_limits_high[i] = gameConsts.MAP_WIDTH
				i=i+1
				if(i >= self.obs_size):
					break
				self.obs_limits_low[i] = 0
				self.obs_limits_high[i] = gameConsts.MAP_HEIGHT
				i=i+1
				if(i >= self.obs_size):
					break
				self.obs_limits_low[i] = -180.0
				self.obs_limits_high[i] = 180.0
				i=i+1

		self.observation_space = spaces.Box(low=np.array(self.obs_limits_low), high=np.array(self.obs_limits_high), dtype=np.float32)


	def addToGameObject(self, go):
		self.gameObjects.append(go)

	def render(self):
		# screen.fill(gameConsts.BACKGROUND_COLOR) # TODO: replace with grass
		for x in range(0, gameConsts.MAP_WIDTH, gameConsts.BACKGROUND_SIZE):
			for y in range(0, gameConsts.MAP_WIDTH, gameConsts.BACKGROUND_SIZE):
				self.screen.blit(self.bg, (x,y))
				
		pg.draw.rect(self.screen, (10,10,10,25), (self.scoreboard.x, self.scoreboard.y, self.scoreboard.width, self.scoreboard.height))
		i = 0
		for key, s in self.scoreboard.score.items():
			text = self.scoreboard.font.render(str(key)+":"+str(s), False, (255,255,255))
			self.screen.blit(text, (self.scoreboard.x + gameConsts.TANK_FONT_SIZE, self.scoreboard.y + gameConsts.TANK_FONT_SIZE + (gameConsts.TANK_FONT_SIZE * i)))
			i = i + 1
			
		for obj1 in self.gameObjects:
			if(isinstance(obj1, Tank)):
				if obj1.selected:
 					#pg.draw.circle(screen, gameConsts.SELECTED_COLOR, self.position, self.radius, 1)
					pg.draw.rect(self.screen, gameConsts.SELECTED_COLOR, [obj1.position[0]-obj1.radius,obj1.position[1]-obj1.radius,2*obj1.radius,2*obj1.radius], 1)
				# self.text = self.font.render(f'{str(self.angle)}-{str(self.direction)}', False, gameConsts.SELECTED_COLOR) # DEBUG
				self.screen.blit(obj1.text, (obj1.position[0], obj1.position[1] + obj1.radius))
			
			obj1.getSprite().draw(self.screen)
		
		pg.display.flip()
		
		
	def reset(self):

		pg.init()
		self.screen = gameConsts.screen
		self.selectLast = -gameConsts.SELECT_ADD_TIME # negative to allow immediate select
		self.scoreboard = Scoreboard(gameConsts.players)
		self.bg = pg.image.load(gameConsts.MAP_BACKGROUND)
		self.bg = pg.transform.scale(self.bg, (gameConsts.BACKGROUND_SIZE, gameConsts.BACKGROUND_SIZE))
		self.gameObjects = []

		self.allTanks = []		
		self.selectableTanks = []
		self.nonselectableTanks = []
		self.allAIPlayers = []

		self.allObstacles = []	
		self.allBases = []
		self.allFlags = []
		self.allBullets = []

		self.tot_score += self.team_1_score
		print(self.tot_score) 
		self.team_1_score = 0.0
		self.team_2_score = 0.0
		
		self.time_steps = 0
		self.alive = 0
		self.kill_interval = 0

		self.dist_to_flag = 1.0
		self.dist_to_base = 0.0

		#Add Obstacles
		for o in gameConsts.obstacles:
			obs = Obstacle((o['x'], o['y']), o['size'])
			self.gameObjects.append(obs)
			self.allObstacles.append(obs)

		'''reversed_obs = []
		for o in range(len(self.allObstacles)-1, -1, -1):
			reversed_obs.append(self.allObstacles[o])
		self.allObstacles = reversed_obs'''

		ply_order = []
		i = 0
		while(i < len(gameConsts.players)):
			next_player = random.randint(0, len(gameConsts.players)-1)
			if(next_player not in ply_order):
				ply_order.append(next_player)
				i += 1
		ply_order = [0, 1]
		print(ply_order)

		
		# Add Bases before tanks so that tanks are on top
		for i in range(len(gameConsts.players)):
			base = Base(gameConsts.players[i]['color'], (gameConsts.players[ply_order[i]]['base']['x'], gameConsts.players[ply_order[i]]['base']['y']), gameConsts.players[i]['base']['size'])
			self.gameObjects.append(base)
			self.allBases.append(base)
			
		#Add flags after tanks so they are on top
		for i in range(len(gameConsts.players)):
			flag = Flag(gameConsts.players[i]['color'], (gameConsts.players[ply_order[i]]['base']['x'], gameConsts.players[ply_order[i]]['base']['y']), gameConsts.FLAG_SIZE)
			self.gameObjects.append(flag)
			self.allFlags.append(flag)

	    
		#Add Tanks
		tankNum = 1 # TODO: replace with dictionary index?
		for i in range(len(gameConsts.players)):
			for t in gameConsts.players[ply_order[i]]['tanks']:
				tank = Tank(color = gameConsts.players[i]['color'], position=(t['position']['x'], t['position']['y'],), number = tankNum, addToGameObject = self.addToGameObject)
				self.gameObjects.append(tank)
				tankNum = tankNum + 1
				self.allTanks.append(tank)
				if gameConsts.players[i]['human']:
					self.selectableTanks.append(tank)
				if not gameConsts.players[i]['human']:
					self.allAIPlayers.append(AI(gameConsts.players[i]['color'], self.gameObjects))
					self.nonselectableTanks.append(tank)


		#Display background
		# screen.fill(gameConsts.BACKGROUND_COLOR) # TODO: replace with grass
		'''for x in range(0, gameConsts.MAP_WIDTH, gameConsts.BACKGROUND_SIZE):
			for y in range(0, gameConsts.MAP_WIDTH, gameConsts.BACKGROUND_SIZE):
				self.screen.blit(self.bg, (x,y))'''


		self.scoreboard.update()
		#Place objects on top of background
		for obj1 in self.gameObjects:
			for obj2 in self.gameObjects:
				if obj1 is obj2:
					continue
				if not(obj1.top > obj2.bottom or obj1.right < obj2.left or obj1.bottom < obj2.top or obj1.left > obj2.right):
					self.handleHit(obj1, obj2)
			if(obj1.markedForTermination):
				self.gameObjects.remove(obj1)
				del obj1
				continue
			if (isinstance(obj1, Tank)):
				self.checkWalls(obj1)
				if obj1.respawn:
					obj1.checkRespawn(self.time_steps)
					continue
			if (isinstance(obj1, Flag)):
				if obj1.pickedUp and obj1.pickedUpBy.respawn:
					obj1.pickedUpBy.setFlag(None) 
					obj1.dropped()
			if(isinstance(obj1, Tank)):
				if(obj1.tank_num <= len(self.selectableTanks)):
					obj1.update(0)
				else:
					obj1.update(1)
			else:
				obj1.update()
			#obj1.getSprite().draw(self.screen)
	       
	 
		#pg.display.flip()
		self.obs = [0]*self.obs_size
		self.obs = self.observation()
		
		return self.obs


	def step(self, actions):
		'''if(self.time_steps > 100):
			self.eps = 0.15'''
		self.time_steps = self.time_steps + 1
		local_rewards = [0]*len(self.selectableTanks)
		global_rewards = 0.0
		C = 10
		render_freq = 25
		# screen.fill(gameConsts.BACKGROUND_COLOR) # TODO: replace with grass
		'''for x in range(0, gameConsts.MAP_WIDTH, gameConsts.BACKGROUND_SIZE):
			for y in range(0, gameConsts.MAP_WIDTH, gameConsts.BACKGROUND_SIZE):
				self.screen.blit(self.bg, (x,y))'''

		done = False
		if(self.team_1_score > 0 or self.time_steps == self.ep_len):
			done = True
			self.episode = self.episode + 1

	
		t = 0
		for i,a in enumerate(actions):
			if(i%3 == 0):
				a = (a - (-1)) / (1 - (-1))
				self.selectableTanks[t].speed = a * gameConsts.TANK_MAX_SPEED
			elif(i%3 == 1):
				self.selectableTanks[t].angleSpeed = a * gameConsts.TANK_MAX_ROTATION
			else:
				a = (a - (-1)) / (1 - (-1))
				a = round(a)
				if(a == 1):
					if(self.selectableTanks[t].fired == 0 and self.selectableTanks[t].respawn == False):	
						self.allBullets.append(self.selectableTanks[t].fire())
						#local_rewards[self.selectableTanks[t].tank_num-1] = local_rewards[self.selectableTanks[t].tank_num-1] - 1.0
					

				t = t+1
		
		self.kill_interval = self.kill_interval + 1

		if(self.selectableTanks[0].respawn == False):
			self.alive = self.alive + 1
		else:
			self.alive = 0

		for ai in self.allAIPlayers:
			ai.control()


		self.scoreboard.update()
		for obj1 in self.gameObjects:
			for obj2 in self.gameObjects:
				if obj1 is obj2:
					continue
				#if not(obj1.top > obj2.bottom or obj1.right < obj2.left or obj1.bottom < obj2.top or obj1.left > obj2.right):
				if(math.sqrt(pow(obj1.position[0] - obj2.position[0], 2) + pow(obj1.position[1] - obj2.position[1], 2)) < max(obj1.size[0], obj1.size[1])/2 + max(obj2.size[0], obj2.size[1])/2 + 10):

					hit_rewards = self.handleHit(obj1, obj2)
					local_rewards = local_rewards + hit_rewards[0]
					global_rewards = global_rewards + hit_rewards[1]
			if(obj1.markedForTermination):
				self.gameObjects.remove(obj1)
				del obj1
				continue
			if (isinstance(obj1, Tank)):

				self.checkWalls(obj1)
				if obj1.respawn:
					obj1.checkRespawn(self.time_steps)
					continue
			if (isinstance(obj1, Flag)):
				if obj1.pickedUp and obj1.pickedUpBy.respawn:
					obj1.pickedUpBy.setFlag(None) 
					obj1.dropped()
			if(isinstance(obj1, Tank)):
				if(obj1.tank_num <= len(self.selectableTanks)):
					obj1.update(0)
				else:
					obj1.update(1)
			else:
				obj1.update()
			#obj1.getSprite().draw(self.screen)
	       
		if(self.time_steps % render_freq == 0):
			self.render()
	 
		for t in self.allTanks:
			self.scoreboard.updateScore(self.team_1_score, self.team_2_score)

		rew = np.array(self.reward(local_rewards, global_rewards) #+ 100*(1-self.stole_flag)*(self.prev_dist_to_flag - self.dist_to_flag) + 100*self.stole_flag*(self.prev_dist_to_base - self.dist_to_base))

		#pg.display.flip()
		self.obs = self.observation()
		
		return [self.obs, rew, done, {}]


	def observation(self):
		obs = [0]*int(self.obs_size/4)
		i = 0
		vals = 3
		tank = self.selectableTanks[0] 
		
		obs[i] = (int(isinstance(tank.flag, Flag)) - self.obs_limits_low[i]) / (self.obs_limits_high[i] - self.obs_limits_low[i])
		self.stole_flag = obs[i]
		i = i+1
		if(tank.fired == 2):
			obs[i] = 1
		else:
			obs[i] = 0
		obs[i] = (obs[i] - self.obs_limits_low[i]) / (self.obs_limits_high[i] - self.obs_limits_low[i])
		i = i+1
		obs[i] = (tank.reload_time - tank.fired - self.obs_limits_low[i]) / (self.obs_limits_high[i] - self.obs_limits_low[i])
		i = i+1

		for obj in self.allTanks:
			if(obj != tank):
				obs[i] = (abs(obj.position[0] - tank.position[0]) - self.obs_limits_low[i]) / (self.obs_limits_high[i] - self.obs_limits_low[i])
				i = i+1
				obs[i] = (abs(obj.position[1] - tank.position[1]) - self.obs_limits_low[i]) / (self.obs_limits_high[i] - self.obs_limits_low[i])
				i = i+1

				obs[i] = (self.get_ang_diff(obj.angle, self.get_rel_ang(obj.position[0], obj.position[1], tank.position[0], tank.position[1])) - self.obs_limits_low[i]) / (self.obs_limits_high[i] - self.obs_limits_low[i])
				i = i+1

				obs[i] = (self.get_ang_diff(tank.angle, self.get_rel_ang(tank.position[0], tank.position[1], obj.position[0], obj.position[1])) - self.obs_limits_low[i]) / (self.obs_limits_high[i] - self.obs_limits_low[i])
				i = i+1

				obs[i] = (int(isinstance(obj.flag, Flag)) - self.obs_limits_low[i]) / (self.obs_limits_high[i] - self.obs_limits_low[i])
				i = i+1

				if(obj.fired == 2):
					obs[i] = 1
				else:
					obs[i] = 0
				obs[i] = (obs[i] - self.obs_limits_low[i]) / (self.obs_limits_high[i] - self.obs_limits_low[i])
				i = i+1
				obs[i] = (obj.reload_time - obj.fired - self.obs_limits_low[i]) / (self.obs_limits_high[i] - self.obs_limits_low[i])
				i = i+1
				#obs[i] = obj.fired
				#i = i+1

		for obj in self.allFlags:
			obs[i] = (abs(obj.position[0] - tank.position[0]) - self.obs_limits_low[i]) / (self.obs_limits_high[i] - self.obs_limits_low[i])
			i = i+1
			obs[i] = (abs(obj.position[1] - tank.position[1]) - self.obs_limits_low[i]) / (self.obs_limits_high[i] - self.obs_limits_low[i])
			i = i+1
			obs[i] = (self.get_ang_diff(tank.angle, self.get_rel_ang(tank.position[0], tank.position[1], obj.position[0], obj.position[1])) - self.obs_limits_low[i]) / (self.obs_limits_high[i] - self.obs_limits_low[i])
			i = i+1

		for obj in self.allBases:
			obs[i] = (abs(obj.position[0] - tank.position[0]) - self.obs_limits_low[i]) / (self.obs_limits_high[i] - self.obs_limits_low[i])
			i = i+1
			obs[i] = (abs(obj.position[1] - tank.position[1]) - self.obs_limits_low[i]) / (self.obs_limits_high[i] - self.obs_limits_low[i])
			i = i+1
			obs[i] = (self.get_ang_diff(tank.angle, self.get_rel_ang(tank.position[0], tank.position[1], obj.position[0], obj.position[1])) - self.obs_limits_low[i]) / (self.obs_limits_high[i] - self.obs_limits_low[i])
			i = i+1

		for obj in self.allObstacles:
			obs[i] = (abs(obj.position[0] - tank.position[0]) - self.obs_limits_low[i]) / (self.obs_limits_high[i] - self.obs_limits_low[i])
			i = i+1
			obs[i] = (abs(obj.position[1] - tank.position[1]) - self.obs_limits_low[i]) / (self.obs_limits_high[i] - self.obs_limits_low[i])
			i = i+1
			obs[i] = (self.get_ang_diff(tank.angle, self.get_rel_ang(tank.position[0], tank.position[1], obj.position[0], obj.position[1])) - self.obs_limits_low[i]) / (self.obs_limits_high[i] - self.obs_limits_low[i])
			i = i+1
			
		for obj in self.allFlags:
			if(obj.color == "red"):
				obs[i] = (self.get_dist(tank.position[0], tank.position[1], obj.position[0], obj.position[1]) - self.obs_limits_low[i]) / (self.obs_limits_high[i] - self.obs_limits_low[i])
				self.prev_dist_to_flag = self.dist_to_flag
				self.dist_to_flag = obs[i]
				i = i+1

		for obj in self.allBases:
			if(obj.color == "blue"):
				obs[i] = (self.get_dist(tank.position[0], tank.position[1], obj.position[0], obj.position[1]) - self.obs_limits_low[i]) / (self.obs_limits_high[i] - self.obs_limits_low[i])
				self.prev_dist_to_base = self.dist_to_base
				self.dist_to_base = obs[i]
				i = i+1


		ob = np.concatenate((np.array(self.obs[int(self.obs_size/4):]), np.array(obs)))
			
		return np.array(ob)


	def reward(self, local_reward, global_reward):
		'''r = []
		for i in range(len(local_reward)):
			r.append(local_reward[i] + global_reward)'''
		r = 0.0
		for i in range(len(local_reward)):
			r = r + local_reward[i]
		r = r + global_reward
		return r



	def handleHit(self, o1, o2):
		local_rewards = [0]*len(self.selectableTanks)
		global_rewards = 0.0
		o1Tank = isinstance(o1, Tank)
		o2Tank = isinstance(o2, Tank)
		if(isinstance(o1, Bullet) and isinstance(o2, Obstacle)):
			if(o1 in self.allBullets):
				self.allBullets.remove(o1)
			o1.terminate()
		if(o1Tank and isinstance(o2, Obstacle) or o1Tank and o2Tank):
			if (o1.ghost and o2Tank or o2Tank and o2.ghost): # recently respawn tanks can drive through other tanks
				return [local_rewards, global_rewards]
			xDiff = o2.position[0] - o1.position[0]
			yDiff = o2.position[1] - o1.position[1]
			if abs(abs(xDiff) - abs(yDiff)) < 0.0: # ignore exact corner collision. work around
				return [local_rewards, global_rewards]
			angle = (math.atan2(-(xDiff), (yDiff)) * 180 / math.pi) % 360
			# print(angle)
			#print("Handle_hit", self.time_steps)
			if(angle <= 225 and angle > 135):
				o1.preventMovement('up')
			elif(angle > 225 and angle <= 315):
				o1.preventMovement('right')
			elif (angle <= 45 or angle > 315):
				o1.preventMovement('down')
			else: #(angle > 45 and angle <= 135):
				o1.preventMovement('left')
		if(isinstance(o1, Bullet) and o2Tank and (o1.color != o2.color or gameConsts.FRIENDLY_FIRE) and o1.tank_num != o2.tank_num):
			if(o2.ghost or o2.respawn):
				return [local_rewards, global_rewards]
			'''#Local Rewards
			if(o1.tank_num <= len(self.selectableTanks)):
				if(o1.color != o2.color):
					if(isinstance(o2.flag, Flag)):
						local_rewards[o1.tank_num-1] = local_rewards[o1.tank_num-1] + 10.0 - 0.001 * self.kill_interval
					else:
						local_rewards[o1.tank_num-1] = local_rewards[o1.tank_num-1] + 10.0 - 0.001 * self.kill_interval
					self.kill_interval = 0
				else:
					local_rewards[o1.tank_num-1] = local_rewards[o1.tank_num-1] - 20.0
			else:
				if(o2.tank_num <= len(self.selectableTanks)):
					if(isinstance(o2.flag, Flag)):
						local_rewards[o2.tank_num-1] = local_rewards[o2.tank_num-1] - 0.0#20.0
					else:
						local_rewards[o2.tank_num-1] = local_rewards[o2.tank_num-1] - 0.0


			if(o1.tank_num <= len(self.selectableTanks)):
				if(o1.color != o2.color):
					self.team_1_score += 10.0
				else:
					self.team_1_score += -20.0
			else:
				if(o1.color != o2.color):
					self.team_2_score += 10.0
				else:
					self.team_2_score += -20.0'''


			if(o1 in self.allBullets):
				self.allBullets.remove(o1)
			o1.terminate()

			if isinstance(o2.flag, Flag):
				o2.flag.dropped()
				o2.setFlag(None)
			o2.setRespawn(self.time_steps)
		if(o1Tank and isinstance(o2, Flag) and (o1.color != o2.color)):
			'''if(isinstance(o1.flag, Flag) == False):
				if(self.selectableTanks[0].color == o2.color):
					global_rewards = global_rewards - 50
					self.team_2_score += 50
				else:
					global_rewards = global_rewards + 50
					self.team_1_score += 50'''

			o1.setFlag(o2)
			o2.setPickedUp(o1)
		if(o1Tank and isinstance(o2, Base) and isinstance(o1.flag, Flag) and o1.color == o2.color):
			self.scoreboard.updateScore(o1.color, gameConsts.POINTS_RETURNING_FLAG)
			o1.flag.respawn()
			o1.flag.dropped()
			o1.setFlag(None)
			if(self.selectableTanks[0].color == o1.color):
				global_rewards = global_rewards + 100
				self.team_1_score += 100
			else:
				global_rewards = global_rewards - 0#100			
				self.team_2_score += 100

		return [local_rewards, global_rewards]



	def checkWalls(self, obj):
		if(obj.top <= 0):
			obj.preventMovement('up')
		elif(obj.bottom >= gameConsts.MAP_HEIGHT):
			obj.preventMovement('down')
		if(obj.right >= gameConsts.MAP_WIDTH):
			obj.preventMovement('right')
		elif(obj.left <= 0):
			obj.preventMovement('left')


	def control(self, obs):
		for t in self.selectableTanks:
			if(random.random() < self.gen_eps):# and isinstance(t.flag, Flag) == False):
				rand_act = self.action_space.sample()
				allow_attack = 0
				for e in self.nonselectableTanks:
					if(self.get_dist(t.position[0], t.position[1], e.position[0], e.position[1]) < gameConsts.FIRE_ENEMY_RANGE):
						allow_attack = 1
						break
				if(allow_attack == 0):
					rand_act[2] = 0
				rand_act[2] = (rand_act[2] * (1 - (-1))) + (-1)
				return rand_act
			else:
				des_x = t.position[0]
				des_y = t.position[1]
				fire = 0
				if (t.respawn):
					speed, ang_speed = self.control_update(t.position[0], t.position[1], des_x, des_y, t.direction, t.radius)
					return np.array([speed, ang_speed, fire])
				if (isinstance(t.flag, Flag)):
					des_x, des_y = self.allBases[0].position
					speed, ang_speed = self.control_update(t.position[0], t.position[1], des_x, des_y, t.direction, t.radius)
					return np.array([speed, ang_speed, fire])

				# find closest tank
				tankTarget = {'dist': math.inf, 'enemyTank': None}
				for e in self.nonselectableTanks:
					dist = math.hypot(e.position[0] - t.position[0], e.position[1] - t.position[1])
					if (not e.respawn and dist < tankTarget['dist']):
						tankTarget['enemyTank'] = e
						tankTarget['dist'] = dist

				#random.shuffle(self.allFlags)
				attackMode = True
				if(self.allFlags[1].pickedUpBy is None):
					des_x, des_y = self.allFlags[1].position
					attackMode = False


				if((tankTarget['enemyTank'] is not None and (dist < gameConsts.SIGHT_ENEMY_RANGE or attackMode))):
					des_x, des_y = tankTarget['enemyTank'].position
					if(tankTarget['dist'] < gameConsts.FIRE_ENEMY_RANGE+20):
						fire = 1
			
				speed, ang_speed = self.control_update(t.position[0], t.position[1], des_x, des_y, t.direction, t.radius)
				speed = (speed * (1 - (-1))) + (-1)
				fire = (fire * (1 - (-1))) + (-1)
				speed = np.random.normal(speed, 1.0, 1)[0]
				ang_speed = np.random.normal(ang_speed, 1.0, 1)[0]
				if(speed < -1):
					speed = -1
				elif(speed > 1):
					speed = 1
				if(ang_speed < -1):
					ang_speed = -1
				elif(ang_speed > 1):
					ang_speed = 1
				return np.array([speed, ang_speed, fire])


	def control_update(self, cur_x, cur_y, des_x, des_y, direction, radius):
		xDiff = des_x - cur_x
		yDiff = des_y - cur_y
		angleSpeed = 0.0
		if(abs(xDiff) > 1 and abs(yDiff) > 1): # if at destination, don't calc angleSpeed
			destVector = Vector2(xDiff, yDiff)
			destAngle = round(direction.angle_to(destVector), 2)
			if (destAngle > 180): # compensate for destAngle_to picking the wrong direction in these cases
				destAngle = -360 + destAngle
			if (destAngle < -180):
				destAngle = 360 + destAngle
			angleSpeed = round(self.getMaxRotation(destAngle), 2)

		distance = math.hypot(cur_x - des_x, cur_y - des_y)
		if distance < radius:
			speed = round(4 * distance / radius, 2)
		else:
			speed = gameConsts.TANK_MAX_SPEED
		
		speed = speed/gameConsts.TANK_MAX_SPEED
		angleSpeed = angleSpeed / gameConsts.TANK_MAX_ROTATION
		return [speed, angleSpeed]

	def getMaxRotation(self, desiredAngle):
		if (desiredAngle > gameConsts.TANK_MAX_ROTATION):
			return gameConsts.TANK_MAX_ROTATION
		elif (desiredAngle < -gameConsts.TANK_MAX_ROTATION):
			return -gameConsts.TANK_MAX_ROTATION
		else:
			return desiredAngle



	def get_dist(self, source_x, source_y, neighbor_x, neighbor_y):
		return math.sqrt(pow(source_x - neighbor_x, 2) + pow(source_y - neighbor_y, 2))

	def get_rel_ang(self, source_x, source_y, neighbor_x, neighbor_y):
		if((source_x - neighbor_x) == 0.0 and (source_y - neighbor_y) != 0.0):
			if((neighbor_y - source_y) > 0.0):
				return 90.0
			else:
				return 270.0

		if((source_x - neighbor_x) == 0.0 and (source_y - neighbor_y) == 0.0):
			return 0.0

		angle = math.atan((neighbor_y - source_y)/(neighbor_x - source_x)) * 180.0/math.pi
		if(neighbor_x < source_x):
			angle = 180.0 + angle
		elif((neighbor_y < source_y) and (neighbor_x > source_x)):
			angle = 360.0 + angle
		return angle

	def get_ang_diff(self, ang_1, ang_2):
		if(abs(ang_1 - ang_2) > abs(ang_1 + 360.0 - ang_2)):
			ang_1 = ang_1 + 360.0
		elif(abs(ang_1 - ang_2) > abs(ang_1 - ang_2 - 360.0)):
			ang_2 = ang_2 + 360.0

		if(abs(ang_1 - ang_2) > 180.0):
			print("Ang_Diff", ang_1 - ang_2)
		return ang_1 - ang_2


	'''if __name__ == '__main__':
	    main()
	    pg.quit()'''
