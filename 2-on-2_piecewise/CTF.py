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
PHASE_AT_EPISODE = 200
random.seed(1)
np.random.seed(1)

class CTF(gym.Env):

    def __init__(self):
        super(CTF, self).__init__()

        #Last 4 frames of both tank info and location of flags
        self.obs_size = 4*(3+7*3+(2*3)+(4*3)+(2*3))
        self.episode=0

        #Define a continuous action space with action bounds
        self.action_space = spaces.Box(low=np.array([-1, -1, -1, -1, -1, -1]) , high=np.array([1, 1, 1, 1, 1, 1]), dtype=np.float32)

        self.obs_limits_low = [0]*self.obs_size
        self.obs_limits_high = [1000]*self.obs_size
        self.obs = [0]*self.obs_size

        #Used to add randomness when generating trajectory for pre-training
        self.gen_eps = 0.75

        #Team scores
        self.team_1_score = 0.0
        self.team_2_score = 0.0
        self.rew_frac = 1.0

        #Setting maximum and minimum value limits on each observation
        i=0
        while(i<self.obs_size):
            if((i % (self.obs_size/4)) < 6*len(gameConsts.players)):
                self.obs_limits_low[i] = -gameConsts.TANK_SIZE[0]
                self.obs_limits_high[i] = gameConsts.MAP_WIDTH
                i=i+1
                if(i >= self.obs_size):
                    break
                self.obs_limits_low[i] = -gameConsts.TANK_SIZE[0]
                self.obs_limits_high[i] = gameConsts.MAP_HEIGHT
                i=i+1
                if(i >= self.obs_size):
                    break
                self.obs_limits_low[i] = 0.0
                self.obs_limits_high[i] = 360.0
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
            else:
                self.obs_limits_low[i] = 0.0
                self.obs_limits_high[i] = gameConsts.MAP_WIDTH
                i=i+1
                if(i >= self.obs_size):
                    break
                self.obs_limits_low[i] = 0.0
                self.obs_limits_high[i] = gameConsts.MAP_HEIGHT
                i=i+1

        #Define bounded observation space
        self.observation_space = spaces.Box(low=np.array(self.obs_limits_low), high=np.array(self.obs_limits_high), dtype=np.float32)


    def addToGameObject(self, go):
        self.gameObjects.append(go)


    def reset(self):

        if gameConsts.render:
            pg.init()
            self.screen = gameConsts.screen
        self.selectLast = -gameConsts.SELECT_ADD_TIME # negative to allow immediate select
        if gameConsts.render:
            self.scoreboard = Scoreboard(gameConsts.players)
            self.bg = pg.image.load(gameConsts.MAP_BACKGROUND)
            self.bg = pg.transform.scale(self.bg, (gameConsts.BACKGROUND_SIZE, gameConsts.BACKGROUND_SIZE))
        self.gameObjects = []

        #List of all tanks in the environment
        self.allTanks = []
        self.BlueTanks = []
        self.RedTanks = []

        #List of tanks controlled by RL algorithm
        self.selectableTanks = []

        #List of enemy tanks
        self.nonselectableTanks = []
        self.allAIPlayers = []

        #All the obstacles in the environment, such as walls
        self.allObstacles = []

        #The bases
        self.allBases = []
        self.allFlags = []
        self.allBullets = []

        self.team_1_score = 0.0
        self.team_2_score = 0.0

        self.dist_to_flag = [1.0, 1.0, 1.0, 1.0]
        self.prev_dist_to_flag = [1.0, 1.0, 1.0, 1.0]
        self.dist_to_base = [0.0, 0.0, 0.0, 0.0]
        self.prev_dist_to_base = [0.0, 0.0, 0.0, 0.0]

        #Number of time steps of current episode
        self.time_steps = 0

        #Number of time steps the RL agent has been alive without being destroyed
        self.alive = 0

        #Add Obstacles
        for o in gameConsts.obstacles:
            obs = Obstacle((o['x'], o['y']), o['size'])
            self.gameObjects.append(obs)
            self.allObstacles.append(obs)

        # Add Bases before tanks so that tanks are on top
        for p in gameConsts.players:
            base = Base(p['color'], (p['base']['x'], p['base']['y']), p['base']['size'])
            self.gameObjects.append(base)
            self.allBases.append(base)

        #Add Tanks
        tankNum = 1 # TODO: replace with dictionary index?
        for p in gameConsts.players:
            for t in p['tanks']:
                tank = Tank(color = p['color'], position=(t['position']['x'], t['position']['y'],), number = tankNum, addToGameObject = self.addToGameObject)
                self.gameObjects.append(tank)
                tankNum = tankNum + 1   #Identify tanks by giving each tank a number
                self.allTanks.append(tank)
                if p['human']:
                    self.selectableTanks.append(tank)
                    self.BlueTanks.append(tank)
                if not p['human']:
                    self.allAIPlayers.append(AI(p['color'], self.gameObjects))
                    self.nonselectableTanks.append(tank)
                    self.RedTanks.append(tank)

        #Add flags after tanks so they are on top
        for p in gameConsts.players:
            flag = Flag(p['color'], (p['base']['x'], p['base']['y']), gameConsts.FLAG_SIZE)
            self.gameObjects.append(flag)
            self.allFlags.append(flag)

        #Display background
        # screen.fill(gameConsts.BACKGROUND_COLOR) # TODO: replace with grass
        if gameConsts.render:
            for x in range(0, gameConsts.MAP_WIDTH, gameConsts.BACKGROUND_SIZE):
                for y in range(0, gameConsts.MAP_WIDTH, gameConsts.BACKGROUND_SIZE):
                    self.screen.blit(self.bg, (x,y))


            self.scoreboard.update()
        #Place objects on top of background
        for obj1 in self.gameObjects:
            for obj2 in self.gameObjects:
                if obj1 is obj2:
                    continue
                if not(obj1.top > obj2.bottom or obj1.right < obj2.left or obj1.bottom < obj2.top or obj1.left > obj2.right):
                    self.handleHit(obj1, obj2)  #Handling collision of any two objects in the environment
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
            if gameConsts.render:
                obj1.getSprite().draw(self.screen)


        if gameConsts.render:
            pg.display.flip()
        self.obs = [0]*self.obs_size
        self.obs = self.observation()   #Get current observation
        return self.obs


    def step(self, actions):

        self.time_steps = self.time_steps + 1
        #Rewards for each tank accomplishment such as killing a tank
        local_rewards = [0]*len(self.selectableTanks)
        #Global rewards such as stealing flag or having flag stolen
        global_rewards = 0.0

        C = 10
        # screen.fill(gameConsts.BACKGROUND_COLOR) # TODO: replace with grass
        if gameConsts.render:
            for x in range(0, gameConsts.MAP_WIDTH, gameConsts.BACKGROUND_SIZE):
                for y in range(0, gameConsts.MAP_WIDTH, gameConsts.BACKGROUND_SIZE):
                    self.screen.blit(self.bg, (x,y))

        done = False
        #Set length of episode
        if(self.time_steps == 20000):
            done = True
            self.episode = self.episode + 1
            self.change_rew_frac(self.episode)

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
                        #Small negative reward for every shot fired
                        #local_rewards[self.selectableTanks[t].tank_num-1] = local_rewards[self.selectableTanks[t].tank_num-1] - 1.0

                t = t+1

        #Provide increasing reward the longer the agent is alive
        '''
        if(self.selectableTanks[0].respawn == False):
            self.alive = self.alive + 1
            local_rewards[self.selectableTanks[0].tank_num-1] = local_rewards[self.selectableTanks[0].tank_num-1] + 0.0001 * self.alive
        else:
            self.alive = 0'''

        for ai in self.allAIPlayers:
            ai.control()


        if gameConsts.render:
            self.scoreboard.update()
        for obj1 in self.gameObjects:
            for obj2 in self.gameObjects:
                if obj1 is obj2:
                    continue

                #If two objects overlap
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
            if gameConsts.render:
                obj1.getSprite().draw(self.screen)


        if gameConsts.render:
            for t in self.allTanks:
                self.scoreboard.updateScore(self.team_1_score, self.team_2_score)

            pg.display.flip()
        self.obs = self.observation()

        #calculate the shaped rewards based on change in distance to flag/base
        shaped_rew = 0.0
        #shaping_rew = np.array([max(100*(1-self.stole_flag)*(self.prev_dist_to_flag - self.dist_to_flag) + 100*self.stole_flag*(self.prev_dist_to_base - self.dist_to_base), -10.0)])
        for t in range(len(self.BlueTanks)):
            stole_flag = isinstance(self.BlueTanks[t].flag, Flag)
            if self.BlueTanks[t].position[0] != gameConsts.players[0]["tanks"][t]['position']['x'] and self.BlueTanks[t].position[1] != gameConsts.players[0]["tanks"][t]['position']['y']:
                shaped_rew += max(100*(1-stole_flag)*(self.prev_dist_to_flag[t] - self.dist_to_flag[t]) + 100*stole_flag*(self.prev_dist_to_base[t] - self.dist_to_base[t]), -10.0)
            else:
                shaped_rew += 0.0

        #combine all rewards and send to model
        local_reward = sum(local_rewards)
        shaped_rew += local_reward

        all_rew = global_rewards + self.rew_frac*(shaped_rew)
        return [self.obs, all_rew, done, {"shaped":shaped_rew, "unshaped":global_rewards, "frac": self.rew_frac}]


    def change_rew_frac(self, episode):
        if (self.rew_frac <= 0.0):
            self.rew_frac = 0.0
            return

        global PHASE_AT_EPISODE
        if (episode != 0 and episode % PHASE_AT_EPISODE == 0):
            self.rew_frac -= 0.1





    def observation(self):
        obs = [0]*int(self.obs_size/4)
        i = 0
        vals = 3

        #Get observation. The location, angle of each tank. If they have a flag or not and if they fired and time since they fired. Location of flags.
        #Observations are normalized to the [0,1] range to help the model learn more easily
        for obj in self.allTanks:
            obs[i] = (obj.position[0] - self.obs_limits_low[i]) / (self.obs_limits_high[i] - self.obs_limits_low[i])
            i = i+1
            obs[i] = (obj.position[1] - self.obs_limits_low[i]) / (self.obs_limits_high[i] - self.obs_limits_low[i])
            i = i+1
            obs[i] = (obj.angle - self.obs_limits_low[i]) / (self.obs_limits_high[i] - self.obs_limits_low[i])
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


        '''for obj in self.allObstacles:
            obs[i] = (obj.position[0] - self.obs_limits_low[i]) / (self.obs_limits_high[i] - self.obs_limits_low[i])
            i = i+1
            obs[i] = (obj.position[1] - self.obs_limits_low[i]) / (self.obs_limits_high[i] - self.obs_limits_low[i])
            i = i+1
            obs[i] = (obj.size[0] - self.obs_limits_low[i]) / (self.obs_limits_high[i] - self.obs_limits_low[i])
            i = i+1

        for obj in self.allBases:
            obs[i] = (obj.position[0] - self.obs_limits_low[i]) / (self.obs_limits_high[i] - self.obs_limits_low[i])
            i = i+1
            obs[i] = (obj.position[1] - self.obs_limits_low[i]) / (self.obs_limits_high[i] - self.obs_limits_low[i])
            i = i+1
            obs[i] = (obj.size[0] - self.obs_limits_low[i]) / (self.obs_limits_high[i] - self.obs_limits_low[i])
            i = i+1'''

        for obj in self.allFlags:
            obs[i] = (obj.position[0] - self.obs_limits_low[i]) / (self.obs_limits_high[i] - self.obs_limits_low[i])
            i = i+1
            obs[i] = (obj.position[1] - self.obs_limits_low[i]) / (self.obs_limits_high[i] - self.obs_limits_low[i])
            i = i+1

        #update the distance and prev distance to the other tanks and flags
        for tank in self.allTanks:
            tank_num = tank.tank_num
            for obj in self.allFlags:
                if(obj.color == "red"):
                    self.prev_dist_to_flag[tank_num-1] = self.dist_to_flag[tank_num-1]
                    self.dist_to_flag[tank_num-1] = (self.get_dist(tank.position[0], tank.position[1], obj.position[0], obj.position[1]) - self.obs_limits_low[i]) / (self.obs_limits_high[i] - self.obs_limits_low[i])

            for obj in self.allBases:
                if(obj.color == "blue"):
                    self.prev_dist_to_base[tank_num-1] = self.dist_to_base[tank_num-1]
                    self.dist_to_base[tank_num-1] = (self.get_dist(tank.position[0], tank.position[1], obj.position[0], obj.position[1]) - self.obs_limits_low[i]) / (self.obs_limits_high[i] - self.obs_limits_low[i])


        #Concatenate new observation with last 3 observations
        ob = np.concatenate((np.array(self.obs[int(self.obs_size/4):]), np.array(obs)))

        return np.array(ob)


    def reward(self, local_reward, global_reward):
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
            if(o2.ghost):
                return [local_rewards, global_rewards]
            #Local Rewards
            if(o1.tank_num <= len(self.selectableTanks)):
                if(o1.color != o2.color):
                    if(isinstance(o2.flag, Flag)):
                        local_rewards[o1.tank_num-1] = local_rewards[o1.tank_num-1] + 20.0  #Reward for shooting down enemy carrying your flag
                    else:
                        local_rewards[o1.tank_num-1] = local_rewards[o1.tank_num-1] + 10.0  #Reward for shooting enemy without a flag
                else:
                    local_rewards[o1.tank_num-1] = local_rewards[o1.tank_num-1] - 20.0  #Reward for shooting teammate (friendly fire)
            else:
                if(o2.tank_num <= len(self.selectableTanks)):
                    if(isinstance(o2.flag, Flag)):
                        local_rewards[o2.tank_num-1] = local_rewards[o2.tank_num-1] - 20.0  #Reward for getting shot while carrying flag
                    else:
                        local_rewards[o2.tank_num-1] = local_rewards[o2.tank_num-1] - 10.0  #Reward for getting shot without flag


            #Team score that are displayed
            if(o1.tank_num <= len(self.selectableTanks)):
                if(o1.color != o2.color):
                    self.team_1_score += 10.0
                else:
                    self.team_1_score += -20.0
            else:
                if(o1.color != o2.color):
                    self.team_2_score += 10.0
                else:
                    self.team_2_score += -20.0


            if(o1 in self.allBullets):
                self.allBullets.remove(o1)
            o1.terminate()

            if isinstance(o2.flag, Flag):
                o2.flag.dropped()
                o2.setFlag(None)
            o2.setRespawn(self.time_steps)
        if(o1Tank and isinstance(o2, Flag) and (o1.color != o2.color)):
            #Reward for stealing flag or flag getting stolen
            if(isinstance(o1.flag, Flag) == False):
                if(self.selectableTanks[0].color == o2.color):
                    global_rewards = global_rewards - 50
                    self.team_2_score += 50
                else:
                    global_rewards = global_rewards + 50
                    self.team_1_score += 50

            o1.setFlag(o2)
            o2.setPickedUp(o1)
        if(o1Tank and isinstance(o2, Base) and isinstance(o1.flag, Flag) and o1.color == o2.color):
            if gameConsts.render:
                self.scoreboard.updateScore(o1.color, gameConsts.POINTS_RETURNING_FLAG)
            o1.flag.respawn()
            o1.flag.dropped()
            o1.setFlag(None)
            #Reward for bringing flag back to base or getting flag taken to enemy base
            if(self.selectableTanks[0].color == o1.color):
                global_rewards = global_rewards + 100
                self.team_1_score += 100
            else:
                global_rewards = global_rewards - 100
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


    #Function used to pre-train a model. Resembles the programmed AI but with randomness
    def control(self, obs):
        for t in self.selectableTanks:
            if(random.random() < self.gen_eps and isinstance(t.flag, Flag) == False):
                rand_act = self.action_space.sample()
                allow_attack = 0
                for e in self.nonselectableTanks:
                    if(self.get_dist(t.position[0], t.position[1], e.position[0], e.position[1]) < gameConsts.FIRE_ENEMY_RANGE):
                        allow_attack = 1
                        break
                if(allow_attack == 0):
                    rand_act[2] = 0
                return rand_act
            else:
                des_x = t.position[0]
                des_y = t.position[1]
                fire = 0
                if (t.respawn):
                    speed, ang_speed = self.control_update(t.position[0], t.position[1], des_x, des_y, t.direction, t.radius)
                    return np.array([speed, ang_speed, fire])
                '''if (isinstance(t.flag, Flag)):
                    des_x, des_y = self.allBases[0].position
                    speed, ang_speed = self.control_update(t.position[0], t.position[1], des_x, des_y, t.direction, t.radius)
                    return np.array([speed, ang_speed, fire])'''

                # find closest tank
                tankTarget = {'dist': math.inf, 'enemyTank': None}
                for e in self.nonselectableTanks:
                    dist = math.hypot(e.position[0] - t.position[0], e.position[1] - t.position[1])
                    if (not e.respawn and dist < tankTarget['dist']):
                        tankTarget['enemyTank'] = e
                        tankTarget['dist'] = dist

                #random.shuffle(self.allFlags)
                attackMode = True
                '''if(self.allFlags[1].pickedUpBy is None):
                    des_x, des_y = self.allFlags[1].position
                    attackMode = False'''


                if((tankTarget['enemyTank'] is not None and (dist < gameConsts.SIGHT_ENEMY_RANGE or attackMode))):
                    des_x, des_y = tankTarget['enemyTank'].position
                    if(tankTarget['dist'] < gameConsts.FIRE_ENEMY_RANGE):
                        fire = 1

                speed, ang_speed = self.control_update(t.position[0], t.position[1], des_x, des_y, t.direction, t.radius)
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
