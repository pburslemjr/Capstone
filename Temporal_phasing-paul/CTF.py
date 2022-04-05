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
from self_play_ppo2 import self_play_ppo2
from stable_baselines.common import make_vec_env

random.seed(1)
np.random.seed(1)

class CTF(gym.Env):

    def __init__(self):
        super(CTF, self).__init__()

        self.obs_size = 4*(3+7*3+(2*3)+(2*3)+(4*3)) #Tank, enemy tank, flag, base, obstacle
        self.episode=0

        #Set action bounds
        self.action_space = spaces.Box(low=np.array([-1, -1, -1]) , high=np.array([1, 1, 1]), dtype=np.float32)

        self.obs_limits_low = [0]*self.obs_size
        self.obs_limits_high = [1000]*self.obs_size
        self.obs = [0]*self.obs_size
        self.gen_eps = 0.75 #Used in the control function to randomly sample between expert and random action
        self.ep_len = 20480 #Length of one episode

        self.time_steps = 0
        self.team_1_score = 0.0
        self.team_2_score = 0.0

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

        #Setting the highest and lowest value for each feature in the observation in order to normalize the state space
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

        #Set Observation space limits
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
        self.BlueTanks = []
        self.RedTanks = []
        self.allAIPlayers = []

        self.allObstacles = []
        self.allBases = []
        self.allFlags = []
        self.allBullets = []

        #self.team_1_score = 0.0
        #self.team_2_score = 0.0

        self.alive_blue = 0
        self.alive_red = 0
        self.kill_interval = 0

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
                tankNum = tankNum + 1
                self.allTanks.append(tank)
                if p['human']:
                    self.BlueTanks.append(tank)
                if not p['human']:
                    self.allAIPlayers.append(AI(p['color'], self.gameObjects)) #Uncomment  to make all red bots controller the rule based AI
                    self.RedTanks.append(tank)

        #Add flags after tanks so they are on top
        for p in gameConsts.players:
            flag = Flag(p['color'], (p['base']['x'], p['base']['y']), gameConsts.FLAG_SIZE)
            self.gameObjects.append(flag)
            self.allFlags.append(flag)

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
                if (obj1.color == "red"):
                    obj1.update(1)
                else:
                    obj1.update(0)
            else:
                obj1.update()
            #obj1.getSprite().draw(self.screen)


        pg.display.flip()

        observations = []
        for obj in self.allTanks:
            observations.append(self.observation(obj))

        return observations


    def step(self, actions):

        self.time_steps = self.time_steps + 1

        local_rewards = [0]*len(self.allTanks)  #Used to set each tanks independent reward earned
        global_rewards_blue = 0.0   #Reward for all Blue tanks
        global_rewards_red = 0.0    #Reward for all Red tanks

        C = 10
        render_freq = 25
        '''if(self.time_steps % render_freq == 0):
            # screen.fill(gameConsts.BACKGROUND_COLOR) # TODO: replace with grass
            for x in range(0, gameConsts.MAP_WIDTH, gameConsts.BACKGROUND_SIZE):
                for y in range(0, gameConsts.MAP_WIDTH, gameConsts.BACKGROUND_SIZE):
                    self.screen.blit(self.bg, (x,y))'''

        done = False
        if(self.time_steps == self.ep_len):
            done = True
            self.episode = self.episode + 1
            self.time_steps = 0
            self.team_1_score = 0.0
            self.team_2_score = 0.0

        t = 0
        for i,a in enumerate(actions):
            if(i%3 == 0):
                a = (a - (-1)) / (1 - (-1))
                self.allTanks[t].speed = a * gameConsts.TANK_MAX_SPEED
            elif(i%3 == 1):
                self.allTanks[t].angleSpeed = a * gameConsts.TANK_MAX_ROTATION
            elif(i%3 == 2):
                a = (a - (-1)) / (1 - (-1))
                a = round(a)
                if(a == 1):
                    if(self.allTanks[t].fired == 0 and self.allTanks[t].respawn == False):
                        self.allBullets.append(self.allTanks[t].fire())
                        local_rewards[t] = -1.0

                t = t+1

        if(self.BlueTanks[0].respawn == False):
            self.alive_blue = self.alive_blue + 1
        else:
            self.alive_blue = 0
            #self.reset()

        if(self.RedTanks[0].respawn == False):
            self.alive_red = self.alive_red + 1
        else:
            self.alive_red = 0

        #Uncomment to get actions for rule-based AI tanks
        for ai in self.allAIPlayers:
            ai.control()

        self.scoreboard.update()
        for obj1 in self.gameObjects:
            for obj2 in self.gameObjects:
                if obj1 is obj2:
                    continue
                #Check for object collisions
                if(math.sqrt(pow(obj1.position[0] - obj2.position[0], 2) + pow(obj1.position[1] - obj2.position[1], 2)) < max(obj1.size[0], obj1.size[1])/2 + max(obj2.size[0], obj2.size[1])/2 + 10):

                    hit_rewards = self.handleHit(obj1, obj2)
                    for i,r in enumerate(hit_rewards[0]):
                        local_rewards[i] += r

                    global_rewards_blue += hit_rewards[1]
                    global_rewards_red += hit_rewards[2]

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
                if (obj1.color == "red"):
                    obj1.update(1)
                else:
                    obj1.update(0)

            else:
                obj1.update()
            #obj1.getSprite().draw(self.screen)


        for t in self.allTanks:
            self.scoreboard.updateScore(self.team_1_score, self.team_2_score)

        #pg.display.flip()
        observations = []
        for obj in self.allTanks:
            observations.append(self.observation(obj))

        if(self.time_steps % render_freq == 0):
            self.render()


        #rew = self.reward(local_rewards[:len(self.BlueTanks)], global_rewards_blue)
        #shaping_rew = np.array(0.0)

        return [observations, global_rewards_blue, done, {}]


    #Get the state of the environment from a tanks frame of reference
    def observation(self, tank):
        obs = [0]*int(self.obs_size/4)
        i = 0
        vals = 3

        obs[i] = (int(isinstance(tank.flag, Flag)) - self.obs_limits_low[i]) / (self.obs_limits_high[i] - self.obs_limits_low[i])
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

        ob = np.concatenate((np.array(self.obs[int(self.obs_size/4):]), np.array(obs)))

        return np.array(ob)


    def reward(self, local_reward, global_reward):
        r = []
        for i in range(len(local_reward)):
            r.append(local_reward[i] + global_reward)
        return r



    def handleHit(self, o1, o2):
        local_rewards = [0]*len(self.allTanks)
        global_rewards_blue = 0.0
        global_rewards_red = 0.0
        o1Tank = isinstance(o1, Tank)
        o2Tank = isinstance(o2, Tank)
        if(isinstance(o1, Bullet) and isinstance(o2, Obstacle)):
            if(o1 in self.allBullets):
                self.allBullets.remove(o1)
            o1.terminate()
        if(o1Tank and isinstance(o2, Obstacle) or o1Tank and o2Tank):
            if (o1.ghost and o2Tank or o2Tank and o2.ghost): # recently respawn tanks can drive through other tanks
                return [local_rewards, global_rewards_blue, global_rewards_red]
            xDiff = o2.position[0] - o1.position[0]
            yDiff = o2.position[1] - o1.position[1]
            if abs(abs(xDiff) - abs(yDiff)) < 0.0: # ignore exact corner collision. work around
                return [local_rewards, global_rewards_blue, global_rewards_red]
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
                return [local_rewards, global_rewards_blue, global_rewards_red]

            #Local Rewards
            '''if(o1.tank_num <= len(self.BlueTanks)):
                #Blue tank shoots red
                if(o1.color != o2.color):
                    if(isinstance(o2.flag, Flag)):
                        local_rewards[o1.tank_num-1] = local_rewards[o1.tank_num-1] + 20.0  #blue tank
                        local_rewards[o2.tank_num-1] = local_rewards[o2.tank_num-1] - 20.0  #red tank
                    else:
                        local_rewards[o1.tank_num-1] = local_rewards[o1.tank_num-1] + 20.0
                        local_rewards[o2.tank_num-1] = local_rewards[o2.tank_num-1] - 20.0
                #Blue tank shoots blue tank
                else:
                    local_rewards[o1.tank_num-1] = local_rewards[o1.tank_num-1] - 0.0
                    local_rewards[o2.tank_num-1] = local_rewards[o1.tank_num-1] + 0.0
            #Red  bullet hit
            else:
                #Red bullet hit blue tank
                if(o1.color != o2.color):
                    if(isinstance(o2.flag, Flag)):
                        local_rewards[o1.tank_num-1] = local_rewards[o1.tank_num-1] + 20.0
                        local_rewards[o2.tank_num-1] = local_rewards[o2.tank_num-1] - 20.0
                    else:
                        local_rewards[o1.tank_num-1] = local_rewards[o1.tank_num-1] + 20.0
                        local_rewards[o2.tank_num-1] = local_rewards[o2.tank_num-1] - 20.0
                    self.kill_interval = 0
                else:
                    local_rewards[o1.tank_num-1] = local_rewards[o1.tank_num-1] - 20.0
                    local_rewards[o2.tank_num-1] = local_rewards[o2.tank_num-1] - 0.0


            if(o1.tank_num <= len(self.BlueTanks)):
                if(o1.color != o2.color):
                    self.team_1_score += 10.0
                    self.team_2_score -= 10.0
                else:
                    self.team_1_score += 0.0
                    self.team_2_score -= 0.0
            else:
                if(o1.color != o2.color):
                    self.team_1_score -= 10.0
                    self.team_2_score += 10.0
                else:
                    self.team_1_score -= 0.0
                    self.team_2_score += -20.0'''


            if(o1 in self.allBullets):
                self.allBullets.remove(o1)
            o1.terminate()

            if isinstance(o2.flag, Flag):
                o2.flag.dropped()
                o2.setFlag(None)
            o2.setRespawn(self.time_steps)

        if(o1Tank and isinstance(o2, Flag) and (o1.color != o2.color)):
            o1.setFlag(o2)
            o2.setPickedUp(o1)
        if(o1Tank and isinstance(o2, Base) and isinstance(o1.flag, Flag) and o1.color == o2.color):
            self.scoreboard.updateScore(o1.color, gameConsts.POINTS_RETURNING_FLAG)
            o1.flag.respawn()
            o1.flag.dropped()
            o1.setFlag(None)
            #Rewards for stealing or losing flag
            if(self.BlueTanks[0].color == o1.color):
                global_rewards_blue = global_rewards_blue + 100
                global_rewards_red = global_rewards_red - 100
                self.team_1_score += 100
            else:
                global_rewards_red = global_rewards_red + 100
                global_rewards_blue = global_rewards_blue - 100
                self.team_2_score += 100
        return [local_rewards, global_rewards_blue, global_rewards_red]



    def checkWalls(self, obj):
        if(obj.top <= 0):
            obj.preventMovement('up')
        elif(obj.bottom >= gameConsts.MAP_HEIGHT):
            obj.preventMovement('down')
        if(obj.right >= gameConsts.MAP_WIDTH):
            obj.preventMovement('right')
        elif(obj.left <= 0):
            obj.preventMovement('left')


    def control_blue(self, obs):
        actions = np.zeros(shape=(2,3))

        for t in self.BlueTanks:
            des_x = t.position[0]
            des_y = t.position[1]
            fire = 0
            if t.respawn:
                speed, ang_speed = self.control_update(t.position[0], t.position[1], des_x, des_y, t.direction, t.radius)
                actions[t.tank_num - 1][0] = speed
                actions[t.tank_num - 1][1] = ang_speed
                actions[t.tank_num - 1][2] = fire

                continue
            #if(random.random() < self.gen_eps and isinstance(t.flag, Flag) == False):
            if (isinstance(t.flag, Flag)):
                des_x, des_y = self.allBases[0].position
                speed, ang_speed = self.control_update(t.position[0], t.position[1], des_x, des_y, t.direction, t.radius)
                actions[t.tank_num - 1][0] = speed
                actions[t.tank_num - 1][1] = ang_speed
                actions[t.tank_num - 1][2] = fire
                continue

                # find closest tank
            tankTarget = {'dist': math.inf, 'enemyTank': None}
            for e in self.RedTanks:
                dist = math.hypot(e.position[0] - t.position[0], e.position[1] - t.position[1])
                if (not e.respawn and dist < tankTarget['dist']):
                    tankTarget['enemyTank'] = e
                    tankTarget['dist'] = dist

            #random.shuffle(self.allFlags)
            attackMode = True
            if(self.allFlags[1].pickedUpBy is None):
                des_x, des_y = self.allFlags[1].position
                attackMode = False


            if((tankTarget['enemyTank'] is not None and (tankTarget['dist'] < gameConsts.SIGHT_ENEMY_RANGE or attackMode))):
                des_x, des_y = tankTarget['enemyTank'].position
                if(tankTarget['dist'] < gameConsts.FIRE_ENEMY_RANGE):
                    fire = 1

            speed, ang_speed = self.control_update(t.position[0], t.position[1], des_x, des_y, t.direction, t.radius)
            actions[t.tank_num - 1][0] = speed
            actions[t.tank_num - 1][1] = ang_speed
            actions[t.tank_num - 1][2] = fire
        return actions


    def control_red(self, obs):
        for t in self.BlueTanks:
            if(random.random() < self.gen_eps and isinstance(t.flag, Flag) == False):
                rand_act = self.action_space.sample()
                allow_attack = 0
                for e in self.RedTanks:
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
                for e in self.RedTanks:
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
