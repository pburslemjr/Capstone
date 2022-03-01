import math
import numpy as np
import gym
from gym import spaces
import random

random.seed(1)
np.random.seed(1)

class Weight(gym.Env):

	def __init__(self):

		self.obs_size = 4*(3+7+(2*3)+(2*3)+(4*3))	#Tank, enemy tank, flag, base, obstacle
		self.episode=0

		self.action_space = spaces.Box(low=np.array([0, 0]) , high=np.array([1, 1]), dtype=np.float32)

		self.obs_limits_low = [0]*self.obs_size
		self.obs_limits_high = [1]*self.obs_size
		self.obs = [0]*self.obs_size
		self.ep_len = 20480

		self.time_steps = 0
		
		self.observation_space = spaces.Box(low=np.array(self.obs_limits_low), high=np.array(self.obs_limits_high), dtype=np.float32)


	def reset(self):
		self.rew = 0.0
		return self.obs


	def step(self, actions):
		
		self.time_steps = self.time_steps + 1
		
		done = False
		if(self.time_steps == self.ep_len):
			done = True
			self.episode = self.episode + 1
			self.time_steps = 0
		return [self.obs, self.rew, done, {}]
