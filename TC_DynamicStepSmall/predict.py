import CTF
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from stable_baselines import results_plotter
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.noise import AdaptiveParamNoiseSpec
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.gail import ExpertDataset
import os
import numpy as np
import random

from customPPO2 import CustomPPO2

#random.seed(1)
np.random.seed(1)

'''env = CTF.CTF()#make_vec_env(CTF.CTF())
iterations = 200
model = PPO2.load("ppo2_pretrained_CTF")
model.set_env(make_vec_env(lambda:env))
eps = 0.3
i = 0
rn_nums = []
for j in range(iterations):
	rn_nums.append(random.random())

while(i < iterations):
	rn = rn_nums[i]
	print(rn)

	if(rn < eps):
		dataset = ExpertDataset(expert_path="CTF_expert.npz", traj_limitation=3, batch_size = 2048)
		model.pretrain(dataset, n_epochs = 10)
		model.save("tmp_CTF")
		del model
		model = PPO2.load("tmp_CTF")
		model.set_env(make_vec_env(lambda:env))
	else:
		model.learn(total_timesteps = 20000)
	print(i)
	i = i+1

model.save("guided_ppo2_CTF_1")'''



dataset = ExpertDataset(expert_path="CTF_expert.npz", traj_limitation=-1, batch_size = 2048)

env = CTF.CTF()#make_vec_env(CTF.CTF())
iterations = 50

model = CustomPPO2.load("fire_4")
model.set_env(make_vec_env(lambda:env))
#model = PPO2(MlpPolicy, env, verbose=1, seed = 1, n_steps = 2048, tensorboard_log = "log")
#model.pretrain(dataset, n_epochs = 5000)
#model.save("ppo2_pretrained_CTF_2")
obs = env.reset()
while True:
	a, _ = model.predict(obs, deterministic = False)
	obs, r, done, _ = env.step(a)
