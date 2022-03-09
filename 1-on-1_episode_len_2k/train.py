import CTF
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2

env = CTF.CTF()#make_vec_env(CTF.CTF())
iterations = 400
i = 0
model = PPO2(MlpPolicy, env, verbose=1, n_steps = 2048, ent_coef = 0.01, lam = 0.94, gamma = 0.995, seed = 1, tensorboard_log = "log")
'''while(i < iterations):
	model.learn(total_timesteps = 20000)
	model.save("ppo2_CTF")
	i=i+1'''
model.learn(total_timesteps = 20000*iterations, log_interval = 500)
model.save("ppo2_CTF_1")
