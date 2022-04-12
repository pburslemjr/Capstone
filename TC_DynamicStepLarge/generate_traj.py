import CTF
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from stable_baselines.gail import generate_expert_traj

env = CTF.CTF()#make_vec_env(CTF.CTF())
obs = env.reset()
generate_expert_traj(env.control, "CTF_adv", env, n_episodes=30)
