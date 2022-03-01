import CTF
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from stable_baselines.gail import ExpertDataset
import tensorflow as tf

dataset = ExpertDataset(expert_path="CTF_expert.npz", traj_limitation=-1, batch_size = 2048)

env = CTF.CTF()#make_vec_env(CTF.CTF())
iterations = 400
i = 0
policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[64, 64])
model = PPO2(MlpPolicy, env, verbose=1, seed = 1, n_steps = 2048, tensorboard_log = "log", policy_kwargs=policy_kwargs)
print(model.learning_rate)
model.learning_rate = 0.0005
print(model.learning_rate)
model.pretrain(dataset, n_epochs = 5000)
model.save("ppo2_pretrained_attack")
