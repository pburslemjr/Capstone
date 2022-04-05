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
policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[512, 512])
model = PPO2(MlpPolicy, env, verbose=1, seed = 1, n_steps = 20480, tensorboard_log = "log", policy_kwargs=policy_kwargs)

model.gamma = 0.99
model.lam = 0.95
model.nminibatches = 10
model.noptepochs = 3
model.ent_coef = 0.005 #(beta)
model.learning_rate = 0.0003#3
model.cliprange = 0.2 #(epislon)

model.pretrain(dataset, n_epochs = 3000)
model.save("pretrained_rl")
