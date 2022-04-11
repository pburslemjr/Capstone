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
import tensorflow as tf
from customPPO2 import CustomPPO2
import argparse
import re

parser = argparse.ArgumentParser()

parser.add_argument('--load-model', help='Learnt model', default='',
                    type=str, required=False)
parser.add_argument('-n', '--n-episodes', help='Overwrite the number of episodes', default=500,
                    type=int)

args = parser.parse_args()

#random.seed(1)
np.random.seed(1)

last_saved = 0
episode_rewards = [0.0]
episode_shaped_rewards = [0.0]
steps = 0


def create_callback(model, verbose=1):
    """
    Create callback function for saving best model frequently.

    :param algo: (str)
    :param save_path: (str)
    :param verbose: (int)
    :return: (function) the callback function
    """

    def sac_callback(_locals, _globals):
        """
        Callback for saving best model when using SAC.

        :param _locals: (dict)
        :param _globals: (dict)
        :return: (bool) If False: stop training
        """

        global last_saved
        global episode_rewards
        global episode_shaped_rewards
        global steps
        #print(_locals)

        steps = steps + 1

        episode_rewards[-1] = episode_rewards[-1] + _locals['rewards']
        if(steps%20000 == 0):
            print("Episodes Reward = ", episode_rewards[-1], len(episode_rewards))
            print("Policy prob: " + str(model.policy_prob))
            episode_rewards.append(0.0)

        file_num = re.findall("\d*", args.load_model[-2:])
        file_num = [x for x in file_num if x != '']
        if(file_num != []):
            file_num = int(file_num[0])+1
        else:
            file_num = 0
        if(len(episode_rewards) == last_saved + 10):
            print("Saving model", file_num)
            #_locals['self'].save("ppo2_CTF_1")
            model.save("CTF_"+str(int(steps/2000000)+file_num))
            last_saved = len(episode_rewards)
        return True
    return sac_callback


env = CTF.CTF()#make_vec_env(CTF.CTF())
iterations = args.n_episodes

#Accel only for 1,200,000 iteration

if(args.load_model != ""):
    model = CustomPPO2.load(args.load_model, env = make_vec_env(lambda:env))
    print(args.load_model)
else:
    policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[512, 512])
    model = CustomPPO2(MlpPolicy, env, verbose=1, seed = 1, n_steps = 20480, tensorboard_log = "log", policy_kwargs=policy_kwargs)
    model.save("CTF_rand")
    print("Train from scratch")

#model = PPO2(MlpPolicy, env, verbose=1, seed = 1, n_steps = 2048, tensorboard_log = "log")
#model.pretrain(dataset, n_epochs = 5000)
#model.save("ppo2_pretrained_CTF_2")
model.gamma = 0.99
model.lam = 0.95
model.nminibatches = 10
model.noptepochs = 3
model.ent_coef = 0.005 #(beta)
model.learning_rate = 0.0003#3
model.cliprange = 0.2 #(epislon)
#model.seed = 0

kwargs = {}
kwargs.update({'callback': create_callback(model, verbose=1)})

print(model)
model.learn(total_cycles = 1000, iteration = 0, rl_optimization = 20480*100, **kwargs)
model.save("CTF_4")
