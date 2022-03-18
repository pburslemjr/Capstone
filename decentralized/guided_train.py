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
from self_play_ppo2 import self_play_ppo2
from threading import Thread
from queue import Queue
import gym
import argparse
import re
import os
import time

os.environ["PYTHONHASHSEED"] = str(1)

#Parse command line parameters
parser = argparse.ArgumentParser()
parser.add_argument('--blue-model', help='Model for blue agent', default='',
                    type=str, required=False)
parser.add_argument('--red-model', help='Model for red agent', default='',
                    type=str, required=False)
parser.add_argument('-n', '--n-episodes', help='Overwrite the number of episodes', default=1000,
                    type=int)
parser.add_argument('-switch-freq', help='After how many steps does training shift from one agent to the other', default=2500000,
                    type=int)

args = parser.parse_args()



#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

random.seed(1)
np.random.seed(1)

last_saved = 0
episode_rewards = [0.0]
steps = 0

last_saved_2 = 0
episode_rewards_2 = [0.0]
steps_2 = 0
prev_time = time.clock()
current_time = time.clock() - prev_time

def run_learner(conn, total_timesteps, iteration, model_num):


    policy_kwargs = dict(net_arch=[512, 512])
    model = self_play_ppo2(MlpPolicy, conn, verbose=1, seed = 1, n_steps = 20480, model_num = 1, tensorboard_log = "log", policy_kwargs=policy_kwargs)

    model.gamma = 0.99
    model.lam = 0.95
    model.nminibatches = 10
    model.noptepochs = 3
    model.ent_coef = 0.005 #(beta)
    model.learning_rate = 0.0003
    model.cliprange = 0.2 #(epislon)

    model.learn(total_timesteps, iteration)

#Callback function for Model-1
def create_callback_1(model, verbose=1):
    """
    Create callback function for saving best model frequently.
    The model is saved every 10 episodes
    Every 250000 episodes the model is saved under a different name

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
        global steps
        global prev_time
        global current_time


        steps = steps + 1

        episode_rewards[-1] = episode_rewards[-1] + _locals['rewards']
        #When an episode is over, save episode_reward
        if(steps%20480 == 0):
            print("Model: ", _locals["model_num"], "total steps: ", steps, "\nEpisodes Reward = ", episode_rewards[-1], len(episode_rewards))
            current_time = time.clock() - prev_time
            print("Episode took ", current_time, " seconds")
            prev_time = time.clock()
            episode_rewards.append(0.0)

        file_num = re.findall("\d*", args.blue_model[-3:])
        file_num = [x for x in file_num if x != '']
        #Based on the model that was loaded, the number under which the next model is saved is determined
        if(file_num != []):
            file_num = int(file_num[0])+1
        else:
            file_num = 0

        #Save a model every 10 episodes
        if(len(episode_rewards) == last_saved + 10):
            print(_locals["model_num"], "Saving model")
            #_locals['self'].save("ppo2_CTF_1")
            if((int(steps/2000000)+file_num) < 10):
                model.save("Model_"+str(_locals["model_num"])+"_0"+str(int(steps/2000000)+file_num))    #Save every 2,000,000 steps of the model under the same model name, to prevent the creation of too many model files.
            else:
                model.save("Model_"+str(_locals["model_num"])+"_"+str(int(steps/2000000)+file_num))
            last_saved = len(episode_rewards)
        return True
    return sac_callback


#Callback function for Model-2
def create_callback_2(model, verbose=1):
    """
    Create callback function for saving best model frequently.
    The model is saved every 10 episodes
    Every 250000 episodes the model is saved under a different name

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

        global last_saved_2
        global episode_rewards_2
        global steps_2
        global prev_time
        global current_time

        steps_2 = steps_2 + 1

        episode_rewards_2[-1] = episode_rewards_2[-1] + _locals['rewards']
        if(steps_2%20480 == 0):
            print("Model: ", _locals["model_num"], "total steps: ", steps_2, "\nEpisodes Reward = ", episode_rewards_2[-1], len(episode_rewards))
            current_time = time.clock() - prev_time
            print("Episode took ", current_time, " seconds")
            prev_time = time.clock()
            episode_rewards_2.append(0.0)

        file_num = re.findall("\d*", args.red_model[-3:])
        file_num = [x for x in file_num if x != '']
        if(file_num != []):
            file_num = int(file_num[0])+1
        else:
            file_num = 0

        if(len(episode_rewards_2) == last_saved_2 + 10):
            print(_locals["model_num"], "Saving model")
            #_locals['self'].save("ppo2_CTF_1")
            if((int(steps_2/2000000)+file_num) < 10):
                model.save("Model_"+str(_locals["model_num"])+"_0"+str(int(steps_2/2000000)+file_num))
            else:
                model.save("Model_"+str(_locals["model_num"])+"_"+str(int(steps_2/2000000)+file_num))
            last_saved_2 = len(episode_rewards_2)
        return True
    return sac_callback


#dataset = ExpertDataset(expert_path="CTF_expert.npz", traj_limitation=-1, batch_size = 2048)

if __name__ == '__main__':
    env = CTF.CTF()
    iterations = args.n_episodes    #Number of episodes the model will be trained
    training_agents = 2

    conn_manager = []
    for i in range(0, training_agents):
        conn_manager.append([Queue(), Queue()])

    #Load Model-1 and set hyperparameters
    if(args.blue_model != ""):
        model_1 = self_play_ppo2.load(args.blue_model, env = make_vec_env(lambda:env))
        print(args.blue_model)
    else:
        policy_kwargs = dict(net_arch=[512, 512])
        model_1 = self_play_ppo2(MlpPolicy, env, verbose=1, seed = 1, n_steps = 20480, tensorboard_log = "log", policy_kwargs=policy_kwargs)
        print("Train from scratch model 1")

    model_1.n_steps = 20480
    model_1.gamma = 0.99
    model_1.lam = 0.98
    model_1.nminibatches = 10
    model_1.noptepochs = 3
    model_1.ent_coef = 0.005 #(beta)
    model_1.learning_rate = 0.0003
    model_1.cliprange = 0.2 #(epislon)

    #Load Model-2 and set hyperparameters
    if(args.red_model != ""):
        model_2 = self_play_ppo2.load(args.red_model, env = make_vec_env(lambda:env))
        print(args.red_model)
    else:
        policy_kwargs = dict(net_arch=[512, 512])
        model_2 = self_play_ppo2(MlpPolicy, env, verbose=1, seed = 1, n_steps = 20480, tensorboard_log = "log", policy_kwargs=policy_kwargs)
        print("Train from scratch model 2")

    model_2.n_steps = 20480
    model_2.gamma = 0.99
    model_2.lam = 0.98
    model_2.nminibatches = 10
    model_2.noptepochs = 3
    model_2.ent_coef = 0.005 #(beta)
    model_2.learning_rate = 0.0003
    model_2.cliprange = 0.2 #(epislon)


    kwargs_1 = {}
    kwargs_1.update({'callback': create_callback_1(model_1, verbose=1)})

    kwargs_2 = {}
    kwargs_2.update({'callback': create_callback_2(model_2, verbose=1)})

    processes = []

    obs = np.zeros((1,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)
    observations = env.reset()  #Get current observations for each of the tanks
    obs[:] = observations[0]
    observations[0] = obs
    obs[:] = observations[1]
    observations[1] = obs
    obs = observations

    #Start learning for each agent
    for i in range(0, training_agents):
        if(i == 0):
            print("Starting model" + str(i))
            processes.append(Thread(target = model_1.learn, args =(20000*iterations+250000, 0, i+1, conn_manager[0], args.switch_freq, kwargs_1["callback"])))
        else:
            print("Starting model" + str(i))
            processes.append(Thread(target = model_2.learn, args =(20000*iterations+250000, 0, i+1, conn_manager[1], args.switch_freq, kwargs_2["callback"])))
        processes[i].start()

    #Send current observations to agents
    for i in range(training_agents):
        conn_manager[i][0].put(obs[i])

    for i in range(0, training_agents):
        conn_manager[i][0].join()

    #For each step
    for j in range(20000*iterations):
        #Get the actions taken by each agent
        actions = []
        for i in range(training_agents):
            action = conn_manager[i][1].get()
            clipped_actions = np.clip(action, env.action_space.low, env.action_space.high)
            for a in clipped_actions[0]:
                actions.append(a)

        #Execute action in the environment
        obs, rewards, dones, infos = env.step(actions)

        #rewards = [rew, shaped_rew, justglobal]
        if(dones):
            obs = env.reset()

        for i in range(training_agents):
            conn_manager[i][1].task_done()

        #Create and send message to each learning agent with their respective new observations and rewards
        for i in range(training_agents):
            conn_manager[i][0].put([obs[i], [rewards[0][i], rewards[1][i], rewards[2]], dones, infos, actions[(i*3) : (i*3+3)]])

        for i in range(0, training_agents):
            conn_manager[i][0].join()

    for i in range(0, training_agents):
        conn_manager[i][0].join()
        processes[i].join()



