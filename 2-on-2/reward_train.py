from stable_baselines import PPO2
from stable_baselines.common.schedules import get_schedule_fn
from stable_baselines.common.tf_util import total_episode_reward_logger, get_trainable_vars
from stable_baselines.common.math_util import safe_mean

import time
import gym
import numpy as np
import tensorflow as tf
import random
from stable_baselines import logger
from stable_baselines.common import explained_variance, ActorCriticRLModel, tf_util, SetVerbosity, TensorboardWriter
from stable_baselines.common.policies import ActorCriticPolicy, RecurrentActorCriticPolicy
import scipy
from stable_baselines.common.runners import AbstractEnvRunner
from typing import Union, Optional, Any
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common.policies import MlpPolicy
from tensorflow import keras
from stable_baselines.gail import ExpertDataset
from stable_baselines.gail import generate_expert_traj
import re
from os import walk

np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})


class CustomPPO2(PPO2):
        def _make_runner(self):
            return Runner(env=self.env, model=self, n_steps=self.n_steps,
                gamma=self.gamma, lam=self.lam)

        def __init__(self, policy, env, gamma=0.99, n_steps=128, ent_coef=0.01, learning_rate=2.5e-4, vf_coef=0.5,
                 max_grad_norm=0.5, lam=0.95, nminibatches=4, noptepochs=4, cliprange=0.2, cliprange_vf=None,
                 verbose=0, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None,
                 full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None):

            self.learning_rate = learning_rate
            self.cliprange = cliprange
            self.cliprange_vf = cliprange_vf
            self.n_steps = n_steps
            self.ent_coef = ent_coef
            self.vf_coef = vf_coef
            self.max_grad_norm = max_grad_norm
            self.gamma = gamma
            self.lam = lam
            self.nminibatches = nminibatches
            self.noptepochs = noptepochs
            self.tensorboard_log = tensorboard_log
            self.full_tensorboard_log = full_tensorboard_log

            self.action_ph = None
            self.advs_ph = None
            self.rewards_ph = None
            self.old_neglog_pac_ph = None
            self.old_vpred_ph = None
            self.learning_rate_ph = None
            self.clip_range_ph = None
            self.entropy = None
            self.vf_loss = None
            self.pg_loss = None
            self.approxkl = None
            self.clipfrac = None
            self._train = None
            self.loss_names = None
            self.train_model = None
            self.act_model = None
            self.value = None
            self.n_batch = None
            self.summary = None

            super().__init__(policy, env, gamma, n_steps, ent_coef, learning_rate, vf_coef,
                 max_grad_norm, lam, nminibatches, noptepochs, cliprange, cliprange_vf,
                 verbose, tensorboard_log, _init_setup_model, policy_kwargs,
                 full_tensorboard_log, seed, n_cpu_tf_sess)

            norm = tf.keras.constraints.MinMaxNorm(min_value=-1.0, max_value=1.0)

            self.reward_model = keras.Sequential(
                                   [
                                       keras.layers.Dense(512, name="layer1", kernel_regularizer = keras.regularizers.l1(1e-4), kernel_constraint=norm, use_bias=False, input_shape=[self.env.observation_space.shape[0]+self.env.action_space.shape[0]]),
                                       keras.layers.Dense(512, name="layer2", kernel_regularizer = keras.regularizers.l1(1e-4), kernel_constraint=norm, use_bias=False),
                                       keras.layers.Dense(1, name="layer3", activation="sigmoid", kernel_regularizer = keras.regularizers.l1(1e-2), kernel_constraint=norm, use_bias=False),
                                   ]
                                )
            #self.reward_model = keras.models.load_model("Rew_Model_1")#, custom_objects={ 'loss_fn': self.loss_fn })
            self.loading_model = 1
            
            print(self.reward_model.get_weights())

            self.reward_model.summary()
            self.reward_model.save("Rew_Model")

            if _init_setup_model:
                self.setup_model()

        def setup_model(self):
            with SetVerbosity(self.verbose):
                assert issubclass(self.policy, ActorCriticPolicy), "Error: the input policy for the PPO2 model must be " \
                                                               "an instance of common.policies.ActorCriticPolicy."

                self.n_batch = self.n_envs * self.n_steps

                self.graph = tf.Graph()
                with self.graph.as_default():
                    self.set_random_seed(self.seed)
                    self.sess = tf_util.make_session(num_cpu=self.n_cpu_tf_sess, graph=self.graph)

                    n_batch_step = None
                    n_batch_train = None
                    if issubclass(self.policy, RecurrentActorCriticPolicy):
                        assert self.n_envs % self.nminibatches == 0, "For recurrent policies, "\
                            "the number of environments run in parallel should be a multiple of nminibatches."
                        n_batch_step = self.n_envs
                        n_batch_train = self.n_batch // self.nminibatches

                    act_model = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1,
                                            n_batch_step, reuse=False, **self.policy_kwargs)
                    with tf.variable_scope("train_model", reuse=True,
                                           custom_getter=tf_util.outer_scope_getter("train_model")):
                        train_model = self.policy(self.sess, self.observation_space, self.action_space,
                                                  self.n_envs // self.nminibatches, self.n_steps, n_batch_train,
                                                  reuse=True, **self.policy_kwargs)

                    with tf.variable_scope("loss", reuse=False):
                        self.action_ph = train_model.pdtype.sample_placeholder([None], name="action_ph")
                        self.advs_ph = tf.placeholder(tf.float32, [None], name="advs_ph")
                        self.rewards_ph = tf.placeholder(tf.float32, [None], name="rewards_ph")
                        self.old_neglog_pac_ph = tf.placeholder(tf.float32, [None], name="old_neglog_pac_ph")
                        self.old_vpred_ph = tf.placeholder(tf.float32, [None], name="old_vpred_ph")
                        self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")
                        self.clip_range_ph = tf.placeholder(tf.float32, [], name="clip_range_ph")

                        neglogpac = train_model.proba_distribution.neglogp(self.action_ph)
                        self.entropy = tf.reduce_mean(train_model.proba_distribution.entropy())
    
                        vpred = train_model.value_flat

                        # Value function clipping: not present in the original PPO
                        if self.cliprange_vf is None:
                            # Default behavior (legacy from OpenAI baselines):
                            # use the same clipping as for the policy
                            self.clip_range_vf_ph = self.clip_range_ph
                            self.cliprange_vf = self.cliprange
                        elif isinstance(self.cliprange_vf, (float, int)) and self.cliprange_vf < 0:
                            # Original PPO implementation: no value function clipping
                            self.clip_range_vf_ph = None
                        else:
                            # Last possible behavior: clipping range
                            # specific to the value function
                            self.clip_range_vf_ph = tf.placeholder(tf.float32, [], name="clip_range_vf_ph")

                        if self.clip_range_vf_ph is None:
                            # No clipping
                            vpred_clipped = train_model.value_flat
                        else:
                            # Clip the different between old and new value
                            # NOTE: this depends on the reward scaling
                            vpred_clipped = self.old_vpred_ph + \
                                tf.clip_by_value(train_model.value_flat - self.old_vpred_ph,
                                                 - self.clip_range_vf_ph, self.clip_range_vf_ph)

                        vf_losses1 = tf.square(vpred - self.rewards_ph)
                        vf_losses2 = tf.square(vpred_clipped - self.rewards_ph)
                        self.vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

                        ratio = tf.exp(self.old_neglog_pac_ph - neglogpac)
                        pg_losses = -self.advs_ph * ratio
                        pg_losses2 = -self.advs_ph * tf.clip_by_value(ratio, 1.0 - self.clip_range_ph, 1.0 +
                                                                      self.clip_range_ph)

                        self.pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
                        self.approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - self.old_neglog_pac_ph))
                        self.clipfrac = tf.reduce_mean(tf.cast(tf.greater(tf.abs(ratio - 1.0),
                                                                          self.clip_range_ph), tf.float32))
                                                                          
                                                  
                        loss = self.pg_loss - self.entropy * self.ent_coef + self.vf_loss * self.vf_coef

                        tf.summary.scalar('entropy_loss', self.entropy)
                        tf.summary.scalar('policy_gradient_loss', self.pg_loss)
                        tf.summary.scalar('value_function_loss', self.vf_loss)
                        tf.summary.scalar('approximate_kullback-leibler', self.approxkl)
                        tf.summary.scalar('clip_factor', self.clipfrac)
                        tf.summary.scalar('loss', loss)

                        with tf.variable_scope('model'):
                            self.params = tf.trainable_variables()
                            for var in range(len(self.params)):
                                    tf.summary.histogram(self.params[var].name, self.params[var])
                                    if("model/pi/w" in self.params[var].name):
                                        self.weights = self.params[var]               

                            if self.full_tensorboard_log:
                                for var in self.params:
                                    tf.summary.histogram(var.name, var)

                        grads = tf.gradients(loss, self.params)
                        if self.max_grad_norm is not None:
                            grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
                        grads = list(zip(grads, self.params))
                    trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph, epsilon=1e-5)
                    self._train = trainer.apply_gradients(grads)

                    self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

                    with tf.variable_scope("input_info", reuse=False):
                        tf.summary.scalar('discounted_rewards', tf.reduce_mean(self.rewards_ph))
                        tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate_ph))
                        tf.summary.scalar('advantage', tf.reduce_mean(self.advs_ph))
                        tf.summary.scalar('clip_range', tf.reduce_mean(self.clip_range_ph))
                        if self.clip_range_vf_ph is not None:
                            tf.summary.scalar('clip_range_vf', tf.reduce_mean(self.clip_range_vf_ph))

                        tf.summary.scalar('old_neglog_action_probability', tf.reduce_mean(self.old_neglog_pac_ph))
                        tf.summary.scalar('old_value_pred', tf.reduce_mean(self.old_vpred_ph))

                        if self.full_tensorboard_log:
                            tf.summary.histogram('discounted_rewards', self.rewards_ph)
                            tf.summary.histogram('learning_rate', self.learning_rate_ph)
                            tf.summary.histogram('advantage', self.advs_ph)
                            tf.summary.histogram('clip_range', self.clip_range_ph)
                            tf.summary.histogram('old_neglog_action_probability', self.old_neglog_pac_ph)
                            tf.summary.histogram('old_value_pred', self.old_vpred_ph)
                            if tf_util.is_image(self.observation_space):
                                tf.summary.image('observation', train_model.obs_ph)
                            else:
                                tf.summary.histogram('observation', train_model.obs_ph)

                    self.train_model = train_model
                    self.act_model = act_model
                    self.step = act_model.step
                    self.proba_step = act_model.proba_step
                    self.value = act_model.value
                    self.initial_state = act_model.initial_state
                    tf.global_variables_initializer().run(session=self.sess)  # pylint: disable=E1101

                    self.summary = tf.summary.merge_all()
        
        def _train_step(self, learning_rate, cliprange, obs, returns, masks, actions, values, neglogpacs, update,
                        writer, states=None, cliprange_vf=None):
            """
            Training of PPO2 Algorithm
    
            :param learning_rate: (float) learning rate
            :param cliprange: (float) Clipping factor
            :param obs: (np.ndarray) The current observation of the environment
            :param returns: (np.ndarray) the rewards
            :param masks: (np.ndarray) The last masks for done episodes (used in recurent policies)
            :param actions: (np.ndarray) the actions
            :param values: (np.ndarray) the values
            :param neglogpacs: (np.ndarray) Negative Log-likelihood probability of Actions
            :param update: (int) the current step iteration    
            :param writer: (TensorFlow Summary.writer) the writer for tensorboard
            :param states: (np.ndarray) For recurrent policies, the internal state of the recurrent model
            :return: policy gradient loss, value function loss, policy entropy,
                    approximation of kl divergence, updated clipping range, training update operation
            :param cliprange_vf: (float) Clipping factor for the value function
            """
            advs = returns - values
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            td_map = {self.train_model.obs_ph: obs, self.action_ph: actions,
                      self.advs_ph: advs, self.rewards_ph: returns,
                      self.learning_rate_ph: learning_rate, self.clip_range_ph: cliprange,
                      self.old_neglog_pac_ph: neglogpacs, self.old_vpred_ph: values}
            if states is not None:
                td_map[self.train_model.states_ph] = states
                td_map[self.train_model.dones_ph] = masks

            if cliprange_vf is not None and cliprange_vf >= 0:
                td_map[self.clip_range_vf_ph] = cliprange_vf

            if states is None:
                update_fac = max(self.n_batch // self.nminibatches // self.noptepochs, 1)
            else:
                update_fac = max(self.n_batch // self.nminibatches // self.noptepochs // self.n_steps, 1)

            if writer is not None:
                # run loss backprop with summary, but once every 10 runs save the metadata (memory, compute time, ...)
                if self.full_tensorboard_log and (1 + update) % 10 == 0:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    summary, policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ = self.sess.run(
                        [self.summary, self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self._train],
                        td_map, options=run_options, run_metadata=run_metadata)
                    writer.add_run_metadata(run_metadata, 'step%d' % (update * update_fac))
                else:
                    summary, policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ = self.sess.run(
                        [self.summary, self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self._train],
                        td_map)
                writer.add_summary(summary, (update * update_fac))
            else:
                policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ = self.sess.run(
                    [self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self._train], td_map)
    
            return policy_loss, value_loss, policy_entropy, approxkl, clipfrac


        def learn(self, total_cycles, iteration, rl_optimization, callback=None, log_interval=1, tb_log_name="PPO2",
              reset_num_timesteps=True):
                # Transform to callable if needed
                self.learning_rate = get_schedule_fn(self.learning_rate)
                self.cliprange = get_schedule_fn(self.cliprange)
                cliprange_vf = get_schedule_fn(self.cliprange_vf)

                new_tb_log = self._init_num_timesteps(reset_num_timesteps)
                callback = self._init_callback(callback)

                if(self.loading_model == 0):
                    generate_expert_traj(self, "RL_traj_rand", self.env, n_episodes=30)

                file_list = ["CTF_expert.npz"]
                data_all = [np.load(fname, allow_pickle=True) for fname in file_list]
                        
                expert_data = {}
                for i, data in enumerate(data_all):
                    action_data = data["actions"]
                    for k, v in data.items():
                        if(k == "obs"):
                            observations = []
                            for j in range(len(v)):
                                observations.append(np.concatenate((v[j], action_data[j]), axis=0))
                            expert_data.update({k: observations})
                            continue
                                   
                        if(k == "actions"):
                            actions = []
                            for j in range(len(v)):
                                actions.append(1.0)
                            expert_data.update({k: actions})
                        else:
                            expert_data.update({k: v})

                
                with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                        as writer:
                    self._setup_learn()

                    t_first_start = time.time()
                    n_updates = rl_optimization// self.n_batch
                    
                    callback.on_training_start(locals(), globals())

                    #Uncomment to initialize weights
                    '''init = np.ones((512, 3), dtype=float)
                    self.sess.run(tf.assign(self.weights, init))'''
                    for cyc in range(total_cycles):
                        #self.buf.sampling_buffer = []
                        self.new_cycle = 1
                        self.setup_model()
                        for update in range(1, n_updates+1):
                            assert self.n_batch % self.nminibatches == 0, ("The number of minibatches (`nminibatches`) "
                                                                           "is not a factor of the total number of samples "
                                                                           "collected per rollout (`n_batch`), "
                                                                           "some samples won't be used."
                                                                           )
                            batch_size = self.n_batch // self.nminibatches
                            t_start = time.time()
                            frac = 0.0005#max(1.0 - 2*(update - 1.0) / n_updates, 0.00025)
                            lr_now = self.learning_rate(frac)
                            cliprange_now = self.cliprange(frac)
                            cliprange_vf_now = cliprange_vf(frac)
                            #print(tf.trainable_variables())
                            #Uncomment to see changes in weights
                            '''for var in self.params:
                                print(var)
                        
                            print(self.sess.run(self.weights))'''
                            callback.on_rollout_start()
                            # true_reward is the reward without discount
                            rollout = self.runner.run(callback)
                            # Unpack
                            obs, returns, masks, actions, values, neglogpacs, states, ep_infos, true_reward, exp_reward = rollout
                            self.values = values
                            callback.on_rollout_end()
                        
                            self.new_cycle = 0
                        
                            # Early stopping due to the callback
                            if not self.runner.continue_training:
                                break

                            self.ep_info_buf.extend(ep_infos)
                            mb_loss_vals = []
                            if states is None:  # nonrecurrent version
                                update_fac = max(self.n_batch // self.nminibatches // self.noptepochs, 1)
                                inds = np.arange(self.n_batch)
                                for epoch_num in range(self.noptepochs):
                                    np.random.shuffle(inds)
                                    for start in range(0, self.n_batch, batch_size):
                                        timestep = self.num_timesteps // update_fac + ((epoch_num *
                                                                                        self.n_batch + start) // batch_size)    
                                        end = start + batch_size    
                                        mbinds = inds[start:end]
                                        slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                                        mb_loss_vals.append(self._train_step(lr_now, cliprange_now, *slices, writer=writer,    
                                                                             update=timestep, cliprange_vf=cliprange_vf_now))
                            else:  # recurrent version
                                update_fac = max(self.n_batch // self.nminibatches // self.noptepochs // self.n_steps, 1)
                                assert self.n_envs % self.nminibatches == 0
                                env_indices = np.arange(self.n_envs)    
                                flat_indices = np.arange(self.n_envs * self.n_steps).reshape(self.n_envs, self.n_steps)
                                envs_per_batch = batch_size // self.n_steps    
                                for epoch_num in range(self.noptepochs):
                                    np.random.shuffle(env_indices)
                                    for start in range(0, self.n_envs, envs_per_batch):    
                                        timestep = self.num_timesteps // update_fac + ((epoch_num *
                                                                                        self.n_envs + start) // envs_per_batch)
                                        end = start + envs_per_batch
                                        mb_env_inds = env_indices[start:end]
                                        mb_flat_inds = flat_indices[mb_env_inds].ravel()
                                        slices = (arr[mb_flat_inds] for arr in (obs, returns, masks, actions, values, neglogpacs))    
                                        mb_states = states[mb_env_inds]    
                                        mb_loss_vals.append(self._train_step(lr_now, cliprange_now, *slices, update=timestep,
                                                                             writer=writer, states=mb_states,
                                                                             cliprange_vf=cliprange_vf_now))

                            loss_vals = np.mean(mb_loss_vals, axis=0)
                            t_now = time.time()
                            fps = int(self.n_batch / (t_now - t_start))
                            if writer is not None:
                                total_episode_reward_logger(self.episode_reward,
                                                            true_reward.reshape((self.n_envs, self.n_steps)),
                                                            masks.reshape((self.n_envs, self.n_steps)),
                                                            writer, self.num_timesteps)

                            if self.verbose >= 1 and (update % log_interval == 0 or update == 1):
    
                                print(np.mean(true_reward))
                                f = open("rewards.txt", "a+")
                                f.write(str(np.mean(true_reward)) + "," + str(np.mean(exp_reward)) + "\n")
                                f.close()
                                print("Cycle", cyc, update)
                                explained_var = explained_variance(values, returns)
                                logger.logkv("serial_timesteps", update * self.n_steps)
                                logger.logkv("n_updates", update)
                                logger.logkv("total_timesteps", (iteration * rl_optimization) + self.num_timesteps)
                                logger.logkv("fps", fps)
                                logger.logkv("explained_variance", float(explained_var))
                                if len(self.ep_info_buf) > 0 and len(self.ep_info_buf[0]) > 0:    
                                    logger.logkv('ep_reward_mean', safe_mean([ep_info['r'] for ep_info in self.ep_info_buf]))
                                    logger.logkv('ep_len_mean', safe_mean([ep_info['l'] for ep_info in self.ep_info_buf]))
                                logger.logkv('time_elapsed', t_start - t_first_start)
                                for (loss_val, loss_name) in zip(loss_vals, self.loss_names):
                                    logger.logkv(loss_name, loss_val)
                                    if(loss_name == "value_loss"):    
                                        f1 = open("loss.txt", "a+")
                                        f1.write(str(loss_val) + "\n")
                                        f1.close()
                                logger.dumpkvs()

                        print("Optimizing Reward")
                      
                        #generate_expert_traj(self, "RL_traj_"+str(cyc+12), self.env, n_episodes=30)
                        
                        filenames = next(walk("."), (None, None, []))[2]
                        saved_trajs = [ f for f in filenames if "RL_traj_" in f]
                        ind = random.randint(0, len(saved_trajs)-1)
                        traj = saved_trajs[ind]

                        for ind in range(len(saved_trajs)):
                            traj = saved_trajs[ind]
                            data_all = [np.load(traj, allow_pickle=True) for fname in file_list]
                        
                            merged_data = expert_data
                            for i, data in enumerate(data_all):
                                action_data = data["actions"]
                                for k, v in data.items():
                                    if(k == "obs"):
                                        observations = []
                                        for j in range(len(v)):
                                            if(j < 20480*5):
                                                expert_actions = self.env.env_method("control", v[j])[0] 
                                                #expert_actions = np.reshape(expert_actions, (1, 3))
                                                expert_actions = (expert_actions + 1)/2.0
                                                observations.append(np.concatenate((v[j], expert_actions), axis=0))
                                            observations.append(np.concatenate((v[j], action_data[j]), axis=0))
                                        merged_data.update({k: merged_data[k]+observations})
                                        continue
                                    
                                    if(k == "actions"):
                                        actions = []
                                        for j in range(len(v)):
                                            if(j < 20480*5):
                                                actions.append(1.0)
                                            actions.append(0.0)
                                        merged_data.update({k: merged_data[k]+actions})
                                    else:
                                        merged_data.update({k: v})
                        
                            print("Total dataset size= ", len(merged_data), ind)
                      
                            rew_sum_RL = 0.0
                            rew_sum_exp = 0.0
                        
                            x = np.array(merged_data["obs"])
                            y = np.array(merged_data["actions"])
                        
                            '''for i in range(len(x)):
                                obs = np.reshape(x[i], (1, len(x[i])))
                                if(y[i] == 1.0):
                                   exp_rew = self.reward_model.predict(obs)[0]#np.reshape(Expert_inp, (1,139)))[0]
                                   rew_sum_exp += exp_rew
                                else:
                                   RL_rew = self.reward_model.predict(obs)[0]#np.reshape(Expert_inp, (1,139)))[0]
                                   rew_sum_RL += RL_rew
                        
                            print("Before ", rew_sum_RL, rew_sum_exp)'''

                            opt = tf.keras.optimizers.Adam(lr=0.0003)
                            self.reward_model.compile(optimizer=opt, loss=tf.keras.losses.binary_crossentropy)#self.loss_fn)
                            loss_history = self.reward_model.fit(x, y, epochs=50, shuffle=True, batch_size = 20480)
                            loss_history = loss_history.history["loss"]
                            loss_history = np.array(loss_history)
                            f = open("loss_history_2.txt", "a+")
                            np.savetxt(f, loss_history, delimiter="\n")
                            f.close()

                            rew_sum_RL = 0.0
                            rew_sum_exp = 0.0
                            '''for i in range(len(x)):
                                obs = np.reshape(x[i], (1, len(x[i])))
                                if(y[i] == 1.0):
                                   exp_rew = self.reward_model.predict(obs)[0]#np.reshape(Expert_inp, (1,139)))[0]
                                   rew_sum_exp += exp_rew
                                else:
                                   RL_rew = self.reward_model.predict(obs)[0]#np.reshape(Expert_inp, (1,139)))[0]
                                   rew_sum_RL += RL_rew
                        
                            print("After ", rew_sum_RL, rew_sum_exp)'''

                            print(self.reward_model.get_weights())

                            self.reward_model.save("Rew_Model_"+str(2+update//(2000000//self.n_batch)))
                        print("Reward Optimized")    
                 
                    callback.on_training_end()
                    return self

        def loss_fn(self, y_true, y_pred):
            #if(y_pred - y_true > 0):
            #loss = 2*(y_pred - y_true)
            #else:
            #loss = (y_pred - self.reward_model.predict(y_true))
            #loss = y_true*(y_pred)+0.1# + 0.001*np.sum(np.abs(self.reward_model.get_weights()))
            #loss = (tf.sigmoid(y_pred) - y_true)
            loss = (y_true*(tf.log(tf.sigmoid(y_pred))) + (1-y_true)*(tf.log(1-tf.sigmoid(y_pred))))
            return tf.reduce_mean(loss)
    
    
class Runner(AbstractEnvRunner):
    def __init__(self, *, env, model, n_steps, gamma, lam):
        """
        A runner to learn the policy of an environment for a model

        :param env: (Gym environment) The environment to learn from
        :param model: (Model) The model to learn
        :param n_steps: (int) The number of steps to run for each environment
        :param gamma: (float) Discount factor
        :param lam: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        """
        super().__init__(env=env, model=model, n_steps=n_steps)
        self.lam = lam
        self.gamma = gamma
        self.likelihood_ratio = 1.0
        self.policy_prob = 0.0
        self.norm_w = 1.0
        self.thresh_steps = 0
        self.last_trust_update = -1
        self.prev_mean_reward = 0.0#-0.035 #-0.067
        self.prev_ep_reward = 0.0
        self.cur_mean_reward = 0.0
        self.mean_updates = 1
        self.ep_reward = []
        self.exp_ep_reward = []
        self.og_model = self.model

    def run(self, callback: Optional[BaseCallback] = None) -> Any:
        """
        Collect experience.

        :param callback: (Optional[BaseCallback]) The callback that will be called
            at each environment step.
        """
        self.callback = callback
        self.continue_training = True
        return self._run()

    def _run(self):
        """
        Run a learning step of the model

        :return:
            - observations: (np.ndarray) the observations
            - rewards: (np.ndarray) the rewards
            - masks: (numpy bool) whether an episode is over or not
            - actions: (np.ndarray) the actions
            - values: (np.ndarray) the value function output
            - negative log probabilities: (np.ndarray)
            - states: (np.ndarray) the internal states of the recurrent policies
            - infos: (dict) the extra information of the model
        """
        '''if(self.model.new_cycle == 1):
            reward_mod = self.model.reward_model
            self.model = self.og_model
            self.model.reward_model = reward_mod
            print("Reverting Model")'''
        # mb stands for minibatch
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [], [], [], [], [], []
        mb_states = self.states
        ep_infos = []
        traj_val = 0.0
        expert_traj_val = 0.0
        loss = 0.0
        self.ep_reward = []
        self.exp_ep_reward = []
        for step in range(self.n_steps):
            actions, values, self.states, neglogpacs = self.model.step(self.obs, self.states, self.dones)
            expert_actions = self.env.env_method("control", self.obs)[0] 
            expert_actions = np.reshape(expert_actions, (1, 3))   

            clipped_actions = actions
            clipped_expert = expert_actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.env.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.env.action_space.low    , self.env.action_space.high)
                clipped_expert = np.clip(expert_actions, self.env.action_space.low, self.env.action_space.high)
          
            clipped_actions[0][0] = (clipped_actions[0][0] + 1)/2.0
            clipped_actions[0][1] = (clipped_actions[0][1] + 1)/2.0
            clipped_actions[0][2] = (clipped_actions[0][2] + 1)/2.0
            clipped_expert[0][0] = (clipped_actions[0][0] + 1)/2.0
            clipped_expert[0][1] = (clipped_expert[0][1] + 1)/2.0
            clipped_expert[0][2] = (clipped_expert[0][2] + 1)/2.0

            RL_inp = np.concatenate((self.obs, clipped_actions), axis=1)
            Expert_inp = np.concatenate((self.obs, clipped_expert), axis=1)
            
            mb_obs.append(self.obs.copy())
            mb_dones.append(self.dones)
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)

            rewards = self.model.reward_model.predict(RL_inp)[0]
            exp_rewards = self.model.reward_model.predict(Expert_inp)[0]

            #if(step < 10):
                #print(rewards)

            self.ep_reward.append(rewards)
            self.exp_ep_reward.append(exp_rewards)

            loss += (rewards - exp_rewards)
            
            '''mean_act, std_act = self.model.reward_model.proba_step(self.obs, self.states, self.dones)
            action_probs = scipy.stats.norm(mean_act.flatten()[0], std_act.flatten()[0]).pdf(control_actions[0][1])
                    neglogpacs = [-np.sum(np.log(action_probs))]

            RL_classification = tf.math.exp(rewards) / (tf.math.exp(rewards) + tf.math.exp(log_p) + 1e-8)

            return self.sigmoid(value)'''


            #Execute action in the environment to find the reward
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.env.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)
            self.obs[:], _, self.dones, infos = self.env.step(clipped_actions)
            
            self.model.num_timesteps += self.n_envs

            if self.callback is not None:
                # Abort training early
                self.callback.update_locals(locals())
                if self.callback.on_step() is False:
                    self.continue_training = False
                    # Return dummy values
                    return [None] * 9

            for info in infos:
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    ep_infos.append(maybe_ep_info)
            mb_rewards.append(rewards)

        print("Expected Loss", loss/self.n_steps)
        print("RL Reward = ", sum(self.ep_reward), "Expert Reward = ", sum(self.exp_ep_reward))

        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_exp_rewards = np.asarray(self.exp_ep_reward, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, self.states, self.dones)
        
        # discount/bootstrap off value fn
        mb_advs = np.zeros_like(mb_rewards)
        true_reward = np.copy(mb_rewards)
        last_gae_lam = 0
        for step in reversed(range(self.n_steps)):
            if step == self.n_steps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[step + 1]
                nextvalues = mb_values[step + 1]
            delta = mb_rewards[step] + self.gamma * nextvalues * nextnonterminal - mb_values[step]
            mb_advs[step] = last_gae_lam = delta + self.gamma * self.lam * nextnonterminal * last_gae_lam
        mb_returns = mb_advs + mb_values

        mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward, mb_exp_rewards= \
            map(swap_and_flatten, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward, mb_exp_rewards))

        return mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, mb_states, ep_infos, true_reward, mb_exp_rewards
        
        
# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def swap_and_flatten(arr):
    """
    swap and then flatten axes 0 and 1

    :param arr: (np.ndarray)
    :return: (np.ndarray)
    """
    shape = arr.shape
    return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])