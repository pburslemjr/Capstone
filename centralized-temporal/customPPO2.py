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
                        self.AI_used = tf.placeholder(tf.float32, [None], name="AI_used")
                        self.RL_used = tf.placeholder(tf.float32, [None], name="RL_used")

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
                        self.Z = tf.reduce_sum(tf.maximum(self.AI_used*ratio, tf.clip_by_value(self.AI_used*ratio, 1.0 - self.clip_range_ph, 1.0 + self.clip_range_ph)))
                        self.pg_sample_loss = (tf.reduce_sum(tf.maximum(self.AI_used*pg_losses, self.AI_used*pg_losses2)) / self.Z) + (self.runner.norm_w)*tf.log(self.Z)
                        self.pg_rl_loss = tf.reduce_mean(tf.maximum(self.RL_used*pg_losses, self.RL_used*pg_losses2))

                        self.pg_loss = self.pg_sample_loss + self.pg_rl_loss
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


        def _train_step(self, learning_rate, cliprange, obs, returns, masks, actions, values, neglogpacs, AI_used, update,
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
            RL_used = np.ones(AI_used.shape) - AI_used
            td_map = {self.train_model.obs_ph: obs, self.action_ph: actions,
                      self.advs_ph: advs, self.rewards_ph: returns,
                      self.learning_rate_ph: learning_rate, self.clip_range_ph: cliprange,
                      self.old_neglog_pac_ph: neglogpacs, self.old_vpred_ph: values, self.AI_used: AI_used, self.RL_used: RL_used}
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
                            obs, returns, masks, actions, values, neglogpacs, states, ep_infos, unshaped_rew, policy_prob, AI_used = rollout
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
                                        slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs, AI_used))
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
                                                            unshaped_rew.reshape((self.n_envs, self.n_steps)),
                                                            masks.reshape((self.n_envs, self.n_steps)),
                                                            writer, self.num_timesteps)

                            if self.verbose >= 1 and (update % log_interval == 0 or update == 1):

                                print("mean of unshaped reward (global rewards): " + str(np.mean(unshaped_rew)))
                                f = open("rewards.txt", "a+")
                                #write the average true, shaped, and unshaped step reward for each episode
                                #each episode is on a new line and rewards are separated by a space
                                f.write(str(0.0) + " " + str(0.0) + " " + str(np.mean(unshaped_rew)) + " " + str(policy_prob) + "\n")
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

                    callback.on_training_end()
                    return self

        def loss_fn(self, y_true, y_pred):
            #if(y_pred - y_true > 0):
            #loss = 2*(y_pred - y_true)
            #else:
            #loss = (y_pred - self.reward_model.predict(y_true))
            #loss = y_true*(y_pred)+0.1# + 0.001*np.sum(np.abs(self.reward_model.get_weights()))
            #loss = (tf.sigmoid(y_pred) - y_true)
            #loss = (y_true*(tf.log(tf.sigmoid(y_pred))) + (1-y_true)*(tf.log(1-tf.sigmoid(y_pred))))
            loss = self.sign*tf.keras.losses.binary_crossentropy(y_true, y_pred)
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
        self.policy_prob = 0.3
        self.norm_w = 1e-3
        self.thresh_steps = 0
        self.last_trust_update = -1
        self.prev_mean_reward = 0.0#-0.035 #-0.067
        self.prev_ep_reward = 0.0
        self.cur_mean_reward = 0.0
        self.mean_updates = 1
        self.ep_reward = []
        self.exp_ep_reward = []
        self.og_model = self.model

    def policy_decide(self, policy_prob):
        return np.random.rand() > policy_prob

    def phase_condition(self, last_trust_update, cur_mean_reward, prev_mean_reward):
        print("last: " + str(prev_mean_reward) + " curr: " +str(cur_mean_reward))
        return last_trust_update < 0 or (cur_mean_reward >= prev_mean_reward)

    def get_phase_step(self):
        return 0.05

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
        mb_obs, mb_rewards, mb_unshaped_rew, mb_actions, mb_values, mb_dones, mb_neglogpacs, likelihood_ratio = [], [], [], [], [], [], [], []
        mb_states = self.states
        ep_infos = []
        traj_val = 0.0
        expert_traj_val = 0.0
        loss = 0.0
        AI_used = []
        RL_used = 0
        policy_prob = None
        self.ep_reward = []
        self.exp_ep_reward = []
        for step in range(self.n_steps):
            actions, values, self.states, neglogpacs = self.model.step(self.obs, self.states, self.dones)

            mb_obs.append(self.obs.copy())
            mb_dones.append(self.dones)
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            alive = self.env.get_attr("alive")[0]

            clipped_actions = None
            if(self.policy_decide(self.policy_prob)):
                rand_prob = 0.02
                if random.random() < rand_prob:
                    clipped_actions = [np.array([random.random(), random.random(), random.random(), random.random(), random.random(), random.random()])]
                    clipped_actions[0][1] = (clipped_actions[0][1] * (1 -(-1)) + (-1))
                    clipped_actions[0][4] = (clipped_actions[0][4] * (1 -(-1)) + (-1))
                else:
                    clipped_actions = self.env.env_method("control", self.obs)

                AI_used.append(1)
            else:
                clipped_actions = actions
                RL_used+=1
                AI_used.append(0)

            clipped_actions[0][0] = (clipped_actions[0][0] * (1 -(-1)) + (-1))
            clipped_actions[0][2] = (clipped_actions[0][2] * (1 -(-1)) + (-1))
            clipped_actions[0][3] = (clipped_actions[0][3] * (1 -(-1)) + (-1))
            clipped_actions[0][5] = (clipped_actions[0][5] * (1 -(-1)) + (-1))


            if(alive == 0):
                self.likelihood_ratio = 1.0
            #Execute action in the environment to find the reward
            # Clip the actions to avoid out of bound error
            if isinstance(self.env.action_space, gym.spaces.Box):
                clipped_actions = np.clip(clipped_actions, self.env.action_space.low, self.env.action_space.high)

            episode = self.env.get_attr("episode")[0]


            self.obs[:], rewards, self.dones, infos = self.env.step(clipped_actions)

            self.model.num_timesteps += self.n_envs

            if self.callback is not None:
                # Abort training early
                self.callback.update_locals(locals())
                if self.callback.on_step() is False:
                    self.continue_training = False
                    # Return dummy values
                    return [None] * 9

            #for info in infos:
            #    maybe_ep_info = info.get('episode')
            #    if maybe_ep_info is not None:
            #        ep_infos.append(maybe_ep_info)
            self.ep_reward.append(rewards)
            for info in infos:
                maybe_ep_info = info.get('episode')
                #shaped_rew + local
                #just global
                maybe_unshaped_info = info.get('unshaped')
                if maybe_ep_info is not None:
                    ep_infos.append(maybe_ep_info)
                if maybe_unshaped_info is not None:
                    mb_unshaped_rew.append(maybe_unshaped_info)
            mb_rewards.append(rewards)

            if (self.dones):
                self.likelihood_ratio = 1.0
                print("norm_W: " + str(self.norm_w))
                if (self.ep_reward != []):
                    mean_ep_rew = np.mean(np.array(self.ep_reward))
                    self.cur_mean_reward += mean_ep_rew
                    if(mean_ep_rew > self.prev_ep_reward):
                        self.norm_w = max(self.norm_w/10.0, 1e-5)
                    else:
                        self.norm_w = min(self.norm_w*10.0, 1.0)
                    print("Prev ep= ", self.prev_ep_reward, "Cur_ep= ", mean_ep_rew)
                    self.prev_ep_reward = mean_ep_rew
                self.ep_reward = []
                print("EPISODE DONE")
                self.env.reset()


            if (episode % 100 == 0 and episode != self.last_trust_update):
                self.cur_mean_reward = self.cur_mean_reward / 100.0
                if self.phase_condition(self.last_trust_update, self.cur_mean_reward, self.prev_mean_reward):
                    print("PHASING")
                    self.policy_prob = min(self.policy_prob + self.get_phase_step(), 1.0)
                self.prev_mean_reward = max(((self.mean_updates -1) / self.mean_updates) * self.prev_mean_reward + (1.0 / self.mean_updates) * self.cur_mean_reward, 0.0)

                print("Prev mean= ", self.prev_mean_reward, "Cur mean= ", self.cur_mean_reward, "Mean Updates= ", self.mean_updates)

                print("Policy Prob: " + str(self.policy_prob))
                self.mean_updates += 1
                self.cur_mean_reward = 0.0
                self.last_trust_update = episode
            #mb_shaped_rew.append(infos["shaped"])
            #mb_unshaped_rew.append(infos["unshaped"])

        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_unshaped_rew = np.asarray(mb_unshaped_rew, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, self.states, self.dones)
        AI_used = np.asfarray(AI_used, dtype=np.float32)

        # discount/bootstrap off value fn
        mb_advs = np.zeros_like(mb_rewards)
        unshaped_rew = np.copy(mb_unshaped_rew)
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

        print("Proportions RL_used = "+str(RL_used)+" AI_used = "+str(self.n_steps-RL_used))

        mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs= \
            map(swap_and_flatten, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs))

        return mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, mb_states, ep_infos, unshaped_rew, self.policy_prob, AI_used


# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def swap_and_flatten(arr):
    """
    swap and then flatten axes 0 and 1

    :param arr: (np.ndarray)
    :return: (np.ndarray)
    """
    shape = arr.shape
    return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])
