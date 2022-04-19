import time
import random
import gym
import numpy as np
import tensorflow as tf
from os import walk
from stable_baselines import logger
from stable_baselines.common import explained_variance, ActorCriticRLModel, tf_util, SetVerbosity, TensorboardWriter
from stable_baselines.common.runners import AbstractEnvRunner
from stable_baselines.common.policies import ActorCriticPolicy, RecurrentActorCriticPolicy
from stable_baselines.common.schedules import get_schedule_fn
from stable_baselines.common.tf_util import total_episode_reward_logger
from stable_baselines.common.math_util import safe_mean

from typing import Union, Optional, Any
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from customPPO2 import CustomPPO2

from stable_baselines.common.policies import MlpPolicy
from gym import spaces
import scipy

random.seed(1)
np.random.seed(1)
tf.set_random_seed(1)

#The code from the stable_baselines PPO2 is copied and edited as required
class self_play_ppo2(ActorCriticRLModel):
    """
    Proximal Policy Optimization algorithm (GPU version).
    Paper: https://arxiv.org/abs/1707.06347

    :param policy: (ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) Discount factor
    :param n_steps: (int) The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param ent_coef: (float) Entropy coefficient for the loss calculation
    :param learning_rate: (float or callable) The learning rate, it can be a function
    :param vf_coef: (float) Value function coefficient for the loss calculation
    :param max_grad_norm: (float) The maximum value for the gradient clipping
    :param lam: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param nminibatches: (int) Number of training minibatches per update. For recurrent policies,
        the number of environments run in parallel should be a multiple of nminibatches.
    :param noptepochs: (int) Number of epoch when optimizing the surrogate
    :param cliprange: (float or callable) Clipping parameter, it can be a function
    :param cliprange_vf: (float or callable) Clipping parameter for the value function, it can be a function.
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        then `cliprange` (that is used for the policy) will be used.
        IMPORTANT: this clipping depends on the reward scaling.
        To deactivate value function clipping (and recover the original PPO implementation),
        you have to pass a negative value (e.g. -1).
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        WARNING: this logging can take a lot of space quickly
    :param seed: (int) Seed for the pseudo-random generators (python, numpy, tensorflow).
        If None (default), use random seed. Note that if you want completely deterministic
        results, you must set `n_cpu_tf_sess` to 1.
    :param n_cpu_tf_sess: (int) The number of threads for TensorFlow operations
        If None, the number of cpu of the current machine will be used.
    """
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


        super().__init__(policy=policy, env=env, verbose=verbose, requires_vec_env=True,
                         _init_setup_model=_init_setup_model, policy_kwargs=policy_kwargs,
                         seed=seed, n_cpu_tf_sess=n_cpu_tf_sess)

        if _init_setup_model:
            self.setup_model()

    #Initialize the runner class
    def _make_runner(self):
        return Runner(env=self.env, model=self, n_steps=self.n_steps,
                      gamma=self.gamma, lam=self.lam, conn=self.conn)

    def _get_pretrain_placeholders(self):
        policy = self.act_model
        if isinstance(self.action_space, gym.spaces.Discrete):
            return policy.obs_ph, self.action_ph, policy.policy
        return policy.obs_ph, self.action_ph, policy.deterministic_action

    #This function is used to train the model by calculating its loss based on data collected
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
                    self.Importance_weight = tf.placeholder(tf.float32, [], name="Importance_weight")

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

                    #Normal PPO policy loss
                    pg_losses = -self.advs_ph * ratio
                    pg_losses2 = -self.advs_ph * tf.clip_by_value(ratio, 1.0 - self.clip_range_ph, 1.0 +
                                                                  self.clip_range_ph)
                    #self.pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))

                    #Applied importance sampling
                    self.Z = tf.reduce_sum(tf.maximum(self.AI_used*ratio, tf.clip_by_value(self.AI_used*ratio, 1.0 - self.clip_range_ph, 1.0 + self.clip_range_ph)))
                    self.pg_sample_loss = (tf.reduce_sum(tf.maximum(self.AI_used*pg_losses, self.AI_used*pg_losses2)) / self.Z) + (self.Importance_weight)*tf.log(self.Z)
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

    #This function is used to pass the data to calculate the various loss values, log and return them
    def _train_step(self, learning_rate, cliprange, obs, returns, masks, actions, values, neglogpacs, AI_used, imp_weight, update,
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
                  self.old_neglog_pac_ph: neglogpacs, self.old_vpred_ph: values, self.AI_used: AI_used, self.RL_used: RL_used, self.Importance_weight: imp_weight}
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


    #This is the main function that runs in a loop
    #Model_num is used to differentiate between the two models. 1 is for evade and 2 is for attack
    def learn(self, total_timesteps, iteration, model_num, conn, switch_freq, callback=None, log_interval=1, tb_log_name="PPO2",
              reset_num_timesteps=True):
        # Transform to callable if needed

        self.conn = conn
        self.learning_rate = get_schedule_fn(self.learning_rate)
        self.cliprange = get_schedule_fn(self.cliprange)
        cliprange_vf = get_schedule_fn(self.cliprange_vf)

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        callback = self._init_callback(callback)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                           as writer:
            self._setup_learn()

            t_first_start = time.time()
            n_updates = total_timesteps // self.n_batch

            prev_update = 0

            callback.on_training_start(locals(), globals())


            #We start by training model 1 and not allowing model 2 to update
            if(model_num == 1):
                allow_update = 1
            else:
                allow_update = 0

            for update in range(1, n_updates + 1):
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

		#Choose whether the model will be trained in this step or not. Every switch_freq steps the training shifts between model 1 and model 2
                if(update%(switch_freq//self.n_batch) == 0):
                    if(allow_update == 1):
                        allow_update = 0
                    else:
                        allow_update = 1

                if((allow_update != prev_update) and (update != 1)):
                    random.seed(1)
                    np.random.seed(1)
                    tf.set_random_seed(1)
                    print("RE-SEEDING")
                prev_update = allow_update

                callback.on_rollout_start()
                # call the run function to get trajectory data
                rollout = self.runner.run(model_num, allow_update, callback)

                if(allow_update):

                    # Unpack
                    obs, returns, masks, actions, values, neglogpacs, states, ep_infos, true_reward, AI_used, imp_weight, policy_prob  = rollout

                    callback.on_rollout_end()

                    # Early stopping due to the callback
                    if not self.runner.continue_training:
                        break

                    self.ep_info_buf.extend(ep_infos)
                    mb_loss_vals = []
                    if states is None and allow_update:  # nonrecurrent version
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
                                mb_loss_vals.append(self._train_step(lr_now, cliprange_now, *slices, imp_weight, writer=writer,
                                                                     update=timestep, cliprange_vf=cliprange_vf_now))
                    '''else:  # recurrent version
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
                                                                     cliprange_vf=cliprange_vf_now))'''

                    loss_vals = np.mean(mb_loss_vals, axis=0)
                    t_now = time.time()
                    fps = int(self.n_batch / (t_now - t_start))

                    if writer is not None:
                        total_episode_reward_logger(self.episode_reward,
                                                    true_reward.reshape((self.n_envs, self.n_steps)),
                                                    masks.reshape((self.n_envs, self.n_steps)),
                                                    writer, self.num_timesteps)

                    if self.verbose >= 1 and allow_update:
                        #log rewards and loss
                        print(np.mean(true_reward), np.shape(true_reward))
                        f = open("rewards_"+str(model_num)+".txt", "a+")
                        f.write(str(np.mean(true_reward)) + "," + str(policy_prob) + "\n")
                        f.close()
                        explained_var = explained_variance(values, returns)
                        logger.logkv("serial_timesteps", update * self.n_steps)
                        logger.logkv("n_updates", update)
                        logger.logkv("total_timesteps", (iteration * total_timesteps) + self.num_timesteps)
                        logger.logkv("fps", fps)
                        logger.logkv("explained_variance", float(explained_var))
                        if len(self.ep_info_buf) > 0 and len(self.ep_info_buf[0]) > 0:
                            logger.logkv('ep_reward_mean', safe_mean([ep_info['r'] for ep_info in self.ep_info_buf]))
                            logger.logkv('ep_len_mean', safe_mean([ep_info['l'] for ep_info in self.ep_info_buf]))
                        logger.logkv('time_elapsed', t_start - t_first_start)
                        for (loss_val, loss_name) in zip(loss_vals, self.loss_names):
                            logger.logkv(loss_name, loss_val)
                            if(loss_name == "value_loss"):
                                f1 = open("loss_"+str(model_num)+".txt", "a+")
                                f1.write(str(loss_val) + "\n")
                                f1.close()
                        logger.dumpkvs()

            callback.on_training_end()
            return self

    def save(self, save_path, cloudpickle=False):
        data = {
            "gamma": self.gamma,
            "n_steps": self.n_steps,
            "vf_coef": self.vf_coef,
            "ent_coef": self.ent_coef,
            "max_grad_norm": self.max_grad_norm,
            "learning_rate": self.learning_rate,
            "lam": self.lam,
            "nminibatches": self.nminibatches,
            "noptepochs": self.noptepochs,
            "cliprange": self.cliprange,
            "cliprange_vf": self.cliprange_vf,
            "verbose": self.verbose,
            "policy": self.policy,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "n_envs": self.n_envs,
            "n_cpu_tf_sess": self.n_cpu_tf_sess,
            "seed": self.seed,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs
        }

        params_to_save = self.get_parameters()

        self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)

    #This function is used to predict the action the model would take for a given observation, as well as the value of that state decided by the learnt value function
    def predict(self, observation, state=None, mask=None, deterministic=False):
        if state is None:
            state = self.initial_state
        if mask is None:
            mask = [False for _ in range(self.n_envs)]
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        actions, values, states, _ = self.step(observation, state, mask, deterministic=deterministic)

        clipped_actions = actions
        # Clip the actions to avoid out of bound error
        if isinstance(self.action_space, gym.spaces.Box):
            clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

        if not vectorized_env:
            if state is not None:
                raise ValueError("Error: The environment must be vectorized when using recurrent policies.")
            clipped_actions = clipped_actions[0]

        return clipped_actions, values, states


class Runner(AbstractEnvRunner):
    def __init__(self, *,  env: Union[gym.Env, VecEnv], model: 'BaseRLModel', n_steps, gamma, lam, conn):
        """
        A runner to learn the policy of an environment for a model

        :param env: (Gym environment) The environment to learn from
        :param model: (Model) The model to learn
        :param n_steps: (int) The number of steps to run for each environment
        :param gamma: (float) Discount factor
        :param lam: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        """
        self.env = env
        self.model = model
        n_envs = env.num_envs
        self.batch_ob_shape = (n_envs * n_steps,) + env.observation_space.shape
        self.obs = np.zeros((n_envs,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)
        self.obs = conn[0].get()
        conn[0].task_done()

        self.n_steps = n_steps
        self.states = model.initial_state
        self.dones = [False for _ in range(n_envs)]
        self.callback = None  # type: Optional[BaseCallback]
        self.continue_training = True
        self.n_envs = n_envs

        self.lam = lam
        self.gamma = gamma
        self.conn = conn

        self.policy_prob = 0.0
        self.norm_w = 1e-3
        self.last_trust_update = -1
        self.prev_mean_reward = 0.0
        self.prev_ep_reward = 0.0
        self.cur_mean_reward = 0.0
        self.mean_updates = 1
        self.ep_reward = []

    def run(self, model_num, allow_update, callback: Optional[BaseCallback] = None) -> Any:
        """
        Collect experience.

        :param callback: (Optional[BaseCallback]) The callback that will be called
            at each environment step.
        """
        self.callback = callback
        self.continue_training = True
        self.model_num = model_num
        self.update_buffers = allow_update
        return self._run()

    def policy_decide(self, policy_prob):
        return np.random.rand() > policy_prob

    def phase_condition(self, last_trust_update, cur_mean_reward, prev_mean_reward):
        return last_trust_update < 0 or (cur_mean_reward >= prev_mean_reward)

    def get_phase_step(self):
        return 0.1

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

        # mb stands for minibatch
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs, mb_unshaped_reward = [], [], [], [], [], [], []

        mb_states = self.states
        ep_infos = []
        model = self.model
        RL_used = 0
        AI_used = []
        #If a model is not being trained but only used for prediction. In a non-self-play setting this section of code can be ignored.
        if(self.update_buffers == 0):
            filenames = next(walk("."), (None, None, []))[2]
            #list of all previous saved models
            saved_models = [ f for f in filenames if "Model_"+str(self.model_num) in f]
            saved_models.sort()
            model_decider = random.random()
            f = open("model_used_"+str(self.model_num)+".txt", "a+")
            #Randomly pick from among older versions of the model. This is used to train a model against older versions of its opponent to prevent overfitting
            old_policy_range = 10	#how many older policies should be included in the pool to randomly pick from
            if(model_decider > 0.0 and saved_models != [] and len(saved_models[:-old_policy_range]) > 0):
                ind = 0
                if len(saved_models[:-old_policy_range]) > 1:
                    ind = random.randint(0, len(saved_models[:-old_policy_range])-1)
                fi = saved_models[:-old_policy_range][ind]
                print("Using file "+fi, ind, model_decider)
                model = self_play_ppo2.load(fi)
                model.set_env(self.env)
                f.write("0\n")
            else:
                print("Using latest model for tank " + str(self.model_num))
                f.write("1\n")
            f.close()


        #Run the environment for n time steps
        for _ in range(self.n_steps):
            actions, values, self.states, neglogpacs = model.step(self.obs, self.states, self.dones)
            #If the model is not allowed to train it will only predict
                #Choose between the RL policy action or the demonstrators action or even a random action
            if(self.policy_decide(self.policy_prob)):#if(time_steps > self.thresh_steps):# and alive != 0):
                rand_prob = 0.01
                #Demonstrator action is sampled
                if(self.model_num == 1):
                    control_actions = self.env.env_method("control_blue", self.obs)[0][0]
                else:
                    control_actions = self.env.env_method("control_blue", self.obs)[0][1]
                #Choose between random action and demonstrator action
                if(random.random() < rand_prob):
                    control_actions = np.array([random.random(), random.random(), random.random()])
                    control_actions[1] = (control_actions[1] * (1 - (-1))) + (-1)
                    control_action_prob = rand_prob
                else:
                    control_action_prob = 1.0 - rand_prob
                control_actions[0] = (control_actions[0] * (1 - (-1))) + (-1)
                control_actions[2] = (control_actions[2] * (1 - (-1))) + (-1)
                AI_used.append(1)
            else:
                if(self.update_buffers == 0):
                    control_actions, _, _ = model.predict(self.obs, deterministic = False)
                else:
                    #RL action is sampled
                    control_action_prob = 1.0
                    control_actions = actions
                    RL_used += 1
                    AI_used.append(0)

            control_actions = control_actions.reshape((1, 3))

            if(self.update_buffers == 1):
                if(self.dones):
                    print("Current RL policy sampling probability: ", self.policy_prob, "Normalizing coefficient for importance sampling: ", self.norm_w)
                    #Keep a track of the mean episode rewards
                    if(self.ep_reward != []):
                        mean_ep_reward = np.mean(np.array(self.ep_reward))
                        self.cur_mean_reward += mean_ep_reward
                        #If the policy performed better this episode compared to previous episode then reduce the effect of the demonstrations by reducing norm_w
                        if(mean_ep_reward > self.prev_ep_reward):
                            self.norm_w = max(self.norm_w/10.0, 1e-6)
                        #If the policy performed worse this episode compared to previous episode then increase the effect of the demonstrations by increasing norm_w
                        else:
                            self.norm_w = min(self.norm_w*10, 1e-2)
                        print("Prev ep= ", self.prev_ep_reward, "Cur_ep= ", mean_ep_reward)
                        self.prev_ep_reward = mean_ep_reward
                    print("Prev mean= ", self.prev_mean_reward, "Cur_mean= ", self.cur_mean_reward)
                    self.ep_reward = []


                episode = self.env.get_attr("episode")[0]
                #After every 50 episodes, check if the policy is performing well enough to phase it more control. This metric can be modified
                if(episode % 100 == 0 and episode != self.last_trust_update):
                    self.cur_mean_reward = self.cur_mean_reward/100.0
                    if(self.phase_condition(self.last_trust_update, self.cur_mean_reward, self.prev_mean_reward)):
                        self.policy_prob = min(self.policy_prob+self.get_phase_step(), 1.0)
                        self.prev_mean_reward = max(((self.mean_updates-1)/self.mean_updates)*self.prev_mean_reward + (1/self.mean_updates)*self.cur_mean_reward, 0.0)
                    #else:
                        #self.policy_prob = max(self.policy_prob-get_phase_step(), 0.1)


                    print("Prev mean= ", self.prev_mean_reward, "Cur mean= ", self.cur_mean_reward, "Mean Updates= ", self.mean_updates)
                    self.mean_updates += 1
                    self.cur_mean_reward = 0.0
                    self.last_trust_update = episode

                #Get the action probability if the action is sampled randomly or by the demonstrator
                if(control_action_prob != 1.0):
                    mean_act, std_act = self.model.proba_step(self.obs, self.states, self.dones)
                    action_probs = scipy.stats.norm(mean_act.flatten()[0], std_act.flatten()[0]).pdf(control_actions)
                    if(abs(control_action_prob - rand_prob) < 0.0001):
                        action_probs = np.array([0.5, 0.5, 0.5]) * control_action_prob	#In the case of random actions, all theactions have equal probability
                    else:
                        action_probs = np.array([1.0, 1.0, 1.0]) * control_action_prob	#Since the demonstrator is deterministic the probability of its action is always 1.0
                    neglogpacs = [-np.sum(np.log(action_probs))]

                mb_obs.append(self.obs.copy())
                mb_actions.append(control_actions)
                mb_values.append(values)
                mb_neglogpacs.append(neglogpacs)
                mb_dones.append(self.dones)

            #Communicate the action to be taken to the main training program
            self.conn[1].put(control_actions)
            self.conn[1].join()
            #Recieve the new observation and reward after taking the action
            self.obs[:], rewards, self.dones, infos, clipped_actions = self.conn[0].get()
            self.conn[0].task_done()


            actions = clipped_actions

            if(self.update_buffers == 1):
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
                mb_unshaped_reward.append(rewards)
                self.ep_reward.append(rewards)

        if(self.update_buffers == 0):
            return [], [], [], [], [], [], [], [], []

        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_rewards = np.reshape(mb_rewards, (self.n_steps, 1))
        mb_unshaped_reward = np.asarray(mb_unshaped_reward, dtype=np.float32)
        mb_unshaped_reward = np.reshape(mb_unshaped_reward, (self.n_steps, 1))
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, self.states, self.dones)
        AI_used = np.asarray(AI_used, dtype=np.float32)

        # discount/bootstrap off value fn
        mb_advs = np.zeros_like(mb_rewards)
        true_reward = np.copy(mb_unshaped_reward)
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

        true_reward = np.reshape(true_reward, (self.n_steps, 1))
        mb_dones = np.reshape(mb_dones, (self.n_steps, 1))

        print("Proportions RL_used = "+str(RL_used)+" AI_used = "+str(self.n_steps-RL_used))

        mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward = \
            map(swap_and_flatten, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward))

        return mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, mb_states, ep_infos, true_reward, AI_used, self.norm_w, self.policy_prob


# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def swap_and_flatten(arr):
    """
    swap and then flatten axes 0 and 1

    :param arr: (np.ndarray)
    :return: (np.ndarray)
    """
    shape = arr.shape
    return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

