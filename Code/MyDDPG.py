import numpy as np
import pandas as pd
import scipy.special
import matplotlib.pyplot as plt
import gym
import tensorflow as tf
import keras
from keras.callbacks import TensorBoard

import gym
from keras import layers
from keras.models import Model
from keras.models import Sequential
from keras.models import clone_model
from keras.models import load_model
from keras.optimizers import RMSprop
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras import backend as K
import math
import time
from skimage.transform import resize
from collections import deque
import random
import datetime

from config import MAX_THROTTLE, MIN_THROTTLE, MAX_STEERING, MIN_STEERING

from donkey_gym.envs.donkey_sim import DonkeyUnitySimContoller

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import keras.backend.tensorflow_backend as KTF

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
sess = tf.Session(config=config)
KTF.set_session(sess)  # 设置session

IMAGE_HEIGHT = 80
IMAGE_WIDTH = 160
IMAGE_CHANNEL = 3

INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL)

# action  0 steer   1 throttle
ACTIONS = np.array([
    [MIN_STEERING, MIN_THROTTLE],[0.0, MIN_THROTTLE],[MAX_STEERING, MIN_THROTTLE],
#    [-1.0, 0.5], [0.0, 0.5], [1.0, 0.5],
    [MIN_STEERING, MAX_THROTTLE], [0.0, MAX_THROTTLE], [MAX_STEERING, MAX_THROTTLE]
])

n_actions = len(ACTIONS)

action_types = 2

class DQNReplayer:
    def __init__(self, capacity):
        self.memory = pd.DataFrame(index=range(capacity),
                                   columns=['observation', 'action', 'reward',
                                            'next_observation', 'done'])
        self.i = 0
        self.count = 0
        self.capacity = capacity

    def store(self, *args):
        self.memory.loc[self.i] = args
        self.i = (self.i + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def sample(self, size):
        indices = np.random.choice(self.count, size=size)
        # print(indices)
        # print(np.array(((np.stack(self.memory.loc[indices, field]) for field in
        #         self.memory.columns))).shape)
        return (np.stack(self.memory.loc[indices, field]) for field in
                self.memory.columns)

class OrnsteinUhlenbeckProcess:
    def __init__(self, mu=0., theta=.15, sigma=1., dt=.01):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt

    def __call__(self):
        W = np.random.normal()
        self.x += (self.theta * (self.mu - self.x) * self.dt +
                   self.sigma * np.sqrt(self.dt) * W)
        return self.x

    def reset(self, x=0.):
        self.x = x

#class SACAgent(QActorCriticAgent):
class SACAgent:
    def __init__(self, env, sess,
                 gamma=0.99, target_learning_rate=0.005,
                 replayer_initial_transitions=5000,
                 network_learning_rate=0.0001, replayer_capacity=1000,
                 batches=1, batch_size=4, savedic="./wjh_DDPG_Models",
                 resume=False, resumeDic = "",
                 explore = True, noise_scale=0.1 ):
        self.env = env
        self.action_low = [MIN_STEERING, MIN_THROTTLE]
        self.action_high = [MAX_STEERING, MAX_THROTTLE]

        self.sess = sess
        self.input_shape = INPUT_SHAPE
        self.action_n = n_actions
        self.gamma = gamma

        #  ------------------explore------------------------
        self.explore = explore
        self.replayer_initial_transitions = replayer_initial_transitions
        self.throttle_noise = OrnsteinUhlenbeckProcess(mu=0.0, theta=0.6, sigma=1., dt=.01)
        self.steer_noise = OrnsteinUhlenbeckProcess(mu=0.5, theta=1.0, sigma=1., dt=.001)
        self.throttle_noise.reset()
        self.steer_noise.reset()

        self.target_learning_rate = target_learning_rate
        self.network_learning_rate = network_learning_rate
        self.savedic = savedic
        self.resume = resume
        self.resumeDic = resumeDic

        self.batches = batches
        self.batch_size = batch_size
        self.replayer = DQNReplayer(replayer_capacity)

        # -----------------------------Actor--------------------------------------
        self.actor_state_input, self.actor_evaluate_net = self.build_actor(state_shape = INPUT_SHAPE,
                                                   output_n = action_types,
                                                   learning_rate=self.network_learning_rate)
        _, self.actor_target_net = self.build_actor(state_shape = INPUT_SHAPE,
                                                   output_n = action_types,
                                                   learning_rate=self.network_learning_rate)
        self.actor_critic_grad = tf.placeholder(tf.float32, [None, action_types])  # where we will feed de/dC (from critic)
        actor_model_weights = self.actor_evaluate_net.trainable_weights
        self.actor_grads = tf.gradients(self.actor_evaluate_net.output,
                                        actor_model_weights, -self.actor_critic_grad)  # dC/dA (from actor)
        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize = tf.train.AdamOptimizer(self.network_learning_rate).apply_gradients(grads)

        # -----------------------------Critic--------------------------------------
        self.critic_state_input, \
        self.critic_action_input,\
        self.q_evaluate_net = self.build_critic(state_shape = INPUT_SHAPE,
                                                   action_n = action_types,
                                                   learning_rate=self.network_learning_rate)
        _, _, self.q_target_net = self.build_critic(state_shape = INPUT_SHAPE,
                                                   action_n = action_types,
                                                   learning_rate=self.network_learning_rate)
        self.critic_grads = tf.gradients(self.q_evaluate_net.output,
                                         self.critic_action_input)  # where we calcaulte de/dC for feeding above

        # Initialize for later gradient calculations
        self.sess.run(tf.initialize_all_variables())


        self.update_target_net(self.q_target_net, self.q_evaluate_net)
        self.update_target_net(self.actor_target_net, self.actor_evaluate_net)

        if resume:
            print("Loading models...")
            self.load_models(resumeDic)

        print(self.actor_evaluate_net.summary())
        print(self.q_evaluate_net.summary())


    def build_actor(self, state_shape, output_n, learning_rate):

        input = layers.Input(state_shape, name='images')
        normalized = layers.Lambda(lambda x: x / 255.0, name='normalization', input_shape=state_shape)(input)

        conv1 = layers.Conv2D(32, (3, 3), strides=(2, 2), activation='relu', kernel_initializer='he_uniform')(normalized)
        pool1 = layers.MaxPool2D(2, 2)(conv1)
        conv2 = layers.Conv2D(64, (3, 3), strides=(2, 2), activation='relu', kernel_initializer='he_uniform')(pool1)
        pool2 = layers.MaxPool2D(2, 2)(conv2)
        conv3 = layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu', kernel_initializer='he_uniform')(pool2)
        flat = Flatten()(conv3)
        dense1 = layers.Dense(units=500, activation='tanh')(flat)
        out = layers.Dense(units=output_n, activation='tanh')(dense1)

        model = Model(input=input, output=out)
        optimizer = keras.optimizers.Adam(lr=learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        return input, model

    def build_critic(self, state_shape, action_n, learning_rate):

        state_input = layers.Input(state_shape, name='images')
        normalized = layers.Lambda(lambda x: x / 255.0, name='normalization', input_shape=state_shape)(state_input)

        conv1 = layers.Conv2D(32, (3, 3), strides=(2, 2), activation='relu', kernel_initializer='he_uniform')(normalized)
        pool1 = layers.MaxPool2D(2, 2)(conv1)
        conv2 = layers.Conv2D(64, (3, 3), strides=(2, 2), activation='relu', kernel_initializer='he_uniform')(pool1)
        pool2 = layers.MaxPool2D(2, 2)(conv2)
        conv3 = layers.Conv2D(128, (3, 3), strides=(1, 1), activation='relu', kernel_initializer='he_uniform')(pool2)
        flat = Flatten()(conv3)
        dense_state = layers.Dense(units=500, activation='relu')(flat)

        action_input = layers.Input((action_types,), name='action_input')
        dense_action = layers.Dense(500, activation='relu')(action_input)

        merged = layers.concatenate([dense_state, dense_action])
        dense_merged = layers.Dense(500, activation='relu')(merged)
        output = layers.Dense(units=1, activation='linear')(dense_merged)

        model = Model(input=[state_input, action_input], output=output)
        optimizer = keras.optimizers.Adam(lr=learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        return state_input, action_input, model


    def decide(self, observation):
        if self.explore and self.replayer.count < \
                self.replayer_initial_transitions:
            action = np.random.uniform(-1, 1, size=action_types)
            action = self.scale_action(action)
            return action

        action = self.actor_evaluate_net.predict(observation[np.newaxis])[0]
        action = self.scale_action(action)
        print(action)

        if self.explore:
            action[0] += self.steer_noise()
            action[1] += 0.01 * self.throttle_noise()
            action = np.clip(action, self.action_low, self.action_high)
            print(action)
        return action


    # ------------------把动作调整到正确的范围并加入噪声-----------------------
    def scale_action(self, action):

        # 输出动作是tanh (-1,1) 所以需要 (a+1)/2*(high-low)+low
        sc_action = (np.array(action) + 1) / 2 * \
                 (np.array(self.action_high) - np.array(self.action_low))\
                 + np.array(self.action_low)

        return sc_action

    def update_target_net(self, target_net, evaluate_net, learning_rate=0.1):
        target_weights = target_net.get_weights()
        evaluate_weights = evaluate_net.get_weights()
        average_weights = [(1. - learning_rate) * t + learning_rate * e
                           for t, e in zip(target_weights, evaluate_weights)]
        target_net.set_weights(average_weights)

    def save_models(self, DicName):
        DicName = str(DicName)
        isExists = os.path.exists(self.savedic + "/" + DicName)
        if not isExists:
            os.makedirs(self.savedic + "/" + DicName)

        self.actor_target_net.save(self.savedic + "/" + DicName + "/A.h5")
        self.q_target_net.save(self.savedic + "/" + DicName + "/Q.h5")

    def load_models(self, DicName):
        DicName = str(DicName)
        isExists = os.path.exists(self.savedic + "/"+ DicName)
        if not isExists:
            print("model dic not Exist")
            return False

        self.actor_target_net = load_model(
            self.savedic + "/"+ DicName + "/A.h5")
        self.actor_evaluate_net = load_model(
            self.savedic + "/"+ DicName + "/A.h5")
        self.q_target_net = load_model(
            self.savedic + "/"+ DicName + "/Q.h5")
        self.q_evaluate_net = load_model(
            self.savedic + "/"+ DicName + "/Q.h5")


        return True

    def learn(self, observation, action, reward, next_observation, done):
        self.replayer.store(observation, action,
                            reward, next_observation,done)
        if self.replayer.count > self.replayer_initial_transitions:
            if done:
                self.throttle_noise.reset()
                self.steer_noise.reset()
            for batch in range(self.batches):
                # 经验回放
                # observations, actions, rewards, next_observations, \
                # dones = self.replayer.sample(self.batch_size)
                cur_states, actions, rewards, new_states, dones = self.replayer.sample(self.batch_size)

                # ---------------------------训练critic------------------------
                target_actions = self.actor_target_net.predict(new_states)
                future_rewards = self.q_target_net.predict([new_states, target_actions])
                future_rewards = future_rewards.reshape(self.batch_size, )

                rewards += self.gamma * future_rewards * (1 - np.array(dones))

                evaluation = self.q_evaluate_net.fit([cur_states, actions], rewards, verbose=0)

                # ------------------------------训练actor-------------------
                predicted_actions = self.actor_evaluate_net.predict(cur_states)
                grads = self.sess.run(self.critic_grads, feed_dict={
                    self.critic_state_input: cur_states,
                    self.critic_action_input: predicted_actions
                })[0]

                self.sess.run(self.optimize, feed_dict={
                    self.actor_state_input: cur_states,
                    self.actor_critic_grad: grads
                })

                # 更新目标网络
                self.update_target_net(self.q_target_net,
                                       self.q_evaluate_net, self.target_learning_rate)
                self.update_target_net(self.actor_target_net,
                                       self.actor_evaluate_net, self.target_learning_rate)


env = DonkeyUnitySimContoller(level=0, port=9091, max_cte_error=2.5)
print("Environment init success")

sess = tf.Session()
K.set_session(sess)
agent = SACAgent(env, sess, gamma=0.99,
                 replayer_initial_transitions=3000,
                 batches=1, batch_size=4,
                 network_learning_rate=0.000025,
                 replayer_capacity=15000,
                 savedic="./wjh_RewardFunTest_Models_DDPG",
                 resume=False, resumeDic=""
#                 resume = True, resumeDic = "ForceSave1"
                 )

ForceSaveNum = 2
import signal
def signal_handler(signal,frame):
    global agent
    global ForceSaveNum
    print('What do you want?\n'
          'save -> save models\n'
          'exit -> exit\n')
    a = input()
    if a == "save":
        agent.save_models("ForceSave"+str(ForceSaveNum))
        ForceSaveNum += 1
        print("save success")
    elif a == "exit" :
        print("exit")
        exit(0)
    else:
        print("invalid command")

signal.signal(signal.SIGINT,signal_handler)

# 训练
def train():

    SAVE = True
    episodes = 10000
    episode_rewards = []
    all_step_rewards = []
    global_steps = 0
    step_count_each_ep = []

    step_temp = 20000

    for episode in range(episodes):
        episode_reward = 0

        obs, reward, done, info = env.reset()
        step = 0

        while True:
            action = agent.decide(obs)

            next_obs, reward, done, _ = env.step(action)
            #print("ac:{} scale:{}".format(action, agent.scale_action(action)))
            episode_reward += reward
            all_step_rewards.append(reward)
            agent.learn(obs, action, reward, next_obs,done)
            #print("speed:{} reward:{} done:{}".format(env.handler.speed, reward, done))
            if done:
                break
            step += 1
            obs = next_obs
        step_count_each_ep.append(step)
        global_steps += step
        if(SAVE and episode > 0 and global_steps > step_temp):
            print("Save model")
            step_temp += 20000
            if(agent.resume):
                agent.save_models(agent.resumeDic + "+step"+ str(global_steps))
            else:
                agent.save_models("ep" + str(episode) + "#step" + str(global_steps))

        print("action value {}".format(action))
        print("Q1 Value{}".format(agent.q_target_net.predict([obs[np.newaxis], action[np.newaxis]])[0]))
        print("ep:{}, reward:{:.2f}, step:{}, global_steps:{}, memery_num:{}".format(
            episode, episode_reward, step, global_steps, agent.replayer.count))
        episode_rewards.append(episode_reward)

        if(SAVE and episode > 0 and episode %10 == 0):
            np.save(agent.savedic + "/logs/epsdReward4-19.npy", np.array(episode_rewards))
            #np.save(agent.savedic + "/logs/stepReward4-16.npy", np.array(all_step_rewards))
            np.save(agent.savedic + "/logs/stepNum4-19.npy", np.array(step_count_each_ep))


def test():

    agent.resume = True
    agent.resumeDic = "ep1200"
    episodes = 10
    episode_rewards = []
    all_step_rewards = []
    global_steps = 0
    step_count_each_ep = []

    for episode in range(episodes):
        episode_reward = 0

        obs, reward, done, info = env.reset()
        step = 0
        while True:
            action = agent.decide(obs)
            next_obs, reward, done, _ = env.step(agent.scale_action(action))
            episode_reward += reward
            all_step_rewards.append(reward)
            # agent.learn(obs, action,reward, next_obs,done)
            if done:
                break
            step += 1
            obs = next_obs
        step_count_each_ep.append(step)
        global_steps += step

        print("ep:{}, reward:{:.2f}, step:{}, memery_num:{}".format(episode, episode_reward, step, agent.replayer.count))
        episode_rewards.append(episode_reward)

if __name__ == '__main__':
    train()
    #test()
