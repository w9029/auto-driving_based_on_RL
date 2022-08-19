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

# action 0 steer   1 throttle
ACTIONS = np.array([
    [MIN_STEERING, MIN_THROTTLE],[0.0, MIN_THROTTLE],[MAX_STEERING, MIN_THROTTLE],
#    [-1.0, 0.5], [0.0, 0.5], [1.0, 0.5],
    [MIN_STEERING, MAX_THROTTLE], [0.0, MAX_THROTTLE], [MAX_STEERING, MAX_THROTTLE]
])

n_actions = len(ACTIONS)

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


#class SACAgent(QActorCriticAgent):
class SACAgent:
    def __init__(self, env,
                 gamma=0.99, alpha=0.99, target_learning_rate=0.005,
                 network_learning_rate=0.0001, replayer_capacity=1000,
                 batches=1, batch_size=4, savedic="./wjh_SAC_Models",
                 resume=False, resumeDic = ""):
        self.input_shape = INPUT_SHAPE
        self.action_n = n_actions
        self.gamma = gamma
        self.alpha = alpha
        self.target_learning_rate = target_learning_rate
        self.network_learning_rate = network_learning_rate
        self.savedic = savedic
        self.resume = resume
        self.resumeDic = resumeDic

        self.batches = batches
        self.batch_size = batch_size
        self.replayer = DQNReplayer(replayer_capacity)

        self.actor_net = self.build_network(input_shape=self.input_shape,
                                            output_size=self.action_n, output_activation="softmax",
                                            loss=self.sac_loss,
                                            learning_rate=self.network_learning_rate)
        self.q0_net = self.build_network(input_shape=self.input_shape,
                                         output_size=self.action_n,
                                         learning_rate=self.network_learning_rate)
        self.q1_net = self.build_network(input_shape=self.input_shape,
                                         output_size=self.action_n,
                                         learning_rate=self.network_learning_rate)
        self.v_evaluate_net = self.build_network(
            input_shape=self.input_shape, output_size=1, learning_rate=self.network_learning_rate)
        self.v_target_net = self.build_network(
            input_shape=self.input_shape, output_size=1,
            learning_rate=self.network_learning_rate)

        self.update_target_net(self.v_target_net, self.v_evaluate_net)

        if resume:
            print("Loading models...")
            self.load_models(resumeDic)

        print(self.actor_net.summary())
        print(self.q0_net.summary())
        print(self.v_evaluate_net.summary())

        self.update_target_net(self.v_target_net, self.v_evaluate_net)

    def sac_loss(self, y_true, y_pred):
        """ y_true 是 Q(*, action_n), y_pred 是 pi(*, action_n) """
        qs = self.alpha * tf.math.xlogy(y_pred, y_pred) - y_pred * y_true
        return tf.reduce_sum(qs, axis=-1)

    def build_network(self, output_size, input_shape=None,
                    output_activation=None,
                      loss=tf.losses.mean_squared_error, learning_rate=0.0001):

        input = layers.Input(INPUT_SHAPE, name='images')
        normalized = layers.Lambda(lambda x: x / 255.0, name='normalization', input_shape=input_shape)(input)

        conv1 = layers.Conv2D(32, (3, 3), strides=(2, 2), activation='relu', kernel_initializer='he_uniform')(normalized)
        pool1 = layers.MaxPool2D(2, 2)(conv1)
        conv2 = layers.Conv2D(64, (3, 3), strides=(2, 2), activation='relu', kernel_initializer='he_uniform')(pool1)
        pool2 = layers.MaxPool2D(2, 2)(conv2)
        conv3 = layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu', kernel_initializer='he_uniform')(pool2)
        flat = Flatten()(conv3)
        dense1 = layers.Dense(units=500, activation='relu')(flat)
        out = layers.Dense(units=output_size, activation=output_activation)(dense1)

        model = Model(input=input, output=out)
        optimizer = keras.optimizers.Adam(lr=learning_rate)
        model.compile(optimizer=optimizer, loss=loss)
        return model


    def decide(self, observation):
        probs = self.actor_net.predict(observation[np.newaxis])[0]
        action = np.random.choice(self.action_n, p=probs)
        return action

    def update_target_net(self, target_net, evaluate_net, learning_rate=1.):
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

        self.actor_net.save(self.savedic + "/" + DicName + "/A.h5")
        self.q0_net.save(self.savedic + "/" + DicName + "/Q1.h5")
        self.q1_net.save(self.savedic + "/" + DicName + "/Q2.h5")
        self.v_target_net.save(self.savedic + "/" + DicName + "/V.h5")

    def load_models(self, DicName):
        DicName = str(DicName)
        isExists = os.path.exists(self.savedic + "/"+ DicName)
        if not isExists:
            print("model dic not Exist")
            return False

        self.actor_net = load_model(
            self.savedic + "/"+ DicName + "/A.h5",
            custom_objects={'sac_loss': self.sac_loss})
        self.q1_net = load_model(
            self.savedic + "/"+ DicName + "/Q1.h5")
        self.q1_net = load_model(
            self.savedic + "/"+ DicName + "/Q2.h5")
        self.v_target_net = load_model(
            self.savedic + "/"+ DicName + "/V.h5")
        self.v_evaluate_net = load_model(
            self.savedic + "/"+ DicName + "/V.h5")

        return True

    def learn(self, observation, action, reward, next_observation, done):

        self.replayer.store(observation, action,
                            reward, next_observation,done)
        # if done:
        for batch in range(self.batches):
            # 经验回放
            observations, actions, rewards, next_observations, \
            dones = self.replayer.sample(self.batch_size)

            pis = self.actor_net.predict(observations)
            q0s = self.q0_net.predict(observations)
            q1s = self.q1_net.predict(observations)

            # 训练Actor
            self.actor_net.fit(observations, q0s, verbose=0)

            # 训练V
            q01s = np.minimum(q0s, q1s)
            entropic_q01s = pis * q01s - self.alpha * \
                            scipy.special.xlogy(pis, pis)
            v_targets = entropic_q01s.sum(axis=-1)

            self.v_evaluate_net.fit(observations, v_targets, verbose=0)

            # 训练Q
            next_vs = self.v_target_net.predict(next_observations)
            q_targets = rewards + self.gamma * (1. - dones) * \
                        next_vs[:, 0]
            q0s[range(self.batch_size), actions] = q_targets
            q1s[range(self.batch_size), actions] = q_targets

            self.q0_net.fit(observations, q0s, verbose=0)
            self.q1_net.fit(observations, q1s, verbose=0)

            # 更新目标网络
            self.update_target_net(self.v_target_net,
                                   self.v_evaluate_net, self.target_learning_rate)


env = DonkeyUnitySimContoller(level=0, port=9091, max_cte_error=2.5)
print("Environment init success")

agent = SACAgent(env, gamma=0.99, alpha=0.9,
                 batches=1, batch_size=4,
                 network_learning_rate=0.000025,
                 replayer_capacity=15000,
                 #savedic="./wjh_SAC_Models",
                 savedic="./wjh_Good_Models",
                 resume=False, resumeDic=""
                 #resume = True, resumeDic = "ep1900+step20696"
#                resume = True, resumeDic = "Force_Save1"
                 )

def test():

    # agent.load_models("SAC_4-18_ep1900+step20696")
    agent.load_models("SAC_4-18_Force_Save2")
    # agent.load_models("SAC_4-30_ep505#step560046_cte5")
    env.handler.max_cte_error = 3


    for episode in range(2):
        if episode==1:
            agent.load_models("SAC_4-30_ep505#step560046_cte5")

        episode_reward = 0
        speeds = []
        ctes = []

        obs, reward, done, info = env.reset()
        step = 0
        while True:
            action = agent.decide(obs)
            next_obs, reward, done, _ = env.step(ACTIONS[action].reshape((2,)) )
            episode_reward += reward

            speeds.append(env.handler.speed)
            ctes.append(env.handler.cte)

            print("speed:{} reward:{} done:{} cte:{}".format(env.handler.speed, reward, done, env.handler.cte))
            if done:
                break
            step += 1
            obs = next_obs

        print("ep:{}, reward:{:.2f}, steps:{}".format(episode, episode_reward, step))

        if episode == 0:
            np.save("wjh_CTE_Test_Data/cte1_speeds.npy", np.array(speeds))
            np.save("wjh_CTE_Test_Data/cte1_ctes.npy", np.array(ctes))
        else:
            np.save("wjh_CTE_Test_Data/cte5_speeds.npy", np.array(speeds))
            np.save("wjh_CTE_Test_Data/cte5_ctes.npy", np.array(ctes))


if __name__ == '__main__':
    test()
