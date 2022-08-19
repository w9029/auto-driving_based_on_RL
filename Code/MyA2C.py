#!/usr/bin/env python

# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import gym
import gym_carla

import glob
import os
import sys

# Agent
import numpy as np
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
from keras.optimizers import Adam
import tensorflow.keras.backend as K
import math
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.color import rgb2gray
from skimage.transform import resize
from collections import deque
import random
import datetime

from donkey_gym.envs.donkey_sim import DonkeyUnitySimContoller



actor_learning_rate=0.000025
critic_learning_rate=0.000025
n_max_episodes = 50000
gamma = 0.99

IMAGE_HEIGHT = 80
IMAGE_WIDTH = 160
IMAGE_CHANNEL = 3

INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL)

# action 0 steer   1 throttle
ACTIONS = np.array([[-1.0, 0.3],[0.0, 0.3],[1.0, 0.3],
                    [-1.0, 0.5], [0.0, 0.5], [1.0, 0.5],
                    [-1.0, 0.6], [0.0, 0.6], [1.0, 0.6]])

n_actions = 9

class Agent():
    def __init__(self, savedic):
        self.savedic = savedic
        self.epsilon = 1
        self.final_epsilon = 0.01
        self.epsilon_decay = (self.epsilon - self.final_epsilon) / 12000
        self.episode = 0
        self.env = DonkeyUnitySimContoller(level=0, port=9091)
        print("Environment init success")
        self.actor_model = self.get_actor_model()
        print(self.actor_model.summary())
        self.critic_model = self.get_critic_model()
        print(self.critic_model.summary())

    # def pre_processing(self, observe):
    #     processed_observe = np.uint8(
    #         resize(rgb2gray(observe), (IMAGE_HEIGHT, IMAGE_WIDTH), mode='constant') * 255)
    #     return processed_observe

    def get_actor_model(self):
        frames_input = layers.Input(INPUT_SHAPE, name='frames')
        normalized = layers.Lambda(lambda x: x / 255.0, name='normalization')(frames_input)
        conv_1 = layers.convolutional.Conv2D(32, (5, 5), strides=(2, 2), activation='relu', kernel_initializer='he_uniform')(normalized)
        pool_1 = layers.pooling.MaxPool2D((3, 3))(conv_1)
        conv_2 = layers.convolutional.Conv2D(64, (3, 3), strides=(2, 2), activation='relu', kernel_initializer='he_uniform')(pool_1)
        #pool_2 = layers.pooling.MaxPool2D((2, 2))(conv_2)
        conv_3 = layers.convolutional.Conv2D(64, (3, 3), strides=(2, 2), activation='relu', kernel_initializer='he_uniform')(conv_2)
        #conv_4 = layers.convolutional.Conv2D(128, (3, 3), strides=(2, 2), activation='relu', kernel_initializer='he_uniform')(conv_3)

        conv_flattened = layers.core.Flatten()(conv_3)
        hidden = layers.Dense(512, activation='relu', kernel_initializer='he_uniform')(conv_flattened)
        output = layers.Dense(n_actions, activation='softmax', kernel_initializer='he_uniform')(hidden)
        model = Model(inputs=frames_input, outputs=output)
        model.compile(loss=self.actor_loss, optimizer=Adam(lr=actor_learning_rate))
        return model

    def get_critic_model(self):
        frames_input = layers.Input(INPUT_SHAPE, name='frames')
        normalized = layers.Lambda(lambda x: x / 255.0, name='normalization')(frames_input)
        conv_1 = layers.convolutional.Conv2D(32, (5, 5), strides=(2, 2), activation='relu', kernel_initializer='he_uniform')(normalized)
        pool_1 = layers.pooling.MaxPool2D((3, 3))(conv_1)
        conv_2 = layers.convolutional.Conv2D(64, (3, 3), strides=(2, 2), activation='relu', kernel_initializer='he_uniform')(pool_1)
        #pool_2 = layers.pooling.MaxPool2D((2, 2))(conv_2)
        conv_3 = layers.convolutional.Conv2D(64, (3, 3), strides=(2, 2), activation='relu', kernel_initializer='he_uniform')(conv_2)
        #conv_4 = layers.convolutional.Conv2D(128, (3, 3), strides=(2, 2), activation='relu', kernel_initializer='he_uniform')(conv_3)

        conv_flattened = layers.core.Flatten()(conv_3)
        hidden = layers.Dense(512, activation='relu', kernel_initializer='he_uniform')(conv_flattened)
        output = layers.Dense(1, kernel_initializer='he_uniform')(hidden)
        model = Model(inputs=frames_input, outputs=output)
        model.compile(loss='mse', optimizer=Adam(lr=critic_learning_rate))
        return model

    def save_models(self, DicName):
        DicName = str(DicName)
        isExists = os.path.exists(self.savedic + "/" + DicName)
        if not isExists:
            os.makedirs(self.savedic + "/" + DicName)

        self.actor_model.save(self.savedic + "/" + DicName  + "/A.h5")
        self.critic_model.save(self.savedic + "/" + DicName +"/C.h5")

    def load_models(self, Dicname):
        self.actor_model = load_model(
            self.savedic + "/" + Dicname + "_A.h5",
            custom_objects={'actor_loss': self.actor_loss})
        self.critic_model = load_model(self.savedic + "/" + Dicname + "_C.h5")

    def get_action(self, state, model):
        if np.random.rand() <= self.epsilon:
            return random.randrange(n_actions)
        else:
            action_output = model.predict(state)
            return  np.random.choice(np.array(range(n_actions)), size=1, p=action_output.ravel())

    def last_n_reward_average(self, n, game_rewards):
        last_n_rewards = game_rewards[-n:]
        return sum(last_n_rewards) / len(last_n_rewards)

    def actor_loss(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 0.001, 0.999)
        log_prob = tf.log(y_pred)
        loss = log_prob * y_true
        return loss

    def train(self, state, action, reward, dead, new_state):

        advantage = np.zeros(shape=(1, n_actions))
        target = np.zeros(shape=(1, 1))

        current_value = self.critic_model.predict(state, batch_size=1)
        next_value = self.critic_model.predict(new_state, batch_size=1)

        if dead:
            advantage[0][action] = reward - current_value
            target[0][0] = reward
        else:
            advantage[0][action] = reward + gamma * next_value - current_value
            target[0][0] = reward + gamma * next_value

        h1 = self.actor_model.fit(x=state, y=advantage, batch_size=1, verbose=0)
        h2 = self.critic_model.fit(x=state, y=target, batch_size=1, verbose=0)
        return h1.history['loss'][0], h2.history['loss'][0]

    def train_loop(self):
        best_reward = -math.inf
        episode_rewards = deque(maxlen=5000)
        #average_episode_rewards = []
        SAVE = True
        #self.load_models()

        for episode in range(n_max_episodes):
            print("episode "+str(episode)+" - training...")
            self.episode = episode
            oldtime = datetime.datetime.now()
            rewards = []
            states = []
            actions = []
            dones = []
            deads = []

            dead = False
            act_totol_loss, critic_totol_loss = 0,0
            step = 0

            obs, reward, done, info = self.env.reset()
            # input state
            state = obs
            #print(state)
            state = np.reshape([state], (1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL))

            while True:
                step = step+1
                if self.epsilon > self.final_epsilon:
                    self.epsilon -= self.epsilon_decay
                action = self.get_action(state, self.actor_model)

                #print(ACTIONS[action])

                observation, reward, done, info = self.env.step(ACTIONS[action].reshape((2,)))

                new_state = observation
                new_state = np.reshape([new_state], (1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL))

                rewards.append(reward)

                act_loss, critic_loss = self.train(state, action, reward, dead, new_state)
                act_totol_loss += act_loss
                critic_totol_loss += critic_loss

                state = new_state

                if done:
                    break

            #print info
            episode_reward = sum(rewards)
            episode_rewards.append(episode_reward)
            if episode_reward > best_reward:
                best_reward = episode_reward
            avg = self.last_n_reward_average(100, np.array(episode_rewards))
            #average_episode_rewards.append(avg)
            # if(episode!=0 and episode%20==0):
            print("current_V={}, current_prob={}, actor_totol_loss={}, critic_totol_loss={}".format(
                self.critic_model.predict(state),
                self.actor_model.predict(state),
                act_totol_loss,critic_totol_loss))
            print('Episode {}, episode_reward={}, step={}, Last_100_episode_average_reward={}, best reward={}, epsilon={}, time_interval={}'.format(
                episode, episode_reward, step, avg, best_reward, self.epsilon, (datetime.datetime.now() - oldtime).seconds))

            episode_rewards.append(episode_reward)

            if (SAVE and episode > 0 and episode % 10 == 0):
                np.save(self.savedic + "/logs/epsdReward4-19.npy", np.array(episode_rewards))
                # np.save(agent.savedic + "/logs/stepReward4-16.npy", np.array(all_step_rewards))
                #np.save(self.savedic + "/logs/stepNum4-19.npy", np.array(step_count_each_ep))

            #save model
            if(episode> 0 and episode % 100 == 0):
                self.save_models("ep{}".format(episode))

            # add epsilon
            # if(self.epsilon < 0.3):
            #     if(avg < 1):
            #         self.epsilon = 0.9
            # if(self.epsilon < 0.1):
            #     if(avg < 3):
            #         self.epsilon = 0.9
            #     elif(avg < 5):
            #         self.epsilon = 0.6
            #     elif(avg < 10):
            #         self.epsilon = 0.3

    def play(self):
        best_reward = -math.inf
        episode_rewards = deque(maxlen=200)
        #average_episode_rewards = []

        #self.load_models()

        for episode in range(n_max_episodes):
            print("episode "+str(episode)+" - training...")
            self.episode = episode
            oldtime = datetime.datetime.now()
            rewards = []
            states = []
            actions = []
            dones = []
            deads = []

            act_totol_loss, critic_totol_loss = 0,0
            step = 0

            obs, reward, done, info = self.env.reset()
            # input state
            state = obs
            state = np.reshape([state], (1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL))

            while True:
                step = step+1

                action = self.get_action(state, self.actor_model)

                #print(ACTIONS[action])
                observation, reward, done, info = self.env.step(ACTIONS[action].reshape((2,)))

                new_state = observation
                new_state = np.reshape([new_state], (1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL))

                rewards.append(reward)

                #act_loss, critic_loss = self.train(state, action, reward, dead, new_state)
                #act_totol_loss += act_loss
                #critic_totol_loss += critic_loss

                state = new_state

                if done:
                    break

            #print info
            episode_reward = sum(rewards)
            episode_rewards.append(episode_reward)
            if episode_reward > best_reward:
                best_reward = episode_reward
            avg = self.last_n_reward_average(100, np.array(episode_rewards))
            #average_episode_rewards.append(avg)
            # if(episode!=0 and episode%20==0):
            print("current_V={}, current_prob={}".format(
                self.critic_model.predict(state),
                self.actor_model.predict(state)))
            print('Episode {}, episode_reward={}, step={}, Last_100_episode_average_reward={}, best reward={}, time_interval={}'.format(
                episode, episode_reward, step, avg, best_reward, (datetime.datetime.now() - oldtime).seconds))


def main():

  agent = Agent(savedic="./wjh_A3C_Models")
  agent.train_loop()

if __name__ == '__main__':
  main()