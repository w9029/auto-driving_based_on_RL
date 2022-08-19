# Code adapted from https://github.com/araffin/rl-baselines-zoo
# Author: Antonin Raffin

import argparse
import os
import time

import gym
import numpy as np
import cv2
from donkey_gym.envs.donkey_sim import DonkeyUnitySimContoller

# from config import ENV_ID
# from utils.utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams

env = DonkeyUnitySimContoller(level=0, port=9091)
env.reset()
print("reset ok")
for i in range(100):

    observation, reward, done, info = env.step([0,0.5])

    cv2.imshow("img",observation)
    cv2.waitKey()
    print(observation.shape,reward,done,info.keys())
    print(observation)
    if(done):
        env.reset()
    print("---")
# env = create_test_env()
#
# obs = env.reset()
#
# running_reward = 0.0
# ep_len = 0
# for epsd in range(100):
#     for _ in range(1000):
#         action, _ = model.predict(obs, deterministic=deterministic)
#         # Clip Action to avoid out of bound errors
#         if isinstance(env.action_space, gym.spaces.Box):
#             action = np.clip(action, env.action_space.low, env.action_space.high)
#         obs, reward, done, infos = env.step(action)
#
#         env.render('human')
#
#         running_reward += reward[0]
#         ep_len += 1
#
#         if done and args.verbose >= 1:
#             # NOTE: for env using VecNormalize, the mean reward
#             # is a normalized reward when `--norm_reward` flag is passed
#             print("Episode Reward: {:.2f}".format(running_reward))
#             print("Episode Length", ep_len)
#             running_reward = 0.0
#             ep_len = 0
#
#     env.reset()
#     env.envs[0].env.exit_scene()
#     # Close connection does work properly for now
#     # env.envs[0].env.close_connection()
#     time.sleep(0.5)
