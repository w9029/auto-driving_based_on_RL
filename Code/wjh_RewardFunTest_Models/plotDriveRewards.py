import numpy as np
import matplotlib.pyplot as plt


#a = np.load("./wjh_SAC_Models/logs/epsdReward4-22.npy")
a = np.load("./logs/epsdReward4-20.npy")
b = np.load("./logs/stepNum4-20.npy")
#a = np.load("./wjh_A3C_Models/logs/epsdReward4-19.npy")
#a = np.load("./wjh_SAC_Models/logs/stepReward4-17.npy")

a.tolist()
for i in range(len(a)):
    if a[i] > 600:
        print("{} {}".format(i,a[i]))

plt.plot(a[:],label="reward")
plt.plot(b[:],label="step")
plt.xlabel("episode")
plt.ylabel("reward")
plt.legend()
plt.show()
