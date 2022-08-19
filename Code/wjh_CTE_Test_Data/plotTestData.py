import numpy as np
import matplotlib.pyplot as plt

#a = np.load("./wjh_SAC_Models/logs/epsdReward4-22.npy")
a = np.load("./cte1_speeds.npy")
b = np.load("./cte1_ctes.npy")
c = np.load("./cte5_speeds.npy")
d = np.load("./cte5_ctes.npy")
#a = np.load("./wjh_A3C_Models/logs/epsdReward4-19.npy")
#a = np.load("./wjh_SAC_Models/logs/stepReward4-17.npy")

l = 3500
a = a[:l]
b = np.abs(b[:l])
c = c[:l]
d = np.abs(d[:l])

print(np.average(a))
print(np.average(b))
print(np.average(c))
print(np.average(d))


plt.subplot(212)
plt.plot(a[:],label="when cte factor is 1")
plt.plot(c[:],label="when cte factor is 5")
plt.xlabel("steps")
plt.ylabel("speed")
plt.subplot(211)
plt.plot(b[:],label="when cte factor is 1")
plt.plot(d[:],label="when cte factor is 5")
plt.ylabel("cte")
plt.legend()
plt.show()

