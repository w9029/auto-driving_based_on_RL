import numpy as np
import matplotlib.pyplot as plt

# a = np.load("./logs/epsdReward4-20.npy")
# b = np.load("./logs/stepNum4-20.npy")
data =''
with open("VAESACResult.txt", "r") as f:
    data = f.readlines()
    #print(data)

rewards =[]
steps=[]

for a in data:
    a.replace('\n',' ')
    a = a.split(" ")
    #print(float(a[0]),float(a[1]))
    rewards.append(float(a[0]))
    steps.append(float(a[1]))

print(rewards)
print(steps)

# a.tolist()
# for i in range(len(a)):
#     if a[i] > 600:
#         print("{} {}".format(i,a[i]))

plt.plot(rewards[:],label="reward")
plt.plot(steps[:],label="step")
plt.xlabel("episode")
plt.ylabel("reward")
plt.legend()
plt.show()