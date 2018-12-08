import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

epochs = np.arange(1,16)

one = np.array([0.3515625,0.44140625,0.5546875,0.546875,0.51171875,0.63671875,0.6015625,
                  0.55859375,0.65234375,0.625,0.578125,0.6796875,0.62890625,0.6484375,0.60546875])
two = np.array([0.28125,0.4609375,0.50390625,0.51953125,0.58984375,0.5703125,0.6484375,0.5859375,
                0.60546875,0.58984375,0.59375,0.6875,0.6640625,0.6640625,0.68359375])
three = np.array([0.3515625,0.41015625,0.515625,0.5078125,0.58203125,0.5625,0.65625,0.609375,
                  0.60546875,0.62890625,0.67578125,0.62109375,0.58984375,0.6328125,0.61328125])

plt.figure()
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.plot(epochs,one)
plt.plot(epochs,two)
plt.plot(epochs,three)
plt.title("Accuracy vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.grid()
#plt.plot(t,F,zorder=10)
plt.show()
