import matplotlib.pyplot as plt
import numpy as np

plt.xkcd()

data = [0, 5, 100]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.bar([], data, 0.25)

ax.xlabel('Technique')
ax.ylabel('Accuracy (%)')
ax.set_xticks([0, 1, 2])
ax.set_xlim([-0.5, 2.5])
ax.set_ylim([0, 110])
ax.set_xticklabels(['Eigenfaces (new samp)', '', 'Fisherfaces'])
