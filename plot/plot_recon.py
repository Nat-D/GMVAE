import numpy as np
import torchfile
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


#np.random.seed(16)
np.random.seed(11)
# set up
plt.ion()

fig = plt.figure()
ax2 = fig.add_subplot(111)


mean = torchfile.load('../save/m.t7')
colors = {}
for j in xrange(len(mean)):
    colors[j] = np.random.rand(3,1)


plotting = True
count = 0
while plotting:

    x = torchfile.load('../save/xs.t7')
    ax2.cla()
    y = torchfile.load('../datasets/spiral.t7')
    ax2.scatter(y[:,0], y[:, 1], marker='.' , s = 5 )
    y_recon = torchfile.load('../save/recon.t7')
    label   = torchfile.load('../save/label.t7')

    # draw ellipse
    cov = torchfile.load('../save/cov.t7')
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    for i in range(len(cov)):
        vals, vecs = eigsorted(cov[i])

        # plots latents samples
        key = label - 1 == i
        # plot reconstruction
        ax2.scatter(y_recon[key,0], y_recon[key, 1], marker='.' , s = 5 , color=colors[i])



    plt.draw()
    plt.pause(0.05)
    count += 1
    if count > 100000:
        plotting = False
        print('Finish Simulation')

plt.ioff()
plt.show()
