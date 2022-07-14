import numpy as np
from matplotlib import pyplot as plt


def plot_cut(ax, cut):
    xx = np.array(ax.get_xlim())
    yy = np.array(ax.get_ylim())

    if abs(cut[0]) > abs(cut[1]):
        ax.plot((cut[2] - cut[1]*yy)/cut[0], yy, 'k-')
    else:
        ax.plot(xx, (cut[2] - cut[0]*xx)/cut[1], 'k-')
        
    ax.set_xlim(xx)
    ax.set_ylim(yy)