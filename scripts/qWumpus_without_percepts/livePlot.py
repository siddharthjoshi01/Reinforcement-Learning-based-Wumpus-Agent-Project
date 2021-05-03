import random
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time


def animate(i):
    data = pd.read_csv('score.csv')
    episode = data['episode']
    score = data['score']
    
    plt.cla()

    plt.scatter(episode, score, label='score')
    

    plt.legend(loc='upper left')
    plt.tight_layout()
    #time.sleep(0.01)


ani = FuncAnimation(plt.gcf(), animate, interval=50)

plt.tight_layout()
plt.show()