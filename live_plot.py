#! /usr/bin/python
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import sys
import numpy as np

def animate(data):
    global y_max, x, y, time
    time += 1

    try:
        # we must recalculate the abscissa range
        times.append(time)

        s.append(float(data[0]))
        i.append(float(data[1]))
        r.append(float(data[2]))
        d.append(float(data[3]))

        ylist = [s, i, r, d]

        ax1.set_xlim(0, time)
        # add the new plot coordinate
        for lnum,line in enumerate(lines):
            line.set_data(times, ylist[lnum])

        return lines

    except KeyboardInterrupt:
        print("leaving...")

# The data generator take its
# input from file or stdin
def data_gen():
    while True: 
        line = fd.readline().split(",")
        if line:
            yield line

time = 0
fd = sys.stdin
header = fd.readline().split(",")
fig = plt.figure()
ax1  = fig.add_subplot(111)

line, = ax1.plot([], []) 
lines = []
plotlays, plotcols = [4], ["blue", "red", "green", "black"]
for i in range(4):
    lobj = ax1.plot([], [], lw=2, color=plotcols[i])[0]
    lines.append(lobj)

times = []
s = []
i = []
r = []
d = []

y_max = 1500
ax1.set_ylim(0, y_max)

anim = animation.FuncAnimation(fig, animate, frames=data_gen, repeat=False, interval=0)

plt.show()
