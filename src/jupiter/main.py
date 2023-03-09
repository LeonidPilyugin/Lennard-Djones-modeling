from data import Data
import seaborn as sn
import matplotlib.pyplot as plot
import numpy as np
import math
import scipy
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from math import exp, sqrt

data = Data("output.txt")

fig = plot.figure(figsize=(6,5))
axes = plot.axes()
axes.grid()
xx, yy = np.array(range(0, len(data.get_total_energies()) * data.skip, data.skip)), np.array(data.get_total_energies())
axes.plot(xx, data.get_total_energies(), '.r')
axes.axhline(data.get_total_energies()[0], color='green')
axes.set_xlabel("iteration")
axes.set_ylabel("total energy")
plot.savefig("plots/total_energy.png")
plot.gca()

fig = plot.figure(figsize=(6,5))
axes = plot.axes()
xx, yy = np.array(range(0, len(data.get_kinetic_energies()) * data.skip, data.skip)), np.array(data.get_kinetic_energies())
axes.plot(xx, yy, '.r')
axes.grid()
axes.set_xlabel("iteration")
axes.set_ylabel("kinetic energy")
plot.savefig("plots/kinetic_energy.png")
plot.gca()

def plot_ln2(i):
    T = data.frames[i].temperature

    fig = plot.figure(figsize=(6,5))
    axes = plot.axes()
    axes.set_xlabel("$v^2$")
    axes.set_ylabel("$\ln(dn/dv)$")
    axes.set_title(f"{i * data.skip} iterations")

    xx, yy = data.frames[i].get_distribution()
    xx = xx[:200]
    yy = yy[:200] / data.n / data.hist_width
    xx = xx ** 2
    yy = np.log(yy)

    axes.plot(xx, yy, '.', markersize=1)
    yy = data.maxwell(np.sqrt(xx), T)
    yy = np.log(yy)
    axes.plot(xx, yy, markersize=1, linewidth=1)
    axes.grid()
    plot.savefig(f"plots/ln2plots/plot{i}.png")
    plot.gca()

for i in range(len(data.frames)):
    plot_ln2(i)

def plot_1(i):
    T = data.frames[i].temperature

    fig = plot.figure(figsize=(6,5))
    axes = plot.axes()
    axes.set_xlabel("$v$")
    axes.set_ylabel("$dn/dv$")
    axes.set_title(f"{i * data.skip} iterations")

    xx, yy = data.frames[i].get_distribution()
    xx = xx[:200]
    yy = yy[:200] / data.n / data.hist_width

    axes.plot(xx, yy, '.', markersize=1)
    yy = data.maxwell(xx, T)
    yy = yy
    axes.plot(xx, yy, markersize=1, linewidth=1)
    axes.grid()
    plot.savefig(f"plots/plots/1plot{i}.png")
    plot.gca()

for i in range(len(data.frames)):
    plot_1(i)
