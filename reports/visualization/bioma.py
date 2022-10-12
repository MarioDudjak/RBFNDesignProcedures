import csv

import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

font = {'family': 'normal',
        'weight': 'normal',
        'size': 12}
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})

## for Palatino and other serif fonts use:
rc('font', **{'family': 'serif', 'serif': ['Palatino']})
rc('text', usetex=True)
markers = ['d', 'o', 'h', '*', 'P', 'p', 's', 'v', '>', '<', 'd', 'o', 'h', '*', 'P', 'p', 's', 'v', '>', '<', 'd', 'o',
           'h', '*', 'P', 'p', 's', 'v', '>', '<']
colors = ['blue', 'green', 'red', 'cyan', 'orange', 'purple', 'brown', 'olive', 'gray', 'pink', 'blue', 'green', 'red',
          'cyan', 'orange', 'purple', 'brown', 'olive', 'gray', 'pink']

optimisers = ['BPSO', 'SGA', 'TRO']
dimensionality = 30
runs = 50

for optimiser in optimisers:
    filepath = 'results/' + '_'.join([optimiser, 'RandomObjectiveFunction', 'results'])
    try:
        with open(filepath, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            data = []
            for row in reader:
                data.append(row)
    except:
        data = None

    solutions = []
    if data is not None:
        data = [row for row in data if row]
        solutions.append([int(el) for el in row[:-1]])

    barPlotData = []
    for i in range(dimensionality):
        barPlotData.append(np.sum([s[i] for s in solutions]))

    plt.ylim([0, runs])
    plt.xticks(np.arange(1, 31, 1), [r'$' + str(i + 1) + '$' for i in range(dimensionality)], fontsize=9)
    plt.yticks(np.arange(0, 51, 10), fontsize=10)
    plt.xlim([0.4, 30.6])
    plt.bar(range(1, dimensionality + 1), barPlotData)
    plt.show()
    plt.close()
