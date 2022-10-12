import matplotlib.pyplot as plt
from matplotlib import rc

from src.utils.datasets import DatasetProvider
from src.utils.file_handling.processors import CsvProcessor
from src.experiment.setup import classifiers
from src.features.wrappers import fs_wrappers

font = {'family': 'normal',
        'weight': 'normal',
        'size': 12}
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})

## for Palatino and other serif fonts use:
rc('font', **{'family': 'serif', 'serif': ['Palatino']})
rc('text', usetex=True)


Incremental_random_F1 = [0.86,0.98,0.94,0.87,0.64,0.84,0.75,0.93,0.9,0.89,0.82,0.99]
Incremental_random_TPR = [0.87,0.99,0.96,0.93,0.66,0.85,0.78,0.94,0.9,0.89,0.84,0.99]

plt.scatter(Incremental_random_F1, Incremental_random_TPR, alpha=0.7, marker='o', color='royalblue')

Incremental_maxe_F1 = [0.86,0.98,0.95,0.86,0.64,0.84,0.75,0.95,0.91,0.91,0.83,0.99]
Incremental_maxe_TPR = [0.87,0.98,0.97,0.93,0.66,0.84,0.78,0.95,0.92,0.92,0.85,0.99]
plt.scatter(Incremental_maxe_F1, Incremental_maxe_TPR, alpha=0.7, marker='s', color='orange')

PSO_1NN_F1 = [0.86,0.98,0.97,0.86,0.65,0.83,0.78,0.95,0.93,0.92,0.83,0.99]
PSO_1NN_TPR = [0.87,0.98,0.99,0.93,0.69,0.84,0.81,0.95,0.93,0.93,0.85,0.99]

plt.scatter(PSO_1NN_F1, PSO_1NN_TPR, alpha=0.7, marker='D', color='limegreen')

labels = [r'GA', r'DE', r'PSO']
#plt.xlim([-0.1, 0.4])
#plt.ylim([-0.1, 0.4])
plt.xlabel(r'F1', fontsize=15)
plt.ylabel(r'TPR', fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tick_params()
plt.tight_layout()
plt.grid(b=True, which='both', color='grey', linewidth=1, linestyle='-', alpha=0.2)
plt.legend(labels=labels, fancybox=False, framealpha=0.9, ncol=3)
ax = plt.gca()
# ax.spines['top'].set_color('none')
# ax.spines['bottom'].set_position('zero')
# ax.spines['left'].set_position('zero')
# ax.spines['right'].set_color('none')

#ax.axhline(0, linestyle='-', linewidth=1.5, alpha=0.7, color='grey') # horizontal lines
#ax.axvline(0, linestyle='-', linewidth=1.5, alpha=0.7, color='grey') # vertical lines

plt.show()
#plt.savefig('F1_TPR_1NN.pdf', format='pdf', dpi=300)
plt.close()