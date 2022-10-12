import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

from src.utils.datasets import DatasetProvider
from src.utils.file_handling.processors import CsvProcessor

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
no_k_values = 19
runs = 30
iterations = 10000
random = {}
maxsse = {}
maxssemin = {}
nm = {}
nm2 = {}
nm3 = {}
nm0 = {}
pso = {}
datasets= [r'$\mathcal{D}_1$',r'$\mathcal{D}_2$',r'$\mathcal{D}_3$',r'$\mathcal{D}_4$',r'$\mathcal{D}_5$',r'$\mathcal{D}_6$',r'$\mathcal{D}_7$',r'$\mathcal{D}_8$',r'$\mathcal{D}_9$',r'$\mathcal{D}_{10}$',r'$\mathcal{D}_{11}$',r'$\mathcal{D}_{12}$']
better_than_random_MSE = [8.2,16.53,11,9.2,3.13,7.53,10.07,10.23,10.13,14.4,7.43,15.67]
better_than_me_MSE = [10.27,17.2,9.35,7.47,7.2,6.4,9.73,9.63,6.8,12.77,10.33,17.67]

better_than_random = [2,1.47,12.23,0.83,3.77,1.67,9.33,6,9.37,6.47,4,0.97]
better_than_me = [1.83,1.47,9.46,1.07,3.5,2.13,8.43,3.47,6.03,3.97,1.53,1.87]

better_than_fixed = [0.87,0.90,1.67,3.77,0.77,6.60,7.07,10.97,2.47,0.47,0.17]
better_than_de_a = [5.33,5.83,3.53,7.40,8.03,16.03,10.20,8.30,8.43,5.20,3.83]
better_than_pso_a = [7.10,11.23,10.47,6.47,10.47,8.93,11.57,19.00,7.90,11.43,9.90]

# plt.ylabel(r'\#boljih mreža', fontsize=18)
# #plt.plot(pso[dataset], lw=0.75, ms=4, marker=markers[3], color='red')
# plt.plot(better_than_random, lw=1, ms=4, marker='o', color='royalblue')
# plt.plot(better_than_me, lw=1, ms=4, marker='s', color='orange')
# #plt.yticks(np.arange(0, 1, 0.1))
# plt.xticks(np.arange(0, 12), datasets, fontsize=15)
# plt.yticks(fontsize=15)
# plt.xlim([0,11])
# plt.ylim(bottom=0)
# plt.tick_params()
# legend = [r"PSO\textsubscript{r}", r"PSO\textsubscript{e}"]
# plt.legend(labels=legend, fancybox=False, framealpha=0.9, prop={"size": 15})
# plt.tight_layout()
# plt.grid(b=True, linestyle=':')
# #plt.show()
# plt.savefig("RBFN-F1-BoljeMreze.pdf", format='pdf', dpi=300)
# plt.close()


plt.ylabel(r'\#boljih mreža', fontsize=18)
#plt.plot(pso[dataset], lw=0.75, ms=4, marker=markers[3], color='red')
plt.plot(better_than_fixed, lw=1, ms=4, marker='o', color='firebrick')
plt.plot(better_than_de_a, lw=1, ms=4, marker='s', color='darkgoldenrod')
plt.plot(better_than_pso_a, lw=1, ms=4, marker='s', color='darkblue')

#plt.yticks(np.arange(0, 1, 0.1))
plt.xticks(np.arange(0, 12), datasets, fontsize=15)
plt.yticks(fontsize=15)
plt.xlim([0,11])
plt.ylim(bottom=0)
plt.tick_params()
legend = [r"PSO\textsubscript{j}", r"DE\textsubscript{A}", r"PSO\textsubscript{A}"]
plt.legend(labels=legend, fancybox=False, framealpha=0.9, prop={"size": 15})
plt.tight_layout()
plt.grid(b=True, linestyle=':')
#plt.show()
plt.savefig("RBFN_F1_Better_All.pdf", format='pdf', dpi=300)
plt.close()
