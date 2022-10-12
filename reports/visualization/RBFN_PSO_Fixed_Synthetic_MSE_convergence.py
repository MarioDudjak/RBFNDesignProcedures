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
no_k_values = 9
runs = 30
iterations_auto = 10000
iterations = 100
best = {}
best3 = {}
best4 = {}
best5 = {}
datasets = ['IR-0.85-DS-500-SD-2-CO-0-N-0']
for dataset in datasets:
    best[dataset] = []
    best3[dataset] = []
    best4[dataset] = []
    best5[dataset] = []

    filename = 'RBFN-PSO-Automatic-' + str(iterations_auto) + '-Search-' + dataset + "-Performances"
    filename3 = 'Incremental-Fixed-' + str(iterations_auto) + "-" + dataset + "-Performances"
    filename4 = 'RBFN-PSO-Incremental-' + str(iterations) + '-Search-' + dataset + "-Performances"
    filename5 = 'RBFN-PSO-Incremental-Smart-' + str(iterations) + '-Search-' + dataset + "-Performances"

    header, data = CsvProcessor().read_file(filename='results/' + filename)
    header3, data3 = CsvProcessor().read_file(filename='results/' + filename3)
    header4, data4 = CsvProcessor().read_file(filename='results/' + filename4)
    header5, data5 = CsvProcessor().read_file(filename='results/' + filename5)

    if header is not None and data is not None and data:
        data = np.array(data[0][0:-1], dtype=float)
        best[dataset] = data

    if header3 is not None and data3 is not None:
        data3 = [row[:-1] for row in data3 if row]
        data3 = np.array(data3, dtype=float)
        best3[dataset] = data3.flatten()[0:iterations_auto * no_k_values]

    if header4 is not None and data4 is not None:
        data4 = [row[0:-1] for row in data4 if row]
        data4 = np.array(data4, dtype=float)
        best4[dataset] = data4.flatten()[0:iterations * no_k_values]

    if header5 is not None and data5 is not None:
        data5 = [row[0:-1] for row in data5 if row]
        data5 = np.array(data5, dtype=float)
        best5[dataset] = data5.flatten()[0:iterations * no_k_values]

    plt.xlabel(r'Iterations (x${10^4}$)', fontsize=13)
    plt.xlim([0, no_k_values])
    # plt.plot(best[dataset], lw=0.75, ms=4, color=colors[0])
    plt.plot(best3[dataset], lw=0.75, ms=4, color=colors[3])
    # plt.plot(best4[dataset], lw=0.75, ms=4, color=colors[4])
    # plt.plot(best5[dataset], lw=0.75, ms=4, color=colors[5])
    plt.ylabel('MSE', fontsize=13)
    plt.xticks(np.arange(0, no_k_values * iterations_auto, step=iterations_auto),
               [str(i) for i in range(0, no_k_values)])
    plt.tick_params()
    plt.title('RBFN-PSO-' + str(iterations_auto) + '-' + '-'.join(dataset.split('_')))
    legend = [r"PSO-Fixed"]
    plt.legend(labels=legend, fancybox=False, framealpha=0.9)
    plt.tight_layout()
    plt.grid(b=True, linestyle=':')
    plt.show()

    # plt.savefig('_'.join([dataset, 'RBFN_PSO_MSE_convergence', str(iterations)]) + ".pdf", format='pdf', dpi=300)
    plt.close()
