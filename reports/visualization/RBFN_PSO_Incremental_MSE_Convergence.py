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
iterations_auto = 10000
iterations = 500
best = {}
best1 = {}
best2 = {}
best3 = {}
best4 = {}
best5 = {}
best7 = {}
best8 = {}
best9 = {}
datasets = [ 'ionosphere', 'imageseg', 'wine', 'parkinsons', 'glass', 'heart_disease', 'banknote', 'liver', 'statlog',
            'vowel']
for dataset in datasets:
    best[dataset] = []
    best1[dataset] = []
    best2[dataset] = []
    best3[dataset] = []
    best4[dataset] = []
    best5[dataset] = []
    best8[dataset] = []
    best9[dataset] = []

    filename = 'RBFN-PSO-Automatic-' + str(iterations_auto) + '-Search-' + dataset + "-Performances"
    filename1 = 'RBFN-PSO-Incremental-Random1-' + str(iterations) + '-Search-' + dataset + "-Performances"
    filename2 = 'RBFN-PSO-Incremental-KmeansFixed1-' + str(iterations) + '-Search-' + dataset + "-Performances"

    filename3 = 'RBFN-PSO-Fixed-' + str(iterations) + '-Search-' + dataset + "-Performances"
    filename4 = 'RBFN-PSO-Incremental-' + str(iterations) + '-Search-' + dataset + "-Performances"
    filename7 = 'RBFN-PSO-Incremental-Traditional1-' + str(iterations) + '-Search-' + dataset + "-Performances"
    filename8 = 'RBFN-PSO-Incremental-Mixed1-' + str(iterations) + '-Search-' + dataset + "-Performances"
    filename9 = 'RBFN-PSO-Incremental-Smart1-' + str(iterations-100) + '100-Search-' + dataset + "-Performances"


    header, data = CsvProcessor().read_file(filename='results/' + filename)
    header1, data1 = CsvProcessor().read_file(filename='results/' + filename1)
    header2, data2 = CsvProcessor().read_file(filename='results/' + filename2)

    header3, data3 = CsvProcessor().read_file(filename='results/' + filename3)
    header4, data4 = CsvProcessor().read_file(filename='results/' + filename4)
    header7, data7 = CsvProcessor().read_file(filename='results/' + filename7)
    header8, data8 = CsvProcessor().read_file(filename='results/' + filename8)
    header9, data9 = CsvProcessor().read_file(filename='results/' + filename9)

    if header is not None and data is not None and data:
        data = np.array(data[0][0:-1], dtype=float)
        best[dataset] = data

    if header1 is not None and data1 is not None:
        data1 = [row[:-1] for row in data1 if row]
        data1 = data1[1:]
        data1 = np.array(data1, dtype=float)
        best1[dataset] = data1.flatten()[0:iterations * no_k_values]

    if header2 is not None and data2 is not None:
        data2 = [row[:-1] for row in data2 if row]
        data2 = data2[1:]
        data2 = np.array(data2, dtype=float)
        best2[dataset] = data2.flatten()[0:iterations * no_k_values]

    if header3 is not None and data3 is not None:
        data3 = [row[:-1] for row in data3 if row]
        data3 = data3[1:]
        data3 = np.array(data3, dtype=float)
        best3[dataset] = data3.flatten()[0:iterations * no_k_values]

    if header4 is not None and data4 is not None:
        data4 = [row[0:-1] for row in data4 if row]
        data4 = np.array(data4, dtype=float)
        best4[dataset] = data4.flatten()[0:iterations * no_k_values]

    if header7 is not None and data7 is not None:
        data7 = [row[0:-1] for row in data7 if row]
        data7 = data7[1:]
        data7 = np.array(data7, dtype=float)
        best7[dataset] = data7.flatten()[0:iterations * no_k_values]

    if header8 is not None and data8 is not None:
        data8 = [row[0:-1] for row in data8 if row]
        data8 = data8[1:]
        data8 = np.array(data8, dtype=float)
        best8[dataset] = data8.flatten()[0:iterations * no_k_values]

    if header9 is not None and data9 is not None:
        data9 = [row[0:-1] for row in data9 if row]
        data9 = data9[1:]
        data9 = np.array(data9, dtype=float)
        best9[dataset] = data9.flatten()[0:(iterations-100) * no_k_values]


    plt.xlabel(r'Iterations (x${5\cdot10^2}$)', fontsize=13)
    plt.xlim([0, no_k_values])
    #plt.plot(best[dataset], lw=0.75, ms=4, color=colors[0])
    plt.plot(best1[dataset], lw=0.75, ms=4, color=colors[1])
    #plt.plot(best2[dataset], lw=0.75, ms=4, color=colors[2])
    #plt.plot(best3[dataset], lw=0.75, ms=4, color=colors[3])
    #plt.plot(best4[dataset], lw=0.75, ms=4, color=colors[4])
    #plt.plot(best5[dataset], lw=0.75, ms=4, color=colors[5])
    plt.plot(best7[dataset], lw=0.75, ms=4, color=colors[7])
    plt.plot(best8[dataset], lw=0.75, ms=4, color=colors[8])
    plt.plot(best9[dataset], lw=0.75, ms=4, color=colors[9])

    plt.ylabel('MSE', fontsize=13)
    plt.xticks(np.arange(0, no_k_values * iterations, step=iterations), [str(i) for i in range(0, no_k_values)])
    plt.tick_params()
    plt.title('RBFN-PSO6-' + str(iterations) + '-' + '-'.join(dataset.split('_')))
    legend = [r"PSO-Incremental-Random1", r"PSO-Incremental-Traditional1", r"PSO-Incremental-Mixed1", r"PSO-Incremental-Premature1"]
    plt.legend(labels=legend, fancybox=False, framealpha=0.9)
    plt.tight_layout()
    plt.grid(b=True, linestyle=':')
    #plt.show()

    plt.savefig('_'.join([dataset, 'RBFN_PSO_Incremental_versions_MSE_convergence', str(iterations)]) + ".pdf", format='pdf', dpi=300)
    plt.close()
