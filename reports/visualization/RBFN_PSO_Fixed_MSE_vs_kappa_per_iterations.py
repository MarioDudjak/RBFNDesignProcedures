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
iterations1 = 1000
iterations2 = 500
iterations3 = 200
iterations4 = 100
runs = 30

best1 = {}
best2 = {}
best3 = {}
best4 = {}
datasets = ['ionosphere', 'imageseg', 'wine', 'parkinsons', 'glass', 'heart_disease', 'banknote', 'liver', 'statlog',
            'vowel']
# for dataset in DatasetProvider().get_processed_dataset_list():
for dataset in datasets:
    best1[dataset] = []
    best2[dataset] = []
    best3[dataset] = []
    best4[dataset] = []

    filename1 = 'RBFN-PSO-Fixed-' + str(iterations1) + '-Search-' + dataset + '-CSharp'
    filename2 = 'RBFN-PSO-Fixed-' + str(iterations2) + '-Search-' + dataset + '-CSharp'
    filename3 = 'RBFN-PSO-Fixed-' + str(iterations3) + '-Search-' + dataset + '-CSharp'
    filename4 = 'RBFN-PSO-Fixed-' + str(iterations4) + '-Search-' + dataset + '-CSharp'

    header1, data1 = CsvProcessor().read_file(filename='results/' + filename1)
    header2, data2 = CsvProcessor().read_file(filename='results/' + filename2)
    header3, data3 = CsvProcessor().read_file(filename='results/' + filename3)
    header4, data4 = CsvProcessor().read_file(filename='results/' + filename4)

    if header1 is not None and data1 is not None:
        data1 = [row for row in data1 if row]
        data1 = data1[0:-1]
        for i in range(no_k_values):
            best_median_scores = [float(row[-1]) for j, row in enumerate(data1) if j % no_k_values == i]
            best1[dataset].append(np.median(best_median_scores))

    if header2 is not None and data2 is not None:
        data2 = [row for row in data2 if row]
        data2 = data2[0:-1]
        for i in range(no_k_values):
            best_median_scores = [float(row[-1]) for j, row in enumerate(data2) if j % no_k_values == i]
            best2[dataset].append(np.median(best_median_scores))

    if header3 is not None and data3 is not None:
        data3 = [row for row in data3 if row]
        data3 = data3[0:-1]
        for i in range(no_k_values):
            best_median_scores = [float(row[-1]) for j, row in enumerate(data3) if j % no_k_values == i]
            best3[dataset].append(np.median(best_median_scores))

    if header4 is not None and data4 is not None:
        data4 = [row for row in data4 if row]
        data4 = data4[0:-1]
        for i in range(no_k_values):
            best_median_scores = [float(row[-1]) for j, row in enumerate(data4) if j % no_k_values == i]
            best4[dataset].append(np.median(best_median_scores))

    # max_y = np.max([np.max(best1[dataset]), np.max(best2[dataset]), np.max(best3[dataset]), np.max(best4[dataset])])
    plt.xticks(np.arange(0, no_k_values, step=1), [str(i + 2) for i in range(0, no_k_values)])

    # plt.ylim([0, max_y + 0.05])
    plt.xlim([0, no_k_values - 1])
    plt.xlabel(r'Number of centers ($\kappa$)', fontsize=13)
    plt.plot(best1[dataset], lw=0.75, ms=4, marker=markers[1], color=colors[1])
    plt.plot(best2[dataset], lw=0.75, ms=4, marker=markers[2], color=colors[2])
    plt.plot(best3[dataset], lw=0.75, ms=4, marker=markers[3], color=colors[3])
    plt.plot(best4[dataset], lw=0.75, ms=4, marker=markers[4], color=colors[4])
    plt.ylabel('MSE', fontsize=13)
    plt.tick_params()
    plt.title('RBFN-PSO-Fixed $(N=30, c1=c2=1.496, w=0.7289)$ -' + '-'.join(dataset.split('_')))
    legend = [r"PSO (iter=1000)", r"PSO (iter=500)", r"PSO (iter=200)", r"PSO (iter=100)"]
    plt.legend(labels=legend, fancybox=False, framealpha=0.9)
    plt.tight_layout()
    plt.grid(b=True, linestyle=':')
    #plt.show()
    plt.savefig('_'.join([dataset, 'RBFN_PSO_MSE_vs_kappa_multiple_iterations']) + ".pdf", format='pdf', dpi=300)
    plt.close()
