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
iterations = 500

best = {}
best1 = {}
best2 = {}
best3 = {}
best4 = {}
best5 = {}
best6 = {}
best7 = {}
best8 = {}
best9 = {}
# datasets = ['ionosphere', 'imageseg', 'wine', 'parkinsons', 'glass', 'heart_disease', 'banknote', 'liver', 'statlog',
#             'vowel', 'pima', 'yeast']
datasets = ['banknote']

for dataset in datasets:
    best[dataset] = []
    best1[dataset] = []
    best2[dataset] = []
    best3[dataset] = []
    best4[dataset] = []
    best5[dataset] = []
    best6[dataset] = []
    best7[dataset] = []
    best8[dataset] = []
    best9[dataset] = []

    filename = 'RBFN-PSO-Fixed--Search-' + dataset
    filename1 = 'RBFN-PSO-Incremental-Random1-' + str(iterations) + '-Search-' + dataset + '-CSharp'
    filename2 = 'RBFN-PSO-Incremental-Random2-' + str(iterations) + '-Search-' + dataset + '-CSharp'
    filename3 = 'RBFN-PSO-Incremental-Kmeans1-' + str(iterations) + '-Search-' + dataset + '-CSharp'
    filename4 = 'RBFN-PSO-Incremental-Kmeans2-' + str(iterations) + '-Search-' + dataset + '-CSharp'
    filename5 = 'RBFN-PSO-Incremental-KmeansFixed1-' + str(iterations) + '-Search-' + dataset + '-CSharp'
    filename6 = 'RBFN-PSO-Incremental-KmeansFixed2-' + str(iterations) + '-Search-' + dataset + '-CSharp'
    filename7 = 'RBFN-PSO-Incremental-Traditional1-' + str(iterations) + '-Search-' + dataset + '-CSharp'
    filename8 = 'RBFN-PSO-Incremental-Mixed1-' + str(iterations) + '-Search-' + dataset + '-CSharp'
    filename9 = 'RBFN-PSO-Incremental-Smart1-' + str(iterations) + '-Search-' + dataset
    filename10 = 'RBFN-PSO-Automatic--Search-' + dataset

    header, data = CsvProcessor().read_file(filename='results/' + filename)
    header1, data1 = CsvProcessor().read_file(filename='results/' + filename1)
    header2, data2 = CsvProcessor().read_file(filename='results/' + filename2)
    header3, data3 = CsvProcessor().read_file(filename='results/' + filename3)
    header4, data4 = CsvProcessor().read_file(filename='results/' + filename4)
    header5, data5 = CsvProcessor().read_file(filename='results/' + filename5)
    header6, data6 = CsvProcessor().read_file(filename='results/' + filename6)
    header7, data7 = CsvProcessor().read_file(filename='results/' + filename7)
    header8, data8 = CsvProcessor().read_file(filename='results/' + filename8)
    header9, data9 = CsvProcessor().read_file(filename='results/' + filename9)
    header10, data10 = CsvProcessor().read_file(filename='results/' + filename10)

    if header is not None and data is not None:
        data = [row for row in data if row]
        data = data[1:-1]
        for i in range(no_k_values):
            best_median_scores = [float(row[-3]) for j, row in enumerate(data) if j % no_k_values == i]
            best[dataset].append(np.median(best_median_scores))

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

    if header5 is not None and data5 is not None:
        data5 = [row for row in data5 if row]
        data5 = data5[0:-1]
        for i in range(no_k_values):
            best_median_scores = [float(row[-1]) for j, row in enumerate(data5) if j % no_k_values == i]
            best5[dataset].append(np.median(best_median_scores))

    if header6 is not None and data6 is not None:
        data6 = [row for row in data6 if row]
        data6 = data6[0:-1]
        for i in range(no_k_values):
            best_median_scores = [float(row[-1]) for j, row in enumerate(data6) if j % no_k_values == i]
            best6[dataset].append(np.median(best_median_scores))

    if header7 is not None and data7 is not None:
        data7 = [row for row in data7 if row]
        data7 = data7[0:-1]
        for i in range(no_k_values):
            best_median_scores = [float(row[-1]) for j, row in enumerate(data7) if j % no_k_values == i]
            best7[dataset].append(np.median(best_median_scores))

    if header8 is not None and data8 is not None:
        data8 = [row for row in data8 if row]
        data8 = data8[0:-1]
        for i in range(no_k_values):
            best_median_scores = [float(row[-1]) for j, row in enumerate(data8) if j % no_k_values == i]
            best8[dataset].append(np.median(best_median_scores))

    if header9 is not None and data9 is not None:
        data9 = [row for row in data9 if row]
        data9 = data9[1:-1]
        for i in range(no_k_values):
            best_median_scores = [float(row[-3]) for j, row in enumerate(data9) if j % no_k_values == i]
            best9[dataset].append(np.median(best_median_scores))

    if header10 is not None and data10 is not None:
        data10 = [row for row in data10 if row]
        data10 = data10[1:-1]
        automatic_F1 = [float(row[-3]) for row in data10]
        automatic_k = [float(row[-2]) for row in data10]
        automatic_F1 = np.median(automatic_F1)
        automatic_k = int(np.median(automatic_k))
        automatic_F1 = [automatic_F1 for i in range(19)]
    else:
        automatic_k = 10
        automatic_F1 = [1 for i in range(19)]

    # max_y = np.max([np.max(best1[dataset]), np.max(best2[dataset]), np.max(best3[dataset]), np.max(best4[dataset])])
    plt.xticks(np.arange(0, no_k_values, step=1), [str(i + 2) for i in range(0, no_k_values)])
    # plt.ylim([0, max_y + 0.05])
    plt.xlim([0, no_k_values - 1])
    plt.xlabel(r'Broj čvorova u skrivenom sloju ($k$)', fontsize=13)

    # plt.plot(best1[dataset], lw=0.75, ms=4, marker=markers[1], color=colors[1])
    # plt.plot(best2[dataset], lw=0.75, ms=4, marker=markers[2], color=colors[2])
    # plt.plot(best3[dataset], lw=0.75, ms=4, marker=markers[3], color=colors[3])
    # plt.plot(best4[dataset], lw=0.75, ms=4, marker=markers[4], color=colors[4])
    # plt.plot(best5[dataset], lw=0.75, ms=4, marker=markers[5], color=colors[5])
    # plt.plot(best6[dataset], lw=0.75, ms=4, marker=markers[6], color=colors[6])
    # plt.plot(best7[dataset], lw=0.75, ms=4, marker=markers[7], color=colors[7])
    #plt.plot(best8[dataset], lw=0.75, ms=4, marker=markers[8], color=colors[8])
    plt.plot(best[dataset], lw=0.75, ms=4, marker=markers[0], color='royalblue')
    plt.plot(automatic_F1, lw=0.75, ms=4, marker=markers[1], color='orangered')
    plt.plot(best9[dataset], lw=0.75, ms=4, marker=markers[9], color='limegreen')


    plt.ylabel('F1', fontsize=13)
    plt.tick_params()
    #plt.title('RBFN-PSO6-' + '-'.join(dataset.split('_')))
    # legend = [r"PSO-Incremental-Fixed-10000", r"PSO-Incremental-Random1", r"PSO-Incremental-Random2", r"PSO-Incremental-Kmeans1", r"PSO-Incremental-Kmeans2", r"PSO-Incremental-FixedKmeans1", r"PSO-Incremental-FixedKmeans2"]
    legend = [r"Jednostavni", r"Automatski ($k=" + str(automatic_k) + "$)", r"Predloženi"]
    #legend = [r"Jednostavni", r"Predloženi"]
    plt.legend(labels=legend, fancybox=False, framealpha=0.9)
    plt.tight_layout()
    plt.grid(b=True, linestyle=':')
    #plt.show()
    plt.savefig('_'.join([dataset, 'RBFN_PSO_Incremental_vs_Fixed_F1_test_vs_kappa']) + ".pdf", format='pdf', dpi=300)
    plt.close()
