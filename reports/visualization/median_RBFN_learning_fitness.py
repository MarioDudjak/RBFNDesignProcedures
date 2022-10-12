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
# data_point_labels = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
k_values = [str(i) for i in range(2, 21)]
results = {}
best1 = {}
best2 = {}
best3 = {}
best4 = {}
automatic_results = {}
for dataset in DatasetProvider().get_processed_dataset_list():
    results[dataset.name] = []
    automatic_results[dataset.name] = []
    best1[dataset.name] = []
    best2[dataset.name] = []
    best3[dataset.name] = []
    best4[dataset.name] = []

    filename = 'RBFN-DEAutomatic-Search-' + dataset.name
    filename1 = 'RBFN-DEFixed-Search-' + dataset.name
    filename2 = 'RBFN-DEIncremental3-100-Search-' + dataset.name
    filename3 = 'RBFN-DEFixed3-Search-' + dataset.name
    filename4 = 'RBFN-DEFixed4-Search-' + dataset.name
    print("Processing file {0}".format(filename1))
    header, data = CsvProcessor().read_file(filename='results/' + filename)
    header1, data1 = CsvProcessor().read_file(filename='results/' + filename1)
    header2, data2 = CsvProcessor().read_file(filename='results/' + filename2)
    header3, data3 = CsvProcessor().read_file(filename='results/' + filename3)
    header4, data4 = CsvProcessor().read_file(filename='results/' + filename4)
    if header1 is not None and data1 is not None:
        data = [row for row in data if row]
        data = np.array(data, dtype=float)

        data1 = [row for row in data1 if row]
        data1 = np.array(data1, dtype=float)
        data2 = [row for row in data2 if row]

        data3 = [row for row in data3 if row]
        #data3 = np.array(data3, dtype=float)
        data4 = [row for row in data4 if row]
        #data4 = np.array(data4, dtype=float)

        for i in range(no_k_values):
            #results_across_runs = [row for j, row in enumerate(data1) if j % no_k_values == i]
            # median_scores = np.median(results_across_runs, axis=0)
            # results[dataset.name].append(median_scores)
            best_median_scores = [row[-1] for j, row in enumerate(data1) if j % no_k_values == i]
            best1[dataset.name].append(np.median(best_median_scores))
            best_median_scores = [float(row[-1]) for j, row in enumerate(data2) if j % no_k_values == i]
            best2[dataset.name].append(np.median(best_median_scores))
            best_median_scores = [float(row[-1]) for j, row in enumerate(data3) if j % no_k_values == i]
            best3[dataset.name].append(np.median(best_median_scores))
            best_median_scores = [float(row[-1]) for j, row in enumerate(data4) if j % no_k_values == i]
            best4[dataset.name].append(np.median(best_median_scores))

        # results[filename] = np.asarray(results[filename]).flatten()
        # best[dataset.name].append(np.median([row[-1] for row in enumerate(data1)]))

        # automatic_results[filename] = np.median(data2, axis=0)

    plt.xlim([2, 18])
    plt.xticks(range(19), k_values)
    # plt.xlabel(r'Number of centers ($k$)', fontsize=13)
    plt.plot(best1[dataset.name], lw=0.75, ms=4, marker=markers[1], color=colors[1])
    plt.plot(best2[dataset.name], lw=0.75, ms=4, marker=markers[3], color=colors[3])
    # plt.plot(best4[dataset.name], lw=0.75, ms=4, marker=markers[4], color=colors[4])
    # plt.plot(results[dataset.name], lw=0.75, ms=4, marker=markers[0], color=colors[0])
    # plt.plot(automatic_results[dataset.name], lw=0.75, ms=4, marker=markers[2], color=colors[2])
    plt.ylabel('MSE', fontsize=13)
    plt.xlabel(r'$kappa$', fontsize=13)
    plt.tick_params()
    plt.title('RBFN-DE-' + '-'.join(dataset.name.split('_')))
    legend = [r"DE-Incremental (constant iter)", r"DE-Incremental (rising iter)"]
    plt.legend(labels=legend, fancybox=False, framealpha=0.9)
    plt.tight_layout()
    plt.grid(b=True, linestyle=':')
    plt.show()

    #plt.savefig('_'.join([dataset.name, 'RBFN_DE_Incremental_MSE_vs_kappa']) + ".pdf", format='pdf', dpi=300)
    #plt.close()
