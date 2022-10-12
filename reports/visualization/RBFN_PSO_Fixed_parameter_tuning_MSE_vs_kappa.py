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
iterations = 100
runs = 30

best = {}
best1 = {}
best2 = {}
best3 = {}
best4 = {}
best5 = {}

#datasets = ['ionosphere', 'imageseg', 'wine', 'parkinsons', 'glass', 'heart_disease', 'banknote', 'liver', 'statlog',
#            'vowel']
datasets = ['ionosphere']

best_params = [
    {
        'c1': 1,
        'c2': 2.5,
        'w': 0.4
    },
    {
        'c1': 1,
        'c2': 1,
        'w': 0.8
    },
    {
        'c1': 1.5,
        'c2': 2,
        'w': 0.5
    },
    {
        'c1': 1.5,
        'c2': 1,
        'w': 0.8
    },
    {
        'c1': 2,
        'c2': 1.5,
        'w': 0.6
    }]

# for dataset in DatasetProvider().get_processed_dataset_list():
for dataset in datasets:
    best[dataset] = []
    best1[dataset] = []
    best2[dataset] = []
    best3[dataset] = []
    best4[dataset] = []
    best5[dataset] = []

    filename1 = 'RBFN-PSO-Fixed-Parameters-' + dataset
    filename = 'RBFN-PSO-Fixed-' + str(iterations) + '-Search-' + dataset + '-CSharp'

    header, data = CsvProcessor().read_file(filename='results/' + filename)
    header_params, data_params = CsvProcessor().read_file(filename='results/' + filename1)
    legend = []
    if header is not None and data is not None:
        data = [row for row in data if row]
        data = data[0:-1]
        scores = [float(row[-1]) for j, row in enumerate(data) if j < no_k_values]
        best[dataset] = np.asarray(scores)
        legend.append(r"PSO-Fixed $(N=30, c1=c2=1.496, w=0.7289)$")

    if header_params is not None and data_params is not None:
        data1 = [float(row[-1]) for row in data_params if int(row[2]) == iterations and float(row[3]) == best_params[0]['c1'] and float(row[4]) == best_params[0]['c2'] and float(row[5]) == best_params[0]['w']]
        best1[dataset] = np.asarray(data1)
        legend.append(r"PSO-Fixed $(N=30, c1=" + str(best_params[0]['c1']) + ", c2=" + str(best_params[0]['c2']) + ', w=' + str(
        best_params[0]['w']) + ")$")

        data2 = [float(row[-1]) for row in data_params if
                 int(row[2]) == iterations and float(row[3]) == best_params[1]['c1'] and float(row[4]) ==
                 best_params[1]['c2'] and float(row[5]) == best_params[1]['w']]
        best2[dataset] = np.asarray(data2)
        legend.append(
            r"PSO-Fixed $(N=30, c1=" + str(best_params[1]['c1']) + ", c2=" + str(best_params[1]['c2']) + ', w=' + str(
                best_params[1]['w']) + ")$")
        data3 = [float(row[-1]) for row in data_params if
                 int(row[2]) == iterations and float(row[3]) == best_params[2]['c1'] and float(row[4]) ==
                 best_params[2]['c2'] and float(row[5]) == best_params[2]['w']]
        best3[dataset] = np.asarray(data3)
        legend.append(
            r"PSO-Fixed $(N=30, c1=" + str(best_params[2]['c1']) + ", c2=" + str(best_params[2]['c2']) + ', w=' + str(
                best_params[2]['w']) + ")$")
        data4 = [float(row[-1]) for row in data_params if
                 int(row[2]) == iterations and float(row[3]) == best_params[3]['c1'] and float(row[4]) ==
                 best_params[3]['c2'] and float(row[5]) == best_params[3]['w']]
        best4[dataset] = np.asarray(data4)
        legend.append(
            r"PSO-Fixed $(N=30, c1=" + str(best_params[3]['c1']) + ", c2=" + str(best_params[3]['c2']) + ', w=' + str(
                best_params[3]['w']) + ")$")
        data5 = [float(row[-1]) for row in data_params if
                 int(row[2]) == iterations and float(row[3]) == best_params[4]['c1'] and float(row[4]) ==
                 best_params[4]['c2'] and float(row[5]) == best_params[4]['w']]
        best5[dataset] = np.asarray(data5)
        legend.append(
            r"PSO-Fixed $(N=30, c1=" + str(best_params[4]['c1']) + ", c2=" + str(best_params[4]['c2']) + ', w=' + str(
                best_params[4]['w']) + ")$")


    # max_y = np.max([np.max(best3[dataset]), np.max(best4[dataset])])
    plt.xticks(np.arange(0, no_k_values, step=1), [str(i + 2) for i in range(0, no_k_values)])

    # plt.ylim([0, max_y + 0.05])
    plt.xlim([0, no_k_values - 1])
    plt.xlabel(r'Number of centers ($\kappa$)', fontsize=13)
    plt.plot(best[dataset], lw=0.75, ms=4, marker=markers[0], color=colors[0])
    plt.plot(best1[dataset], lw=0.75, ms=4, marker=markers[1], color=colors[1])
    plt.plot(best2[dataset], lw=0.75, ms=4, marker=markers[2], color=colors[2])
    plt.plot(best3[dataset], lw=0.75, ms=4, marker=markers[3], color=colors[3])
    plt.plot(best4[dataset], lw=0.75, ms=4, marker=markers[4], color=colors[4])
    plt.plot(best5[dataset], lw=0.75, ms=4, marker=markers[5], color=colors[5])
    plt.ylabel('MSE', fontsize=13)
    plt.tick_params()
    plt.title('RBFN-PSO-' + str(iterations) + '-' + '-'.join(dataset.split('_')))
    plt.legend(labels=legend, fancybox=False, framealpha=0.9)
    plt.tight_layout()
    plt.grid(b=True, linestyle=':')
    #plt.show()
    plt.savefig('_'.join([dataset, 'RBFN_PSO_Parameter_tuning_MSE_vs_kappa', str(iterations)]) + ".pdf", format='pdf', dpi=300)
    plt.close()
