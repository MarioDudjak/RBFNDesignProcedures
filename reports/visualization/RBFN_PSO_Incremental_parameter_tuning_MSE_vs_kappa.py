import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

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
iterations = 500
runs = 30

best1 = {}
best2 = {}
best3 = {}
best4 = {}
best5 = {}

datasets = ['ionosphere', 'imageseg', 'wine', 'parkinsons', 'glass', 'heart_disease', 'banknote', 'liver', 'statlog',
           'vowel']


best_params = [
    {
        'c1': '1.468',
        'c2': '1.468',
        'w': '0.724',
        'name': 'PSO-6 (c1=c2=1.468, w=0.724)'
    },
{
        'c1': '1.49618',
        'c2': '1.49618',
        'w': '0.7298',
        'name': 'PSO-1 (c1=c2=1.49618, w=0.7298)'
    },
    {
        'c1': '1.331',
        'c2': '1.331',
        'w': '0.785',
        'name': 'PSO-7 (c1=c2=1.331, w=0.785)'
    },
    {
        'c1': '1.8',
        'c2': '1.8',
        'w': '0.6',
        'name': 'PSO-12 (c1=c2=1.8, w=0.6)'
    },
    {
        'c1': '1.7',
        'c2': '1.7',
        'w': '0.6',
        'name': 'PSO-3 (c1=c2=1.7, w=0.6)'
    }]

# for dataset in DatasetProvider().get_processed_dataset_list():
for dataset in datasets:
    best1[dataset] = []
    best2[dataset] = []
    best3[dataset] = []
    best4[dataset] = []
    best5[dataset] = []

    filename = 'RBFN-PSO-Incremental-Parameters-' + dataset

    header, data = CsvProcessor().read_file(filename='results/' + filename)

    legend = []

    if header is not None and data is not None:
        data1 = [float(row[-1]) for row in data if int(row[2]) == iterations and row[3] == best_params[0]['c1'] and row[4] == best_params[0]['c2'] and row[5] == best_params[0]['w']]
        best1[dataset] = np.asarray(data1)
        legend.append(best_params[0]['name'])

        data2 = [float(row[-1]) for row in data if
                 int(row[2]) == iterations and row[3] == best_params[1]['c1'] and row[4] == best_params[1]['c2'] and
                 row[5] == best_params[1]['w']]

        best2[dataset] = np.asarray(data2)
        legend.append(best_params[1]['name'])

        data3 = [float(row[-1]) for row in data if
                 int(row[2]) == iterations and row[3] == best_params[2]['c1'] and row[4] == best_params[2]['c2'] and
                 row[5] == best_params[2]['w']]
        best3[dataset] = np.asarray(data3)
        legend.append(best_params[2]['name'])

        data4 = [float(row[-1]) for row in data if
                 int(row[2]) == iterations and row[3] == best_params[3]['c1'] and row[4] == best_params[3]['c2'] and
                 row[5] == best_params[3]['w']]
        best4[dataset] = np.asarray(data4)
        legend.append(best_params[3]['name'])

        data5 = [float(row[-1]) for row in data if
                 int(row[2]) == iterations and row[3] == best_params[4]['c1'] and row[4] == best_params[4]['c2'] and
                 row[5] == best_params[4]['w']]
        best5[dataset] = np.asarray(data5)
        legend.append(best_params[4]['name'])

    # max_y = np.max([np.max(best3[dataset]), np.max(best4[dataset])])
    plt.xticks(np.arange(0, no_k_values, step=1), [str(i + 2) for i in range(0, no_k_values)])

    # plt.ylim([0, max_y + 0.05])
    plt.xlim([0, no_k_values - 1])
    plt.xlabel(r'Number of centers ($\kappa$)', fontsize=13)
    plt.plot(best1[dataset], lw=0.75, ms=4, marker=markers[1], color=colors[1])
    plt.plot(best2[dataset], lw=0.75, ms=4, marker=markers[2], color=colors[2])
    plt.plot(best3[dataset], lw=0.75, ms=4, marker=markers[3], color=colors[3])
    plt.plot(best4[dataset], lw=0.75, ms=4, marker=markers[4], color=colors[4])
    plt.plot(best5[dataset], lw=0.75, ms=4, marker=markers[5], color=colors[5])
    plt.ylabel('MSE', fontsize=13)
    plt.tick_params()
    plt.title('RBFN-PSO-Incremental-Random1-' + str(iterations) + '-' + '-'.join(dataset.split('_')))
    plt.legend(labels=legend, fancybox=False, framealpha=0.9)
    plt.tight_layout()
    plt.grid(b=True, linestyle=':')
    #plt.show()
    plt.savefig('_'.join([dataset, 'RBFN_PSO_Incremental_Parameter_tuning_MSE_vs_kappa', str(iterations)]) + ".pdf", format='pdf', dpi=300)
    plt.close()
