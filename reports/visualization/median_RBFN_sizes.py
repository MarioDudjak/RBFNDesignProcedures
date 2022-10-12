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
iterations = 100000
# data_point_labels = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
k_values = [str(i) for i in range(2, 21)]
results = {}
best = {}

for dataset in DatasetProvider().get_processed_dataset_list():
    filename = 'RBFN-PSO-Automatic-' + str(iterations) + '-Values-' + dataset.name
    print("Processing file {0}".format(filename))
    header, data = CsvProcessor().read_file(filename='results/' + filename)
    if header is not None and data is not None:
        data = [row for row in data if row]
        data = np.array(data, dtype=float)
        results[filename] = np.median(data, axis=0)
        plt.plot(results[filename], lw=0.75, ms=4, marker=markers[0], color=colors[0])
        plt.ylabel('Size of best solution ($k$)', fontsize=13)
        plt.xlabel(r'Iterations', fontsize=13)
        plt.tick_params()
        plt.title('RBFN-DE-' + dataset.name)
        legend = [r"DE-Automatic"]
        plt.legend(labels=legend, fancybox=False, framealpha=0.9)
        plt.tight_layout()
        plt.grid(b=True, linestyle=':')
        plt.show()

        # plt.savefig(experiment_setup + "/" + '_'.join([classifier.name, optimiser_name, dataset]) + ".pdf", format='pdf', dpi=300)
        plt.close()
