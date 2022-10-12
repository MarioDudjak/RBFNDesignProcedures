import numpy as np
import ast
import seaborn as sb
import matplotlib.pyplot as plt
from matplotlib import rc, cm

from src.utils.file_handling.processors import CsvProcessor
from pylab import *

datasets = ['ionosphere', 'imageseg', 'wine', 'parkinsons', 'glass', 'heart_disease', 'banknote', 'liver', 'statlog',
            'vowel']

params_coding_scheme = {
    "PSO": {
        "c1": [0.5, 1, 1.5, 2, 2.5],
        "c2": [0.5, 1, 1.5, 2, 2.5],
        "w": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    }
}

params_heatmap_shape = {
    "PSO": (5, 5, 6)
}

results = {}
for dataset in datasets:
    results[dataset] = {}
    filename = 'RBFN-PSO-Fixed-Parameters-' + dataset
    header, data = CsvProcessor().read_file(filename='results/' + filename)
    if header is not None and data is not None:
        best_params_count = {}
        data = [row for i, row in enumerate(data) if row]
        data = np.array(data)
        for row in data:
            best_params_combo = ','.join([row[3], row[4], row[5]])

            if best_params_combo not in results[dataset].keys():
                results[dataset][best_params_combo] = float(row[-1])
                best_params_count[best_params_combo] = 1
            else:
                results[dataset][best_params_combo] += float(row[-1])
                best_params_count[best_params_combo] += 1

        for best_params, params_score in results[dataset].items():
            results[dataset][best_params] = params_score / (best_params_count[best_params])

print(results)



# data = {}
# for alg_name, alg_best_params in results.items():
#     if alg_name == 'DT':
#         data[alg_name] = np.zeros(params_heatmap_shape[alg_name], float)
#         for params_string, param_count in alg_best_params.items():
#                 param_dict = ast.literal_eval(params_string)
#                 param_locations = []
#
#                 for param_key, param_value in param_dict.items():
#                     if param_key != 'gamma' and param_key != 'activation':
#                         param_locations.append(params_coding_scheme[alg_name][param_key].index(param_value))
#
#                 data[alg_name][param_locations[1], param_locations[0]] = param_count / (30 * 98)
#
#         print(data[alg_name])
#         sb.heatmap(data[alg_name], annot=True, cmap="BrBG", xticklabels=["Gini", "entropy"], yticklabels=params_coding_scheme[alg_name]["max_depth"])
#         plt.xlabel("Splitting criterion", fontsize=13)
#         plt.ylabel("Maximum tree depth", fontsize=13)
#         #plt.show()
#         plt.savefig("DTParametersMap.pdf", format='pdf', dpi=300)
#
#         plt.close()


data = {}
c1s = []
c2s = []
ws = []

for dataset_name, dataset_results in results.items():
    data[dataset_name] = []
    for best_params, params_score in dataset_results.items():
        param_list = [float(param_value) for param_value in best_params.split(',')]
        c1s.append(float(param_list[0]))
        c2s.append(float(param_list[1]))
        ws.append(float(param_list[2]))

        data[dataset_name].append(params_score)
    if data[dataset_name]:
        print(data[dataset_name])
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(c1s, c2s, ws, marker='s', s=50, c=data[dataset_name], cmap='YlGnBu')
        colmap = cm.ScalarMappable(cmap="BrBG")
        colmap.set_array(data[dataset_name])
        cb = fig.colorbar(colmap, pad=0.1)
        ax.set_xlabel('c1 (cognitive component)', fontsize=13)
        ax.set_ylabel('c2 (social component)', fontsize=13)
        ax.set_zlabel('w (intertia weight)', fontsize=13)
        plt.show()
        #plt.savefig("MLPParametersMap.pdf", format='pdf', dpi=300)
        plt.close()

