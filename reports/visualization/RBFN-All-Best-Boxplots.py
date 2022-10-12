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

datasets = ['Biodegradeable', 'Breastcancer', 'Climate', 'glass', 'Heart-statlog', 'HillValley', 'Ionosphere',
            'Uci_Muskv1', 'Uci_parkinsons',
            'Urbanland', 'Wine']
results = {}
for dataset in datasets:
    random[dataset] = []
    maxsse[dataset] = []
    maxssemin[dataset] = []
    nm[dataset] = []
    results[dataset] = []
    filename1 = 'RBFN-PSO-Fixed-' + str(iterations * 20) + '-Search-' + dataset
    filename2 = 'RBFN-DE-Automatic-' + str(iterations * 20) + '-Search-' + dataset
    filename3 = 'RBFN-PSO-Automatic-' + str(iterations * 20) + '-Search-' + dataset
    filename4 = 'RBFN-PSO-Incremental-NM1-' + str(iterations) + '-Search-' + dataset
    filename5 = 'RBFN-PSO-Fixed-Kmeans-' + str(iterations*20) + '-Search-' + dataset

    header1, data1 = CsvProcessor().read_file(filename='results/' + filename1)
    header2, data2 = CsvProcessor().read_file(filename='results/' + filename2)
    header3, data3 = CsvProcessor().read_file(filename='results/' + filename3)
    header4, data4 = CsvProcessor().read_file(filename='results/' + filename4)
    header5, data5 = CsvProcessor().read_file(filename='results/' + filename5)

    if header3 is not None and data3 is not None:
        data1 = [row for row in data1 if row]
        data1 = data1[1:]

        data2 = [row for row in data2 if row]
        data2 = data2[1:]

        data3 = [row for row in data3 if row]
        data3 = data3[1:]

        data4 = [row for row in data4 if row]
        data4 = data4[1:]

        data5_raw = [row for row in data5 if row]
        data5_raw = data5[1:]
        data5 = []
        data5.extend(data5_raw)
        data5.extend(data5_raw)
        data5.extend(data5_raw)


        best_fixeds = []
        best_fixed_ks = []
        best_fixed_tprs = []

        best_kmeans = []
        best_kmeans_ks = []
        best_kmeans_tprs = []

        best_nms = []
        best_nm_ks = []
        best_nms_tprs = []

        btrs = []
        btrkm = []
        btrde = []
        btrpso = []
        run = 0

        best_de_automatics = [float(row[-4]) for row in data2]
        best_de_automatics_tprs = [float(row[4]) for row in data2]
        best_de_automatics_ks = [float(row[-2]) for row in data2]
        best_pso_automatics = [float(row[-4]) for row in data3]
        best_pso_automatics_tprs = [float(row[4]) for row in data3]
        best_pso_automatics_ks = [float(row[-2]) for row in data3]

        for i in range(0, len(data1), no_k_values):
            best_fixed = max([float(row[-6]) for j, row in enumerate(data1) if j >= i and j < i + no_k_values])
            best_fixeds.append(best_fixed)
            best_fixed_tpr = [float(row[4]) for j, row in enumerate(data1) if j >= i and j < i + no_k_values][
                [float(row[-6]) for j, row in enumerate(data1) if j >= i and j < i + no_k_values].index(best_fixed)]
            best_random_k = [round(float(row[-6]), 2) for j, row in enumerate(data1) if
                             j >= i and j < i + no_k_values].index(round(best_fixed, 2)) + 2
            best_fixed_ks.append(best_random_k)
            best_fixed_tprs.append(best_fixed_tpr)

            best_kmean = max([float(row[-4]) for j,row in enumerate(data5) if j >= i and j < i + no_k_values])
            best_kmeans.append(best_kmean)
            best_kmean_tpr = [float(row[4]) for j,row in enumerate(data5) if j >= i and j < i +no_k_values][[float(row[-4]) for j,row in enumerate(data5) if j >= i and j < i + no_k_values].index(best_kmean)]
            best_kmeans_tprs.append(best_kmean_tpr)
            best_kmean_k = [round(float(row[-4]),2) for j,row in enumerate(data5) if j >= i and j < i + no_k_values].index(round(best_kmean,2)) + 2
            best_kmeans_ks.append(best_kmean_k)

            all_nm = [float(row[-7]) for j, row in enumerate(data4) if j >= i and j < i + no_k_values]
            best_nm = max(all_nm)
            best_nms.append(best_nm)
            best_nm_tpr = [float(row[4]) for j, row in enumerate(data4) if j >= i and j < i + no_k_values][
                all_nm.index(best_nm)]
            best_nm_k = [round(float(row[-7]), 2) for j, row in enumerate(data4) if
                         j >= i and j < i + no_k_values].index(round(best_nm, 2)) + 2
            best_nm_ks.append(best_nm_k)
            best_nms_tprs.append(best_nm_tpr)

            better_than_fixed = np.sum([score > best_fixed for score in all_nm])
            better_than_kmeans = np.sum([score > best_kmean for score in all_nm])
            better_than_de = np.sum([score > best_de_automatics[run] for score in all_nm])
            better_than_pso = np.sum([score > best_pso_automatics[run] for score in all_nm])

            btrs.append(better_than_fixed)
            btrkm.append(better_than_kmeans)
            btrde.append(better_than_de)
            btrpso.append(better_than_pso)

            run += 1

        data = []
        data.append(best_fixeds)
        data.append(best_kmeans)
        data.append(best_de_automatics)
        data.append(best_pso_automatics)
        data.append(best_nms)
        colors = ['lightblue'] * 5
        box = plt.boxplot(data, patch_artist=True)

        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)

        plt.ylabel(r'F1', fontsize=15)
        plt.xticks(range(1, 6), [r'JPI\textsubscript{PSO}', r'JPI\textsubscript{$k$-means}', r'API\textsubscript{DE}',r'API\textsubscript{PSO}', r'Predloženi'],
                   fontsize=10)
        plt.yticks(fontsize=11)
        plt.tick_params()
        plt.tight_layout()
        plt.grid(b=True, linestyle=':', alpha=0.9, linewidth=0.9)
        plt.show()
        plt.close()

        # results[dataset].append([dataset, round(np.average(best_fixeds), 2), round(np.std(best_fixeds), 2),
        #                          round(np.average(best_fixed_ks), 2), round(np.std(best_fixed_ks), 2),
        #                          round(np.average(best_fixed_tprs), 2), round(np.std(best_fixed_tprs), 2),
        #                          round(np.average(best_kmeans), 2), round(np.std(best_kmeans), 2),
        #                          round(np.average(best_kmeans_ks), 2), round(np.std(best_kmeans_ks), 2),
        #                          round(np.average(best_kmeans_tprs), 2), round(np.std(best_kmeans_tprs), 2),
        #                          round(np.average(best_de_automatics), 2), round(np.std(best_de_automatics), 2),
        #                          round(np.average(best_de_automatics_ks), 2), round(np.std(best_de_automatics_ks), 2),
        #                          round(np.average(best_de_automatics_tprs), 2), round(np.std(best_de_automatics_tprs), 2),
        #                          round(np.average(best_pso_automatics), 2), round(np.std(best_pso_automatics), 2),
        #                          round(np.average(best_pso_automatics_ks), 2), round(np.std(best_pso_automatics_ks), 2),
        #                          round(np.average(best_pso_automatics_tprs), 2), round(np.std(best_pso_automatics_tprs), 2),
        #                          round(np.average(best_nms), 2), round(np.std(best_nms), 2),
        #                          round(np.average(best_nm_ks), 2), round(np.std(best_nm_ks), 2),
        #                          round(np.average(best_nms_tprs), 2), round(np.std(best_nms_tprs), 2),
        #                          round(np.average(btrs), 2), round(np.average(btrkm), 2),round(np.average(btrde), 2), round(np.average(btrpso), 2)])


