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

datasets = ['Biodegradeable', 'Breastcancer', 'Clean2', 'Climate', 'glass', 'Heart-statlog', 'HillValley', 'Ionosphere', 'Uci_Muskv1', 'Uci_parkinsons',
    'Urbanland', 'Wine']
results = {}
for dataset in datasets:
    random[dataset] = []
    maxsse[dataset] = []
    maxssemin[dataset] = []
    nm[dataset] = []
    results[dataset] = []
    filename1 = 'RBFN-PSO-Random-' + str(iterations) + '-Search-' + dataset
    filename2 = 'RBFN-PSO-MaxSE-' + str(iterations) + '-Search-' + dataset
    filename3 = 'RBFN-PSO-Incremental-NM1-' + str(iterations) + '-Search-' + dataset


    header1, data1 = CsvProcessor().read_file(filename='results/' + filename1)
    header2, data2 = CsvProcessor().read_file(filename='results/' + filename2)
    header3, data3 = CsvProcessor().read_file(filename='results/' + filename3)

    if header3 is not None and data3 is not None:
        data1 = [row for row in data1 if row]
        data1 = data1[1:]

        data2 = [row for row in data2 if row]
        data2 = data2[1:]

        data3 = [row for row in data3 if row]
        data3 = data3[1:]

        best_randoms = []
        best_random_ks = []
        best_maxses = []
        best_maxse_ks = []
        best_nms = []
        best_nm_ks = []
        btrs = []
        btrmes = []
        for i in range(0, len(data1), no_k_values):
            l = [float(row[-3]) for j, row in enumerate(data1) if j>=i and j<i+no_k_values]
            best_random = min([float(row[-3]) for j, row in enumerate(data1) if j>=i and j<i+no_k_values])
            best_randoms.append(best_random)
            #best_random_k = [round(float(row[-7]),2) for j, row in enumerate(data1) if j>=i and j<i+no_k_values].index(round(best_random,2)) + 2
            #best_random_ks.append(best_random_k)
            best_maxse = min([float(row[-3]) for j, row in enumerate(data2) if j>=i and j<i+no_k_values])
            best_maxses.append(best_maxse)
            #best_maxse_k = [round(float(row[-7]),2) for j, row in enumerate(data2) if j>=i and j<i+no_k_values].index(round(best_maxse,2)) + 2
            #best_maxse_ks.append(best_maxse_k)
            all_nm = [float(row[-3]) for j, row in enumerate(data3) if j>=i and j<i+no_k_values]
            best_nm = min(all_nm)
            best_nms.append(best_nm)
            #best_nm_k = [round(float(row[-7]),2) for j, row in enumerate(data3) if j>=i and j<i+no_k_values].index(round(best_nm,2)) + 2
            #best_nm_ks.append(best_nm_k)

            better_than_random = np.sum([score<best_random for score in all_nm])
            better_than_maxse = np.sum([score<best_maxse for score in all_nm])
            btrs.append(better_than_random)
            btrmes.append(better_than_maxse)


        results[dataset].append([dataset, round(np.average(best_randoms),2), round(np.std(best_randoms),2), round(np.average(best_random_ks),2), round(np.std(best_random_ks),2), round(np.average(best_maxses),2), round(np.std(best_maxses),2), round(np.average(best_maxse_ks),2), round(np.std(best_maxse_ks),2), round(np.average(best_nms),2), round(np.std(best_nms),2), round(np.average(best_nm_ks),2), round(np.std(best_nm_ks),2),  round(np.average(btrs),2), round(np.average(btrmes),2)])






CsvProcessor().save_summary_results(
    filename="MSE_avg_test",
    header=['Dataset', 'better than random', 'better than maxse'],
    data=list(results.values()))