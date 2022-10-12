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
nm2 = {}
nm3 = {}
nm0 = {}
de = {}
pso = {}
pso_s = {}
datasets = ['Biodegradeable', 'Breastcancer', 'Climate', 'German_credit', 'Uci_Muskv1',
            'Ionosphere', 'Uci_imageseg', 'Uci_parkinsons', 'Uci_voice',  'Wine',  'Urbanland', 'Heart-statlog', 'HillValley', 'Hepatitis',   'Vote',
            'Connectionistbench', 'Wisconsin-breast-cancer', 'Clean2']
#datasets = ['Biodegradeable']


for dataset in datasets:
    random[dataset] = []
    maxsse[dataset] = []
    maxssemin[dataset] = []
    nm[dataset] = []
    nm2[dataset] = []
    nm3[dataset] = []
    nm0[dataset] = []
    de[dataset] = []
    pso[dataset] = []
    pso_s[dataset] = []

    filename1 = 'RBFN-PSO-Random-' + str(iterations) + '-Search-' + dataset
    filename2 = 'RBFN-PSO-MaxSE-' + str(iterations) + '-Search-' + dataset
    filename3 = 'RBFN-PSO-Incremental-NM1-' + str(iterations) + '-Search-' + dataset
    filename4 = 'RBFN-PSO-Incremental-NM1-BH-' + str(iterations) + '-Search-' + dataset
    filename5 = 'RBFN-PSO-MaxSEMin-' + str(iterations) + '-Search-' + dataset
    filename6 = 'RBFN-PSO-Incremental-NM1-NOBH-' + str(iterations) + '-Search-' + dataset
    filename7 = 'RBFN-PSO-Incremental-NM0-' + str(iterations) + '-Search-' + dataset

    filename8 = 'RBFN-DE-Automatic-' + str(20*iterations) + '-Search-' + dataset
    filename9 = 'RBFN-PSO-Automatic-' + str(20*iterations) + '-Search-' + dataset
    filename10 = 'RBFN-PSO-Fixed-' + str(20*iterations) + "-Search-" + dataset

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



    if header1 is not None and data1 is not None:
        data1 = [row for row in data1 if row]
        data1 = data1[1:]
        for i in range(no_k_values):
            best_avg_scores = [float(row[-8]) for j, row in enumerate(data1) if j % no_k_values == i]
            random[dataset].append(np.average(best_avg_scores))

    if header2 is not None and data2 is not None:
        data2 = [row for row in data2 if row]
        data2 = data2[1:]
        for i in range(no_k_values):
            best_avg_scores = [float(row[-8]) for j, row in enumerate(data2) if j % no_k_values == i]
            maxsse[dataset].append(np.average(best_avg_scores))

    if header3 is not None and data3 is not None:
        data3 = [row for row in data3 if row]
        data3 = data3[1:]
        for i in range(no_k_values):
            best_avg_scores = [float(row[-8]) for j, row in enumerate(data3) if j % no_k_values == i]
            nm[dataset].append(np.average(best_avg_scores))

    if header4 is not None and data4 is not None:
        data4 = [row for row in data4 if row]
        data4 = data4[1:]
        for i in range(no_k_values):
            best_avg_scores = [float(row[-8]) for j, row in enumerate(data4) if j % no_k_values == i]
            nm2[dataset].append(np.average(best_avg_scores))

    if header5 is not None and data5 is not None:
        data5 = [row for row in data5 if row]
        data5 = data5[1:]
        for i in range(no_k_values):
            best_avg_scores = [float(row[-8]) for j, row in enumerate(data5) if j % no_k_values == i]
            maxssemin[dataset].append(np.average(best_avg_scores))

    if header6 is not None and data6 is not None:
        data6 = [row for row in data6 if row]
        data6 = data6[1:]
        for i in range(no_k_values):
            best_avg_scores = [float(row[-8]) for j, row in enumerate(data6) if j % no_k_values == i]
            nm3[dataset].append(np.average(best_avg_scores))

    if header7 is not None and data7 is not None:
        data7 = [row for row in data7 if row]
        data7 = data7[1:]
        for i in range(no_k_values):
            best_avg_scores = [float(row[-8]) for j, row in enumerate(data7) if j % no_k_values == i]
            nm0[dataset].append(np.average(best_avg_scores))

    if header8 is not None and data8 is not None:
        data8 = [row for row in data8 if row]
        data8 = data8[1:]
        best_avg_scores = [float(row[-4]) for row in data8]
        de[dataset] = 20 * [np.average(best_avg_scores)]

    if header9 is not None and data9 is not None:
        data9 = [row for row in data9 if row]
        data9 = data9[1:]
        best_avg_scores = [float(row[-4]) for row in data9]
        pso[dataset] = 20 * [np.average(best_avg_scores)]

    if header10 is not None and data10 is not None:
        data10 = [row for row in data10 if row]
        data10 = data10[1:]
        for i in range(no_k_values):
            best_avg_scores = [float(row[-8]) for j, row in enumerate(data10) if j % no_k_values == i]
            pso_s[dataset].append(np.average(best_avg_scores))

    results = []
    for i in range(no_k_values):
        results.append([random[dataset][i], maxsse[dataset][i], nm[dataset][i]])

    CsvProcessor().save_summary_results(
        filename=dataset + "_TPR_avg_test",
        header=['Random', 'MaxSE', 'NM1'],
        data=results)
