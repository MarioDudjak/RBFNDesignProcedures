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

plt.rc('text.latex', preamble=r'\usepackage[croatian]{babel} \usepackage[utf8]{inputenc}')
markers = ['d', 'o', 'h', '*', 'P', 'p', 's', 'v', '>', '<', 'd', 'o', 'h', '*', 'P', 'p', 's', 'v', '>', '<', 'd', 'o',
           'h', '*', 'P', 'p', 's', 'v', '>', '<']
colors = ['blue', 'green', 'red', 'cyan', 'orange', 'purple', 'brown', 'olive', 'gray', 'pink', 'blue', 'green', 'red',
          'cyan', 'orange', 'purple', 'brown', 'olive', 'gray', 'pink']
no_k_values = 19
runs = 30
iterations = 10000


datasets = ['Biodegradeable', 'Breastcancer', 'Clean2', 'Climate', 'glass', 'Heart-statlog', 'HillValley', 'Ionosphere',
            'Uci_Muskv1', 'Uci_parkinsons',
            'Urbanland', 'Wine']
datasetLabels = [r'$\mathcal{D}_1$', r'$\mathcal{D}_2$', r'$\mathcal{D}_3$', r'$\mathcal{D}_4$', r'$\mathcal{D}_5$', r'$\mathcal{D}_6$', r'$\mathcal{D}_7$', r'$\mathcal{D}_8$', r'$\mathcal{D}_9$', r'$\mathcal{D}_{10}$', r'$\mathcal{D}_{11}$', r'$\mathcal{D}_{12}$']
#datasets = ['Biodegradeable', 'Breastcancer']
nm = []
pso_fixed = []
kmeans_fixed = []
de_a = []
pso_a = []
data = []
for dataset in datasets:
    print(dataset)
    filename1 = 'RBFN-PSO-Fixed-' + str(iterations * 20) + '-Search-' + dataset + "-Time"
    filename2 = 'RBFN-DE-Automatic-' + str(iterations) + '-Search-' + dataset + "-Time"
    filename3 = 'RBFN-PSO-Automatic-' + str(iterations) + '-Search-' + dataset + "-Time"
    filename4 = 'RBFN-PSO-Incremental-NM1-' + str(iterations) + '-Search-' + dataset + "-Time"
    filename5 = 'RBFN-PSO-Fixed-Kmeans-' + str(iterations*20) + '-Search-' + dataset + "-Time"

    header1, data1 = CsvProcessor().read_file(filename='results/' + filename1)
    header2, data2 = CsvProcessor().read_file(filename='results/' + filename2)
    header3, data3 = CsvProcessor().read_file(filename='results/' + filename3)
    header4, data4 = CsvProcessor().read_file(filename='results/' + filename4)
    header5, data5 = CsvProcessor().read_file(filename='results/' + filename5)

    if header3 is not None and data3 is not None:
        data1 = [row for row in data1 if row]
        data2 = [row for row in data2 if row]
        data3 = [row for row in data3 if row]
        data4 = [row for row in data4 if row]
        data5 = [row for row in data5 if row]

        nm1_dataset_times = []
        pso_fixed_dataset_times = []
        kmeans_fixed_dataset_times = []
        for i in range(0, no_k_values):
            time_list = data4[i][0].split(':')
            nm1_dataset_times.append(float(time_list[0]) * 3600 + float(time_list[1]) * 60 + float(time_list[2]))

            time_list = data1[i][0].split(':')
            pso_fixed_dataset_times.append(float(time_list[0]) * 3600 + float(time_list[1]) * 60 + float(time_list[2]))

            time_list = data5[i][0].split(':')
            kmeans_fixed_dataset_times.append(float(time_list[0]) * 3600 + float(time_list[1]) * 60 + float(time_list[2]))

        nm.append(sum(nm1_dataset_times) / 60)
        pso_fixed.append(sum(pso_fixed_dataset_times) / 60)
        kmeans_fixed.append(sum(kmeans_fixed_dataset_times) / 60)

        time_list = data2[0][0].split(':')
        de_a.append((float(time_list[0]) * 3600 + float(time_list[1]) * 60 + float(time_list[2])) / 60)

        time_list = data3[0][0].split(':')
        pso_a.append((float(time_list[0]) * 3600 + float(time_list[1]) * 60 + float(time_list[2])) / 60)


data.append(pso_fixed)
data.append(kmeans_fixed)
data.append(de_a)
data.append(pso_a)
data.append(nm)

plt.plot(pso_fixed, alpha=0.5, lw=0.75, ms=4, marker='d', color='firebrick')
plt.plot(kmeans_fixed,  alpha=0.5, lw=0.75, ms=4, marker='o', color='dodgerblue')
plt.plot(de_a, lw=0.75,  alpha=0.5, ms=4, marker='s', color='orange')
plt.plot(pso_a, lw=0.75,  alpha=0.5, ms=4, marker='p', color='crimson')
plt.plot(nm, lw=0.75,  alpha=0.5, ms=4, marker='D', color='limegreen')

#plt.hist(nm, color='royalblue')
#plt.ylabel(r'Broj skupova podataka', fontsize=15)
plt.ylabel(r'Trajanje izvođenja postupka (min; log. skala)', fontsize=15)

plt.yticks(fontsize=13)
plt.xticks(np.arange(0, len(datasetLabels), step=1), datasetLabels, fontsize=13)
plt.xlim(left=0, right=11)
plt.grid(b=True, linestyle=':', alpha=0.9, linewidth=0.9)
legend = [r"J\textsubscript{PSO}", r"J\textsubscript{$k$-means}", r"A\textsubscript{DE}", r"A\textsubscript{PSO}", r"Predloženi"]
plt.legend(labels=legend, fancybox=False, ncol=2, framealpha=0.9, prop={'size': 12})
plt.yscale('log')
plt.tick_params()
plt.tight_layout()
#plt.show()
plt.savefig('RBFN_trajanje.pdf', format='pdf', dpi=300)
plt.close()

# colors = ['lightblue'] * 5
# box = plt.boxplot(data, patch_artist=True)
#
# for patch, color in zip(box['boxes'], colors):
#     patch.set_facecolor(color)
#
# plt.ylabel(r'Trajanje izvođenja postupka (min)', fontsize=15)
# plt.xticks(range(1, 6), [r'J\textsubscript{PSO}', r'J\textsubscript{$k$-means}', r'A\textsubscript{DE}',r'A\textsubscript{PSO}', r'Predloženi'],
#            fontsize=10)
# plt.yticks(fontsize=11)
# plt.tick_params()
# plt.tight_layout()
# plt.grid(b=True, linestyle=':', alpha=0.9, linewidth=0.9)
# #plt.show()
# plt.savefig('RBFN_trajanje.pdf', format='pdf', dpi=300)
# plt.close()

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


