import numpy as np

from src.utils.file_handling.processors import CsvProcessor

datasets = ['ionosphere', 'imageseg', 'wine', 'parkinsons', 'glass', 'heart_disease', 'banknote', 'liver', 'statlog',
            'vowel']

iterations = 500
iterations_auto = 10000
no_k_values = 19
best = {}
best1 = {}
best2 = {}
best3 = {}
best4 = {}
best5 = {}
best6 = {}
best7 = {}
best8 = {}

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

    filename = 'RBFN-PSO-Fixed-' + str(iterations_auto) + '-Search-' + dataset + "-CSharp"
    filename1 = 'RBFN-PSO-Incremental-Random1-' + str(iterations) + '-Search-' + dataset + "-CSharp"
    filename2 = 'RBFN-PSO-Incremental-Random2-' + str(iterations) + '-Search-' + dataset + "-CSharp"
    filename3 = 'RBFN-PSO-Incremental-Kmeans1-' + str(iterations) + '-Search-' + dataset + "-CSharp"
    filename4 = 'RBFN-PSO-Incremental-Kmeans2-' + str(iterations) + '-Search-' + dataset + "-CSharp"
    filename5 = 'RBFN-PSO-Incremental-KmeansFixed1-' + str(iterations) + '-Search-' + dataset + "-CSharp"
    filename6 = 'RBFN-PSO-Incremental-KmeansFixed2-' + str(iterations) + '-Search-' + dataset + "-CSharp"
    filename7 = 'RBFN-PSO-Incremental-Traditional1-' + str(iterations) + '-Search-' + dataset + "-CSharp"
    filename8 = 'RBFN-PSO-Incremental-Mixed1-' + str(iterations) + '-Search-' + dataset + "-CSharp"

    header, data = CsvProcessor().read_file(filename='results/' + filename)
    header1, data1 = CsvProcessor().read_file(filename='results/' + filename1)
    header2, data2 = CsvProcessor().read_file(filename='results/' + filename3)
    header3, data3 = CsvProcessor().read_file(filename='results/' + filename3)
    header4, data4 = CsvProcessor().read_file(filename='results/' + filename4)
    header5, data5 = CsvProcessor().read_file(filename='results/' + filename5)
    header6, data6 = CsvProcessor().read_file(filename='results/' + filename6)
    header7, data7 = CsvProcessor().read_file(filename='results/' + filename7)
    header8, data8 = CsvProcessor().read_file(filename='results/' + filename8)

    if header is not None and data is not None:
        data = [row for row in data if row]
        data = data[0:-1]
        best[dataset] = [float(row[-3]) for j, row in enumerate(data)]

    if header1 is not None and data1 is not None:
        data1 = [row for row in data1 if row]
        data1 = data1[0:-1]
        best1[dataset] = [float(row[-3]) for j, row in enumerate(data1)]

    if header2 is not None and data2 is not None:
        data2 = [row for row in data2 if row]
        data2 = data2[0:-1]
        best2[dataset] = [float(row[-3]) for j, row in enumerate(data2)]

    if header3 is not None and data3 is not None:
        data3 = [row for row in data3 if row]
        data3 = data3[0:-1]
        best3[dataset] = [float(row[-3]) for j, row in enumerate(data3)]

    if header4 is not None and data4 is not None:
        data4 = [row for row in data4 if row]
        data4 = data4[0:-1]
        best4[dataset] = [float(row[-3]) for j, row in enumerate(data4)]

    if header5 is not None and data5 is not None:
        data5 = [row for row in data5 if row]
        data5 = data5[0:-1]
        best5[dataset] = [float(row[-3]) for j, row in enumerate(data5)]

    if header6 is not None and data6 is not None:
        data6 = [row for row in data6 if row]
        data6 = data6[0:-1]
        best6[dataset] = [float(row[-3]) for j, row in enumerate(data6)]

    if header7 is not None and data7 is not None:
        data7 = [row for row in data7 if row]
        data7 = data7[0:-1]
        best7[dataset] = [float(row[-3]) for j, row in enumerate(data7)]

    if header8 is not None and data8 is not None:
        data8 = [row for row in data8 if row]
        data8 = data8[0:-1]
        best8[dataset] = [float(row[-3]) for j, row in enumerate(data8)]

# CsvProcessor().save_summary_results(filename='RBFN-PSO-Fixed-Summary-Test',
#                                     header=[str(i) for i in range(2, no_k_values + 2)],
#                                     data=list(best.values()))
#
# CsvProcessor().save_summary_results(filename='RBFN-PSO-Incremental-Random1-Summary-Test',
#                                     header=[str(i) for i in range(2, no_k_values + 2)],
#                                     data=list(best1.values()))
#
# CsvProcessor().save_summary_results(filename='RBFN-PSO-Incremental-Random2-Summary-Test',
#                                     header=[str(i) for i in range(2, no_k_values + 2)],
#                                     data=list(best2.values()))
#
# CsvProcessor().save_summary_results(filename='RBFN-PSO-Incremental-Kmeans1-Summary-Test',
#                                     header=[str(i) for i in range(2, no_k_values + 2)],
#                                     data=list(best3.values()))
#
# CsvProcessor().save_summary_results(filename='RBFN-PSO-Incremental-Kmeans2-Summary-Test',
#                                     header=[str(i) for i in range(2, no_k_values + 2)],
#                                     data=list(best4.values()))
#
# CsvProcessor().save_summary_results(filename='RBFN-PSO-Incremental-KmeansFixed1-Summary-Test',
#                                     header=[str(i) for i in range(2, no_k_values + 2)],
#                                     data=list(best5.values()))
#
# CsvProcessor().save_summary_results(filename='RBFN-PSO-Incremental-KmeansFixed2-Summary-Test',
#                                     header=[str(i) for i in range(2, no_k_values + 2)],
#                                     data=list(best6.values()))
#
# CsvProcessor().save_summary_results(filename='RBFN-PSO-Incremental-Traditional1-Summary-Test',
#                                     header=[str(i) for i in range(2, no_k_values + 2)],
#                                     data=list(best7.values()))

CsvProcessor().save_summary_results(filename='RBFN-PSO-Incremental-Mixed1-Summary-Test',
                                    header=[str(i) for i in range(2, no_k_values + 2)],
                                    data=list(best8.values()))
