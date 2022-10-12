import numpy as np

from src.utils.datasets import DatasetProvider
from src.experiment.setup import classifiers, classification_metrics
from src.utils.file_handling.processors import CsvProcessor

overall_results = {}
for file in DatasetProvider().get_processed_dataset_list():
    filename = 'RBFN-PSOFixed-Search-' + file.name
    print("Processing file {0}".format(filename))

    header, data = CsvProcessor().read_file(filename='results/' + filename)
    if header is not None and data is not None:
        data = [row for i, row in enumerate(data) if i % 2 == 0]
        data = np.array(data)
        results = {}
        col = 0
        for metric in header:
            results[metric] = sum([float(value) for value in data[:, col]]) / len(data[:, col])
            col = col + 1

        # CsvProcessor().save_summary_results(filename=filename, header=header,
        #                                     data=results.values())

        overall_results[file.name] = [float(value) for value in list(results.values())]


header = ['dataset'] + classification_metrics
for file in DatasetProvider().get_processed_dataset_list():
    if file.name in overall_results.keys():
        data = [str(file.name)] + overall_results[file.name]

        CsvProcessor().save_summary_results(filename='RBFN-PSOFixed', header=header,
                                        data=data)
