import pandas as pd
import time
from datetime import datetime

import pandas as pd
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, \
    cohen_kappa_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import train_test_split

from src.experiment.setup import experiment_setup
from src.models.classification.classifiers.RBFN.RBFN import RBFN
from src.models.classification.classifiers.RBFN.centers_providers import KMeans, DEIncremental
from src.utils.datasets import DatasetProvider
from src.utils.file_handling.processors import CsvProcessor


def main():
    processed_datasets = DatasetProvider().get_processed_dataset_list()
    for file in processed_datasets:
        for run in range(experiment_setup['runs']):
            print("Processing {0}. Started {1}".format(file.name, datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
            dataset = pd.read_csv(file.path, header=0, index_col=0)
            X = dataset.iloc[:, :-1].values
            y = dataset.iloc[:, -1]
            try:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=experiment_setup['test_size'],
                                                                    random_state=run, stratify=y)
            except:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=experiment_setup['test_size'],
                                                                    random_state=run)

            no_classes = len(set(y))

            for k in range(2, 21):
                de_fixed = DEIncremental(0.9, 0.5, 5, 100, k, X_train, y_train)
                if k == 2:
                    kmeans = KMeans(k=k, max_iter=500, eps=0.001)
                    kmeans.run(X_train)
                    centers = kmeans.get_centers()
                    rbfn = RBFN(centers=centers, widths=None)
                    widths = rbfn._get_widths()

                start_time = time.time()
                performances, kvalues = de_fixed.run(centers=centers, widths=widths)
                centers = de_fixed.get_centers()
                widths = de_fixed.get_widths()
                rbfn = RBFN(centers=centers, widths=widths)
                rbfn.fit(X_train, y_train)
                train_duration = time.time() - start_time

                start_time = time.time()
                predictions = rbfn.predict(X_test)
                predict_duration = time.time() - start_time

                f_score = f1_score(y_test, predictions, average='macro')
                precision = precision_score(y_test, predictions, average='macro')
                recall = recall_score(y_test, predictions, average='macro')
                accuracy = accuracy_score(y_test, predictions)
                mse = mean_squared_error(y_test, predictions)
                kappa = cohen_kappa_score(y_test, predictions)
                if len(set(y)) == 2:
                    roc_auc = roc_auc_score(y, rbfn.predict_proba(X, no_classes)[:, 1])
                else:
                    roc_auc = roc_auc_score(y, rbfn.predict_proba(X, no_classes), multi_class='ovr')

                gmean = geometric_mean_score(y_test, predictions, average='macro')

                # conf_matrix = confusion_matrix(y_test, predictions)
                # tn, fp, fn, tp = conf_matrix.ravel()

                prediction_results = [accuracy, mse, kappa, roc_auc, f_score, gmean, precision, recall, train_duration,
                                      predict_duration]
                header = ['CAC', 'mse', 'kappa', 'AUC', 'F1', 'Gmean', 'Precision', 'Recall', 'train_duration',
                          'predict_duration']

                CsvProcessor().save_raw_results(filename="RBFN-Kmeans-" + str(k) + "-" + file.name, header=header,
                                                data=prediction_results)

                CsvProcessor().save_raw_results(filename="RBFN-DEFixed-Test-" + file.name, header=header,
                                                data=prediction_results)
                CsvProcessor().save_raw_results(filename="RBFN-DEFixed-Search-" + file.name, header=[str(j) for j in range(200)],
                                                data=performances)
                CsvProcessor().save_raw_results(filename="RBFN-Centers-" + file.name, header=[str(j) for j in range(len(centers))],
                                                data=centers)


if __name__ == "__main__":
    main()
