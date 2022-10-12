import copy
import time
from datetime import datetime

import numpy as np
import pandas as pd
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, \
    cohen_kappa_score, mean_squared_error, confusion_matrix
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.cluster import KMeans
from src.experiment.setup import experiment_setup
from src.models.classification.classifiers.RBFN.RBFN import RBFN

from src.utils.datasets import DatasetProvider
from src.utils.file_handling.processors import CsvProcessor


def main():
    processed_datasets = DatasetProvider().get_processed_dataset_list()
    for file in processed_datasets:
        print("Processing {0}. Started {1}".format(file.name, datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
        dataset = pd.read_csv(file.path, header=0, index_col=0)
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        no_classes = len(set(y))

        X_rest, X_test, y_rest, y_test = train_test_split(X, y, test_size=experiment_setup['test_size'],
                                                          random_state=experiment_setup['random_state'], stratify=y)

        rskf = RepeatedStratifiedKFold(n_splits=experiment_setup['parameter_tuning']['folds'],
                                       n_repeats=experiment_setup['parameter_tuning']['repeats'],
                                       random_state=experiment_setup['random_state'])

        tuning_results = {}
        best_tuning_result = 0
        best_k = 2
        for k in range(5, 21):
            tuning_scores = []
            for train_index, val_index in rskf.split(X_rest, y_rest):
                X_train, X_val = X_rest[train_index], X_rest[val_index]
                y_train, y_val = y_rest[train_index], y_rest[val_index]

                # pso_fixed = PSOFixed(X_train, y_train, k, population_size=30, w=0.724, c1=1.468, c2=1.468,
                #                      iterations=20)
                kmeans = KMeans(n_clusters=k, max_iter=500)

                performances = kmeans.fit(X_train)
                centers = kmeans.cluster_centers_
                widths = None
                rbfn = RBFN(centers=centers, widths=widths)
                sse = rbfn.fit(X_train, y_train)
                predictions = rbfn.predict(X_val, no_classes)
                if no_classes == 2:
                    roc_auc = roc_auc_score(y_test, rbfn.predict_proba(X_test, no_classes)[:, 1])
                else:
                    roc_auc = roc_auc_score(y_test, rbfn.predict_proba(X_test, no_classes), multi_class='ovr')

                tuning_scores.append(roc_auc)
            average_score = np.average(tuning_scores)

            if average_score > best_tuning_result:
                best_tuning_result = average_score
                best_k = k

            tuning_results[k] = average_score

        print(tuning_results)
        CsvProcessor().save_raw_results(filename="RBFN-KMeans-Tuning-" + file.name,
                                        header=list(tuning_results.keys()),
                                        data=list(tuning_results.values()))

        print("Best k for dataset {0} is {1}".format(file.name, best_k))
        for run in range(experiment_setup['runs']):
            X_rest, X_test, y_rest, y_test = train_test_split(X, y, test_size=experiment_setup['test_size'],
                                                              random_state=run, stratify=y)
            start_time = time.time()
            kmeans = KMeans(n_clusters=best_k, max_iter=500)
            performances = kmeans.fit(X_rest)
            centers = kmeans.cluster_centers_
            widths = None
            best_rbfn = RBFN(centers=centers, widths=widths)
            sse = best_rbfn.fit(X_rest, y_rest)
            train_duration = time.time() - start_time

            start_time = time.time()
            predictions = best_rbfn.predict(X_test)
            predict_duration = time.time() - start_time

            f_score = f1_score(y_test, predictions, average='macro')
            precision = precision_score(y_test, predictions, average='macro')
            recall = recall_score(y_test, predictions, average='macro')
            accuracy = accuracy_score(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            kappa = cohen_kappa_score(y_test, predictions)
            if no_classes == 2:
                roc_auc = roc_auc_score(y_test, best_rbfn.predict_proba(X_test, no_classes)[:, 1])
            else:
                roc_auc = roc_auc_score(y_test, best_rbfn.predict_proba(X_test, no_classes), multi_class='ovr')

            gmean = geometric_mean_score(y_test, predictions, average='macro')

            conf_matrix = confusion_matrix(y_test, predictions)
            tn, fp, fn, tp = conf_matrix.ravel()

            prediction_results = [best_k, accuracy, mse, kappa, roc_auc, f_score, gmean, precision, recall, tp, fp, fn,
                                  tp,
                                  train_duration,
                                  predict_duration]
            header = ['k', 'CAC', 'mse', 'kappa', 'AUC', 'F1', 'Gmean', 'Precision', 'Recall', 'tp', 'fp', 'fn', 'tp',
                      'train_duration',
                      'predict_duration']

            CsvProcessor().save_raw_results(filename="RBFN-Kmeans-Search-" + file.name,
                                            header=header,
                                            data=prediction_results)

            print(prediction_results)


if __name__ == "__main__":
    main()
