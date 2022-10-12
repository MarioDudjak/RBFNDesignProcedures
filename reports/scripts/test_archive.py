import numpy as np

from src.experiment.setup import experiment_setup, fitnesses, classifiers, classification_metrics
from src.features.wrappers import fs_wrappers
from src.models.classification.classificationProblem import ClassificationProblem
from src.utils.datasets import DatasetProvider
from src.utils.file_handling.processors import CsvProcessor

evaluation_procedures = ['_holdout', '3Fold', '5Fold']
population_size = 100


def main():
    datasets = DatasetProvider().get_processed_dataset_list()

    for fitness in fitnesses:
        for optimiser in fs_wrappers:
            for procedure in evaluation_procedures:
                for classifier in classifiers:
                    for dataset in datasets:

                        filename = '_'.join(
                            [fitness, procedure, optimiser.name + "_", classifier.name, dataset.name,
                             'archive'])
                        print("Processing file {0}".format(filename))

                        archive_header, archive_data = CsvProcessor().read_file(filename='logs/archive/' + filename)

                        if archive_header is not None and archive_data is not None:
                            candidates = [row for row in archive_data if row]

                            run = 0
                            index = 0
                            fitness_function = ClassificationProblem(dataset, classifier,
                                                                     random_state=42,  # dodati run + za variable
                                                                     test_size=experiment_setup["test_size"],
                                                                     validation_size=experiment_setup[
                                                                         "validation_size"],
                                                                     wrapper_fitness_metric=fitness,
                                                                     metrics=classification_metrics)
                            for idx, candidate in enumerate(candidates):
                                if idx > 0 and idx % population_size == 0:  # 100 je population size
                                    run += 1
                                    index = 0

                                    fitness_function = ClassificationProblem(dataset, classifier,
                                                                             random_state=42,
                                                                             # dodati run + za variable
                                                                             test_size=experiment_setup["test_size"],
                                                                             validation_size=experiment_setup[
                                                                                 "validation_size"],
                                                                             wrapper_fitness_metric=fitness,
                                                                             metrics=classification_metrics)

                                feature_vector = np.array(candidate).astype(bool)  # bool array from integer
                                output_quality = fitness_function.evaluate_final_solution(
                                    feature_vector)
                                validation_output_quality = fitness_function.evaluate_final_solution_on_validation(
                                    feature_vector)

                                output_quality['idx'] = index
                                index += 1

                                test_metrics = output_quality.keys()
                                test_metrics = [metric + "_test" for metric in test_metrics]
                                validation_metrics = validation_output_quality.keys()
                                validation_metrics = [metric + "_val" for metric in validation_metrics]

                                metrics = list(validation_metrics) + list(test_metrics)
                                results = list(validation_output_quality.values()) + list(output_quality.values())
                                CsvProcessor().save_log_results(filename='archive/' + filename + "_test",
                                                                header=metrics,
                                                                data=results)

    # for fitness in fitnesses:
    #     for optimiser in fs_wrappers:
    #         for procedure in evaluation_procedures:
    #             for classifier in classifiers:
    #                 for dataset in datasets:
    #
    #                     filename = '_'.join(
    #                         [fitness, procedure, optimiser.name + "_", classifier.name, dataset.name,
    #                          'population'])
    #                     print("Processing file {0}".format(filename))
    #
    #                     val_header, val_data = CsvProcessor().read_file(filename='logs/population/' + filename + "_fitness")
    #                     archive_header, archive_data = CsvProcessor().read_file(filename='logs/population/' + filename)
    #
    #                     if val_header is not None and val_data is not None:
    #                         archive_fitness = [row for row in val_data if row]
    #                         archive_fitness = np.array(archive_fitness)
    #
    #                     run = 0
    #                     if archive_header is not None and archive_data is not None:
    #                         candidates = [row for row in archive_data if row]
    #
    #                         for idx, candidate in enumerate(candidates):
    #                             if idx > 0 and idx % 100 == 0:  # 100 je population size
    #                                 run += 1
    #                             fitness_function = ClassificationProblem(dataset, classifier,
    #                                                                      random_state=run + 42,
    #                                                                      test_size=experiment_setup["test_size"],
    #                                                                      validation_size=experiment_setup[
    #                                                                          "validation_size"],
    #                                                                      wrapper_fitness_metric=fitness,
    #                                                                      metrics=classification_metrics)
    #
    #                             feature_vector = np.array(candidate).astype(bool)  # bool array from integer
    #                             output_quality = fitness_function.evaluate_final_solution(feature_vector)
    #                             validation_output_quality = fitness_function.evaluate_final_solution_on_validation(
    #                                 feature_vector)
    #
    #                             test_metrics = output_quality.keys()
    #                             test_metrics = [metric + "_test" for metric in test_metrics]
    #                             validation_metrics = validation_output_quality.keys()
    #                             validation_metrics = [metric + "_val" for metric in validation_metrics]
    #
    #                             metrics = list(validation_metrics) + list(test_metrics)
    #                             results = list(validation_output_quality.values()) + list(output_quality.values())
    #                             CsvProcessor().save_log_results(filename='population/' + filename + "_test",
    #                                                             header=metrics,
    #                                                             data=results)


if __name__ == "__main__":
    main()
