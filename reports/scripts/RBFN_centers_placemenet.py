import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from src.models.classification.classifiers.RBFN.centers_providers import KMeans, DEAutomatic, DEFixed
from src.models.classification.classifiers.RBFN.RBFN import RBFN
from src.experiment.setup import experiment_setup
from src.utils.file_handling.processors import CsvProcessor

markers = ['d', 'o', 'h', '*', 'P', 'p', 's', 'v', '>', '<']
colors = ['green', 'cyan', 'orange', 'purple', 'brown', 'olive', 'gray', 'pink', 'green', 'yellow']

path = r'C:\Users\MDudjak\Dropbox\Doktorski studij\Disertacija\Klasifikator-Doprinos\Experiment\data\synthetic'
# datasets = ['IR-0.5-DS-500-SD-2-CO-0-N-0.05', 'IR-0.5-DS-500-SD-3-CO-0-N-0', 'IR-0.9-DS-500-SD-3-CO-0-N-0', 'IR-0.85-DS-500-SD-2-CO-0-N-0', 'IR-0.85-DS-2000-SD-1-CO-0-N-0', 'IR-0.95-DS-100-SD-2-CO-0.1-N-0', 'IR-0.975-DS-2000-SD-2-CO-0-N-0', 'IR-0.975-DS-2000-SD-3-CO-0-N-0']
datasets = ['IR-0.85-DS-500-SD-2-CO-0-N-0']


def plot_decision_boundaries(X, y, ax, model_class, **model_params):
    """
    Function to plot the decision boundaries of a classification model.
    This uses just the first two columns of the data for fitting
    the model as we need to find the predicted value for every point in
    scatter plot.
    Arguments:
            X: Feature data as a NumPy-type array.
            y: Label data as a NumPy-type array.
            model_class: A Scikit-learn ML estimator class
            e.g. GaussianNB (imported from sklearn.naive_bayes) or
            LogisticRegression (imported from sklearn.linear_model)
            **model_params: Model parameters to be passed on to the ML estimator

    Typical code example:
            plt.figure()
            plt.title("KNN decision boundary with neighbros: 5",fontsize=16)
            plot_decision_boundaries(X_train,y_train,KNeighborsClassifier,n_neighbors=5)
            plt.show()
    """
    try:
        X = np.array(X)
        y = np.array(y).flatten()
    except:
        print("Coercing input data to NumPy arrays failed")
    # Reduces to the first two columns of data
    reduced_data = X[:, :2]
    # Instantiate the model object
    model = model_class(**model_params)
    # Fits the model with the reduced data
    model.fit(reduced_data, y)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02  # point in the mesh [x_min, m_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    # Meshgrid creation
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh using the model.
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # Predictions to obtain the classification results
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Plotting
    majority_indexes = np.where(y == 1)[0]
    minority_indexes = np.where(y == 0)[0]
    ax.scatter(X[majority_indexes, 0], X[majority_indexes, 1], cmap='RdBu', color='blue', alpha=0.8, label='positive')
    ax.scatter(X[minority_indexes, 0], X[minority_indexes, 1], cmap='RdBu', color='red', alpha=0.8, label='negative')

    ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdBu')

    # plt.xlabel("Feature 1", fontsize=15)
    # plt.ylabel("Feature 2", fontsize=15)
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    ax.legend(fancybox=False, framealpha=0.9)
    return max_sse_idx


def plot_centers(ax, centers, widths):
    # ax = ax.gca()
    x = np.asarray([center[0] for center in centers])
    y = np.asarray([center[1] for center in centers])
    for i in range(len(centers)):
        if i == len(centers) - 1:
            ax.scatter(x[i], y[i], c='black', marker='X', s=200)
        else:
            ax.scatter(x[i], y[i], c=colors[i], marker='X', s=100)

    i = 0
    for center, width in zip(centers, widths):
        if i == len(centers) - 1:
            circle = plt.Circle((center[0], center[1]), width, color='black', fill=False)
        else:
            circle = plt.Circle((center[0], center[1]), width, color=colors[i], fill=False)
        ax.add_artist(circle)
        i += 1


def plot_center(ax, center, width):
    x = center[0]
    y = center[1]
    ax.scatter(x, y, c='black', marker='X', s=100)

    circle = plt.Circle((center[0], center[1]), width, color='black', fill=False)
    ax.add_artist(circle)


for k in range(2, 11):
    for file in datasets:
        dataset = pd.read_csv(path + r"\\" + file + ".csv")
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1]

        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        except:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, )

        plt.figure()
        fig, ax = plt.subplots(nrows=1, ncols=1)

        centers_file = 'Incremental-Fixed-10000-' + file + '-Centers'
        centers_header, centers_data = CsvProcessor().read_file(filename='results/' + centers_file)
        if centers_header is not None and centers_data is not None:
            centers = []
            widths = []
            centers_data = [row for row in centers_data if row]
            for j in range(k):
                centers.append([float(center_coord) for center_coord in centers_data[k - 2][j * 2:j * 2 + 2]])
                widths.append(float(centers_data[k - 2][k * 2 + j]))

            centers = np.asarray(centers)
            widths = np.asarray(widths)
            print(centers)
            print(widths)
            max_sse_idx = RBFN(centers, widths).fit(X_train, y_train)
            centers = np.append(centers, [X_train[max_sse_idx]], axis=0)
            widths = np.append(widths, np.random.uniform(low=0.0, high=1.0))
            plot_decision_boundaries(X_train, y_train, ax, RBFN, centers=centers, widths=widths)
            plot_centers(ax, centers, widths)
            #plot_center(ax, X_train[max_sse_idx)
            ax.set_title('Traditional1-500')

        plt.suptitle('RBFN-PSO-k=' + str(k) + '-' + file)
        plt.tight_layout()
        # plt.savefig('RBFN-PSO-FixedvsIncremental-' + str(k) + '-' + file + '.pdf', dpi=300)
        plt.show()
        plt.close()
