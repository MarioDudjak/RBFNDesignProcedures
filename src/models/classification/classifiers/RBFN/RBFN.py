from scipy.spatial import distance as dst

from src.models.classification.classifiers.baseClassifier import BaseClassifier

import numpy as np


class RBFN(BaseClassifier):

    def __init__(self, centers, widths):
        super().__init__(RBFN, "RBFN")
        self.centers = centers
        if widths is None:
            self.widths = self._get_widths()
        else:
            self.widths = widths
        self.bias = 0

    def rbf(self, x, c, s):
        distance = self._l2_norm(x, c)
        distance *= distance
        distance /= -2 * s * s
        return np.exp(distance)

    def fit(self, X_train, y_train):
        y_train = np.asarray(y_train, dtype=int)
        no_classes = len(set(y_train))
        h = self._calculate_houtputs(X_train)
        d = np.zeros((len(X_train), no_classes))
        for i in range(len(X_train)):
            d[i][y_train[i]] = 1

        t = np.linalg.pinv(h)
        self.w = np.matmul(t, d)

        errors = []
        sse = 0
        max_sse = 0
        max_sse_idx = 0
        for i in range(len(X_train)):
            output_vector = np.zeros(no_classes)
            temp_sse = 0
            for j in range(no_classes):
                output_vector[j] = np.sum([h[i][z] * self.w[z][j] for z in range(len(self.centers))]) + self.bias
                temp_sse += (d[i][j] - output_vector[j]) ** 2
                sse += (d[i][j] - output_vector[j]) ** 2
            if temp_sse > max_sse:
                max_sse_idx = i
                max_sse = temp_sse
            errors.append(temp_sse)
        return sse

    def predict(self, X_test, no_classes=2):
        predictions = []
        for test_instance in X_test:
            output_vector = self._calculate_output(test_instance, no_classes)
            label = np.argmax(output_vector)
            predictions.append(label)

        return np.asarray(predictions)

    def get_params(self):
        pass

    def predict_proba(self, X_test, no_classes=2):
        predictions = []
        for test_instance in X_test:
            output_vector = self._calculate_output(test_instance, no_classes)
            predictions.append(output_vector)

        return np.asarray(predictions)

    @staticmethod
    def _l2_norm(x1, x2):
        norm = [(x1[i] - x2[i]) ** 2 for i in range(len(x1))]
        return np.sqrt(np.sum(norm))

    def _get_widths(self):
        dmax = 0
        for i, center1 in enumerate(self.centers):
            distances = [self._l2_norm(center1, center2) for j, center2 in enumerate(self.centers) if i != j]
            tmp = np.max(distances)
            if i == 0 or dmax < tmp:
                dmax = tmp

        w = dmax / np.sqrt(2 * len(self.centers))
        widths = [w] * len(self.centers)
        return widths

    def _calculate_houtputs(self, X):
        hOut = np.zeros((len(X), len(self.centers)))
        for i in range(len(X)):
            hOut[i] = np.asarray([self.rbf(X[i], self.centers[j], self.widths[j]) for j in range(len(self.centers))])
            hsum = np.sum(hOut[i])
            hOut[i] = np.asarray(hOut[i] / hsum)
            for j in range(len(self.centers)):
                if np.isnan(hOut[i][j]):
                    hOut[i][j] = 0

        return hOut

    def _calculate_output(self, input_vector, no_classes):
        output_vector = np.zeros(no_classes)
        houtput = np.asarray(
            [self.rbf(input_vector, self.centers[i], self.widths[i]) for i in range(len(self.centers))])
        hsum = np.sum(houtput)
        houtput = houtput / hsum
        for j in range(len(self.centers)):
            if np.isnan(houtput[j]):
                houtput[j] = 0

        for i in range(no_classes):
            output_vector[i] = np.sum([houtput[j] * self.w[j][i] for j in range(len(self.centers))]) + self.bias

        return output_vector

    def calculate_sse(self, X_train, y_train):
        y_train = np.asarray(y_train, dtype=int)
        sse = 0
        no_classes = len(set(y_train))
        for i in range(len(X_train)):
            output_vector = self._calculate_output(X_train[i], no_classes)
            desired_output = np.zeros(no_classes)
            desired_output[y_train[i]] = 1

            for j in range(no_classes):
                sse += (desired_output[j] - output_vector[j]) ** 2

        return sse


class RBFN2(BaseClassifier):

    def __init__(self, centers, widths):
        super().__init__(RBFN, "RBFN")
        self.centers = centers
        self.kappa = len(centers)
        if widths is None:
            self.widths = self._get_widths()
        else:
            self.widths = widths
        self.bias = 0

    def rbf(self, x, c, s):
        y = dst.euclidean(x, c)
        y *= y
        y /= -2 * s * s
        y = np.exp(y)
        return y

    def fit(self, X_train, y_train):
        y_train = np.asarray(y_train, dtype=int)
        no_classes = len(set(y_train))
        self.w = np.zeros((self.kappa, no_classes))
        h = self._calculate_houtputs(X_train)
        d = np.zeros((len(X_train), no_classes))
        for i in range(len(X_train)):
            d[i][y_train[i]] = 1

        t = np.linalg.pinv(h)
        self.w = np.matmul(t, d)

        sse = self.calculate_sse(X_train, y_train)
        return sse

    def predict(self, X_test, no_classes=2):
        predictions = []
        for test_instance in X_test:
            output_vector = self._calculate_output(test_instance, no_classes)
            label = np.argmax(output_vector)
            predictions.append(label)

        return np.asarray(predictions)

    def get_params(self):
        pass

    def predict_proba(self, X_test, no_classes=2):
        predictions = []
        for test_instance in X_test:
            output_vector = self._calculate_output(test_instance, no_classes)
            predictions.append(output_vector)

        return np.asarray(predictions)


    def _get_widths(self):
        dmax = 0
        for i, center1 in enumerate(self.centers):
            distances = [dst.euclidean(center1, center2) for j, center2 in enumerate(self.centers) if i != j]
            tmp = np.max(distances)
            if i == 0 or dmax < tmp:
                dmax = tmp

        w = dmax / np.sqrt(2 * len(self.centers))
        widths = [w] * len(self.centers)
        return widths

    def _calculate_houtputs(self, X):
        hOut = np.zeros((len(X), len(self.centers)))
        for i in range(len(X)):
            hsum = 0
            for j in range(self.kappa):
                hOut[i][j] = self.rbf(X[i], self.centers[j], self.widths[j])
                hsum += hOut[i][j]

            for j in range(self.kappa):
                hOut[i][j] /= hsum
                if np.isnan(hOut[i][j]):
                    hOut[i][j] = 0

        return hOut

    def _calculate_output(self, input_vector, no_classes):
        output_vector = np.zeros(no_classes)
        hsum = 0
        houtputs = np.zeros(self.kappa)
        for i in range(self.kappa):
            houtputs[i] = self.rbf(input_vector, self.centers[i], self.widths[i])
            hsum += houtputs[i]

        for i in range(self.kappa):
            houtputs[i] /= hsum
            if np.isnan(houtputs[i]):
                houtputs[i] = 0

        for i in range(no_classes):
            for j in range(self.kappa):
                output_vector[i] += houtputs[j] * self.w[j][i]
            output_vector[i] += self.bias

        return output_vector

    def calculate_sse(self, X_train, y_train):
        y_train = np.asarray(y_train, dtype=int)
        sse = 0
        no_classes = len(set(y_train))
        for i in range(len(X_train)):
            output_vector = self._calculate_output(X_train[i], no_classes)
            desired_output = np.zeros(no_classes)
            desired_output[y_train[i]] = 1

            for j in range(no_classes):
                sse += (desired_output[j] - output_vector[j]) ** 2

        return sse

