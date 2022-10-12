import copy
import time
import numpy as np
from scipy.spatial import distance as dst
from src.models.classification.classifiers.RBFN.RBFN import RBFN, RBFN2


class KMeans:

    def __init__(self, k, max_iter, eps):
        self.name = "KMeans"
        self.k = k
        self.max_iter = max_iter
        self.eps = eps
        self.centers = None
        self.sse = 0
        self.ssepc = self.dppc = None
        self.partition = None

    def run(self, X_train):
        size = len(X_train)
        dimensionality = len(X_train[0])
        self.centers = np.zeros((self.k, dimensionality))
        self.ssepc = np.zeros(self.k)
        self.dppc = np.zeros(self.k)
        self.partition = np.zeros(size)

        irnd = [i for i in range(size)]
        for i in range(self.k):
            r = np.random.randint(low=0, high=size)
            while irnd[r] == -1:
                r = np.random.randint(low=0, high=size)
            for j in range(dimensionality):
                self.centers[i][j] = X_train[irnd[r]][j]
            irnd[r] = -1
            # irnd.pop(r)

        for i in range(self.max_iter):
            prev_centers = self.centers.copy()

            self.Mdp(X_train, size, dimensionality)
            self.Uc(X_train, size, dimensionality)
            converged = True

            for j in range(self.k):
                for l in range(dimensionality):
                    if np.abs(prev_centers[j][l] - self.centers[j][l]) > self.eps:
                        converged = False
                        break
                if not converged:
                    break

            if converged:
                break

        self.Mdp(X_train, size, dimensionality)
        return self.sse

    def get_centers(self):
        return self.centers

    @staticmethod
    def _l2_norm(x1, x2):
        norm = [(x1[i] - x2[i]) ** 2 for i in range(len(x1))]
        return np.sqrt(np.sum(norm))

    def Mdp(self, X_train, size, dimensionality):
        self.sse = 0
        self.ssepc = np.zeros(self.k)
        self.dppc = np.zeros(self.k)

        for i in range(size):
            distances = [self._l2_norm(X_train[i], self.centers[j]) for j in range(self.k)]
            dmin = np.min(distances)
            c = np.argmin(distances)
            self.partition[i] = c
            self.dppc[c] += 1
            self.ssepc[c] += dmin * dmin

        for i in range(self.k):
            if self.dppc[i] == 0:
                ssemax = np.argmax(self.ssepc)
                fdp = 0
                dmax = 0
                for j in range(size):
                    if self.partition[j] == ssemax:
                        t = self._l2_norm(X_train[j], self.centers[ssemax])
                        if t > dmax:
                            fdp = j
                            dmax = t
                self.partition[fdp] = i
                self.dppc[i] += 1
                self.ssepc[ssemax] -= dmax * dmax
                for j in range(dimensionality):
                    self.centers[i][j] = X_train[fdp][j]

        for i in range(self.k):
            self.sse += self.ssepc[i]

    def Uc(self, X_train, size, dimensionality):
        self.centers = np.zeros((self.k, dimensionality))

        for i in range(size):
            for j in range(dimensionality):
                self.centers[int(self.partition[i])][j] += X_train[i][j]

        for i in range(self.k):
            for j in range(dimensionality):
                self.centers[i][j] /= self.dppc[i]


class DEAutomatic:

    def __init__(self, cr, f, population_size, iterations, kmin, kmax, X_train, y_train, strategy="RAND"):
        self.alg_name = "DE"
        self.cr = cr
        self.f = f
        self.strategy = strategy
        self.population_size = population_size
        self.iterations = iterations
        self.dimensionality = len(X_train[0])
        self.no_classes = len(set(y_train))
        self.X_train = X_train
        self.y_train = y_train
        self.lmbda = 0.01 / (2 * self.no_classes)
        self.kmin = kmin
        self.kmax = kmax
        self.kappa = kmax
        self.vsize = self.kappa * (self.dimensionality + 2)
        self.maxbound = 1
        self.minbound = 0
        self.current_population = self.new_population = self.current_population_fitness = self.new_population_fitness = self.current_population_sizes = self.new_population_sizes = None

    def run(self):
        performance = np.zeros(self.iterations)
        kvalues = np.zeros(self.iterations)

        self._initialise_population()
        self.max_cost, self.min_cost, self.avg_cost, self.max_idx, self.min_idx = self._evaluate_population()
        self.best_cost = self.current_population_fitness[self.min_idx]

        for i in range(self.iterations):
            print("Iteration: {}".format(i + 1))
            for j in range(self.population_size):
                self.new_population[j] = copy.deepcopy(self._generate_trail_vector(j))

            for j in range(self.population_size):
                self.new_population_fitness[j], self.new_population_sizes[j] = self._evaluate_candidate(
                    self.new_population[j])
                if self.new_population_fitness[j] <= self.current_population_fitness[j]:
                    self.current_population_fitness[j] = self.new_population_fitness[j]
                    self.current_population_sizes[j] = self.new_population_sizes[j]
                    self.current_population[j] = copy.deepcopy(self.new_population[j])
            self.max_cost, self.min_cost, self.avg_cost, self.max_idx, self.min_idx = self._evaluate_population()
            if i == 0 or self.current_population_fitness[self.min_idx] < self.best_cost:
                self.best_cost = self.current_population_fitness[self.min_idx]

            performance[i] = self.best_cost
            kvalues[i] = self.current_population_sizes[self.min_idx]

        return performance, kvalues

    def _initialise_population(self):
        self.current_population = np.zeros((self.population_size, self.vsize))
        self.new_population = np.zeros((self.population_size, self.vsize))

        self.current_population_fitness = np.zeros(self.population_size)
        self.new_population_fitness = np.zeros(self.population_size)

        self.current_population_sizes = np.zeros(self.population_size)
        self.new_population_sizes = np.zeros(self.population_size)

        for i in range(self.population_size):
            for j in range(self.vsize):
                self.current_population[i][j] = self.minbound + np.random.rand() * (
                        self.maxbound - self.minbound)
            self.current_population_fitness[i], self.current_population_sizes[i] = self._evaluate_candidate(
                self.current_population[i])

    def _evaluate_candidate(self, candidate):
        centers = self._activate_centers(candidate)
        widths = self._obtain_widths(candidate)
        size = len(widths)
        rbfn = RBFN(centers, widths)
        sse = rbfn.fit(self.X_train, self.y_train)
        return sse / len(self.X_train) / self.no_classes + self.lmbda * len(widths), size

    def _activate_centers(self, candidate):
        tmp = [i for i in range(self.kappa) if candidate[i] >= 0.5]

        while len(tmp) < self.kmin:
            r = np.random.randint(low=0, high=self.kappa)
            if r not in tmp:
                tmp.append(r)
                candidate[r] = 0.5 * (1 + np.random.rand())

        centers = np.zeros((len(tmp), self.dimensionality))
        for i in range(len(tmp)):
            for j in range(self.dimensionality):
                centers[i][j] = candidate[self.kappa + tmp[i] * self.dimensionality + j]

        return centers

    def _obtain_widths(self, candidate):
        tmp = [i for i in range(self.kappa) if candidate[i] >= 0.5]

        widths = np.zeros(len(tmp))
        for i in range(len(tmp)):
            widths[i] = candidate[self.kappa * (1 + self.dimensionality) + tmp[i]]

        return widths

    def _evaluate_population(self):
        max_cost = np.max(self.current_population_fitness)
        min_cost = np.min(self.current_population_fitness)
        max_idx = np.argmax(self.current_population_fitness)
        min_idx = np.argmin(self.current_population_fitness)
        avg_cost = np.average(self.current_population_fitness)

        return max_cost, min_cost, avg_cost, max_idx, min_idx

    def _generate_trail_vector(self, idx):
        new_candidate = copy.deepcopy(self.new_population[idx])
        r0 = np.random.randint(low=0, high=self.population_size)
        while r0 == idx:
            r0 = np.random.randint(low=0, high=self.population_size)

        r1 = np.random.randint(low=0, high=self.population_size)
        while r1 == idx or r1 == r0:
            r1 = np.random.randint(low=0, high=self.population_size)

        r2 = np.random.randint(low=0, high=self.population_size)
        while r2 == idx or r2 == r1 or r2 == r0:
            r2 = np.random.randint(low=0, high=self.population_size)

        randi = np.random.randint(low=0, high=self.vsize)

        for i in range(self.vsize):
            if np.random.rand() <= self.cr or i == randi:
                if self.strategy == "BEST":
                    diff = self.current_population[self.min_idx][i] + self.f * (
                                self.current_population[r1][i] - self.current_population[r2][i])
                elif self.strategy == "TBEST":
                    if self.current_population_fitness[r1] < self.current_population_fitness[r0]:
                        tmp = r0
                        r0 = r1
                        r1 = tmp

                    if self.current_population_fitness[r0] < self.current_population_fitness[r2]:
                        tmp = r0
                        r0 = r2
                        r2 = tmp
                    diff = self.current_population[r0][i] + self.f * (
                                self.current_population[r1][i] - self.current_population[r2][i])
                else:
                    diff = self.current_population[r0][i] + self.f * (
                            self.current_population[r1][i] - self.current_population[r2][i])

                if diff < self.minbound:
                    diff = self.current_population[r0][i] + np.random.rand() * (
                            self.minbound - self.current_population[r0][i])
                if diff > self.maxbound:
                    diff = self.current_population[r0][i] + np.random.rand() * (
                            self.maxbound - self.current_population[r0][i])

                new_candidate[i] = diff
            else:
                new_candidate[i] = self.current_population[idx][i]

        return new_candidate

    def get_centers(self):
        return self._activate_centers(self.current_population[self.min_idx])

    def get_widths(self):
        return self._obtain_widths(self.current_population[self.min_idx])


class DEIncremental:

    def __init__(self, cr, f, population_size, iterations, k, X_train, y_train, strategy="RAND"):
        self.alg_name = "DE"
        self.cr = cr
        self.f = f
        self.strategy = strategy
        self.population_size = population_size
        self.iterations = iterations
        self.dimensionality = len(X_train[0])
        self.no_classes = len(set(y_train))
        self.X_train = X_train
        self.y_train = y_train
        self.kappa = k
        self.vsize = self.kappa * (self.dimensionality + 1)
        self.maxbound = 1
        self.minbound = 0

    def run(self, centers, widths):
        performance = np.zeros(self.iterations)
        kvalues = np.zeros(self.iterations)

        self._initialise_population(centers, widths)
        self.max_cost, self.min_cost, self.avg_cost, self.max_idx, self.min_idx = self._evaluate_population()
        self.best_cost = self.current_population_fitness[self.min_idx]

        for i in range(self.iterations):
            print("Iteration: {}".format(i + 1))
            for j in range(self.population_size):
                self.new_population[j] = self._generate_trail_vector(j)
            for j in range(self.population_size):
                self.new_population_fitness[j], self.new_population_sizes[j] = self._evaluate_candidate(
                    self.new_population[j])
                if self.new_population_fitness[j] <= self.current_population_fitness[j]:
                    self.current_population_fitness[j] = self.new_population_fitness[j]
                    self.current_population_sizes[j] = self.new_population_sizes[j]
                    self.current_population[j] = copy.deepcopy(self.new_population[j])
            self.max_cost, self.min_cost, self.avg_cost, self.max_idx, self.min_idx = self._evaluate_population()
            if i == 0 or self.current_population_fitness[self.min_idx] < self.best_cost:
                self.best_cost = self.current_population_fitness[self.min_idx]

            performance[i] = self.best_cost
            kvalues[i] = self.current_population_sizes[self.min_idx]

        return performance, kvalues

    def _initialise_population(self, centers, widths):
        self.current_population = np.zeros((self.population_size, self.vsize))
        for i in range(len(centers)):
            for j in range(self.dimensionality):
                self.current_population[0][i * self.dimensionality + j] = centers[i][j]

            self.current_population[0][self.kappa * self.dimensionality + i] = widths[i]

        if len(centers) < self.kappa:
            for i in range(self.dimensionality):
                self.current_population[0][
                    len(centers) * self.dimensionality + i] = self.minbound + np.random.rand() * (
                        self.maxbound -
                        self.minbound)

            self.current_population[0][-1] = self.minbound + np.random.rand() * (
                    self.maxbound - self.minbound)

        self.new_population = np.zeros((self.population_size, self.vsize))
        self.current_population_fitness = np.zeros(self.population_size)
        self.new_population_fitness = np.zeros(self.population_size)
        self.current_population_sizes = np.zeros(self.population_size)
        self.new_population_sizes = np.zeros(self.population_size)

        for i in range(self.population_size):
            if i != 0:
                for j in range(self.vsize):
                    self.current_population[i][j] = self.minbound + np.random.rand() * (
                            self.maxbound - self.minbound)
            self.current_population_fitness[i], self.current_population_sizes[i] = self._evaluate_candidate(
                self.current_population[i])

    def _evaluate_candidate(self, candidate):
        centers = self._activate_centers(candidate)
        widths = self._obtain_widths(candidate)
        size = len(widths)
        rbfn = RBFN(centers, widths)
        sse = rbfn.fit(self.X_train, self.y_train)
        return sse / len(self.X_train) / self.no_classes, size

    def _activate_centers(self, candidate):
        centers = np.zeros((self.kappa, self.dimensionality))
        for i in range(self.kappa):
            for j in range(self.dimensionality):
                centers[i][j] = candidate[i * self.dimensionality + j]

        return centers

    def _obtain_widths(self, candidate):
        widths = np.zeros(self.kappa)
        for i in range(self.kappa):
            widths[i] = candidate[self.kappa * self.dimensionality + i]

        return widths

    def _evaluate_population(self):
        max_cost = np.max(self.current_population_fitness)
        min_cost = np.min(self.current_population_fitness)
        max_idx = np.argmax(self.current_population_fitness)
        min_idx = np.argmin(self.current_population_fitness)
        avg_cost = np.average(self.current_population_fitness)

        return max_cost, min_cost, avg_cost, max_idx, min_idx

    def _generate_trail_vector(self, idx):
        new_candidate = copy.deepcopy(self.current_population[idx])
        r0 = np.random.randint(low=0, high=self.population_size)
        while r0 == idx:
            r0 = np.random.randint(low=0, high=self.population_size)

        r1 = np.random.randint(low=0, high=self.population_size)
        while r1 == idx or r1 == r0:
            r1 = np.random.randint(low=0, high=self.population_size)

        r2 = np.random.randint(low=0, high=self.population_size)
        while r2 == idx or r2 == r1 or r2 == r0:
            r2 = np.random.randint(low=0, high=self.population_size)

        randi = np.random.randint(low=0, high=self.vsize)

        for i in range(self.vsize):
            if np.random.rand() <= self.cr or i == randi:
                if self.strategy == "BEST":
                    diff = self.current_population[self.min_idx][i] + self.f * (
                            self.current_population[r1][i] - self.current_population[r2][i])
                else:
                    diff = self.current_population[r0][i] + self.f * (
                            self.current_population[r1][i] - self.current_population[r2][i])

                if diff < self.minbound:
                    diff = self.current_population[r0][i] + np.random.rand() * (
                            self.minbound - self.current_population[r0][i])
                if diff > self.maxbound:
                    diff = self.current_population[r0][i] + np.random.rand() * (
                            self.maxbound - self.current_population[r0][i])

                new_candidate[i] = diff
            else:
                new_candidate[i] = self.current_population[idx][i]

        return new_candidate

    def get_centers(self):
        return self._activate_centers(self.current_population[self.min_idx])

    def get_widths(self):
        return self._obtain_widths(self.current_population[self.min_idx])


class DEFixed:

    def __init__(self, cr, f, population_size, iterations, k, X_train, y_train, strategy="RAND"):
        self.alg_name = "DE"
        self.cr = cr
        self.f = f
        self.strategy = strategy
        self.population_size = population_size
        self.iterations = iterations
        self.dimensionality = len(X_train[0])
        self.no_classes = len(set(y_train))
        self.X_train = X_train
        self.y_train = y_train
        self.kappa = k
        self.vsize = self.kappa * (self.dimensionality + 1)
        self.maxbound = 1
        self.minbound = 0

    def run(self):
        performance = np.zeros(self.iterations)
        kvalues = np.zeros(self.iterations)

        self._initialise_population()
        self.max_cost, self.min_cost, self.avg_cost, self.max_idx, self.min_idx = self._evaluate_population()
        self.best_cost = self.current_population_fitness[self.min_idx]

        for i in range(self.iterations):
            print("Iteration: {}".format(i + 1))
            for j in range(self.population_size):
                self.new_population[j] = self._generate_trail_vector(j)
            for j in range(self.population_size):
                self.new_population_fitness[j], self.new_population_sizes[j] = self._evaluate_candidate(
                    self.new_population[j])
                if self.new_population_fitness[j] <= self.current_population_fitness[j]:
                    self.current_population_fitness[j] = self.new_population_fitness[j]
                    self.current_population_sizes[j] = self.new_population_sizes[j]
                    self.current_population[j] = copy.deepcopy(self.new_population[j])
            self.max_cost, self.min_cost, self.avg_cost, self.max_idx, self.min_idx = self._evaluate_population()
            if i == 0 or self.current_population_fitness[self.min_idx] < self.best_cost:
                self.best_cost = self.current_population_fitness[self.min_idx]

            performance[i] = self.best_cost
            kvalues[i] = self.current_population_sizes[self.min_idx]

        return performance, kvalues

    def _initialise_population(self):
        self.current_population = np.zeros((self.population_size, self.vsize))
        self.new_population = np.zeros((self.population_size, self.vsize))

        self.current_population_fitness = np.zeros(self.population_size)
        self.new_population_fitness = np.zeros(self.population_size)

        self.current_population_sizes = np.zeros(self.population_size)
        self.new_population_sizes = np.zeros(self.population_size)

        for i in range(self.population_size):
            for j in range(self.vsize):
                self.current_population[i][j] = self.minbound + np.random.rand() * (
                        self.maxbound - self.minbound)
            self.current_population_fitness[i], self.current_population_sizes[i] = self._evaluate_candidate(
                self.current_population[i])
            self.new_population_sizes[i] = self.current_population_sizes[i]

    def _evaluate_candidate(self, candidate):
        centers = self._activate_centers(candidate)
        widths = self._obtain_widths(candidate)
        size = len(widths)
        rbfn = RBFN2(centers, widths)
        sse = rbfn.fit(self.X_train, self.y_train)
        return sse / len(self.X_train) / self.no_classes, size

    def _activate_centers(self, candidate):
        centers = np.zeros((self.kappa, self.dimensionality))
        for i in range(self.kappa):
            for j in range(self.dimensionality):
                centers[i][j] = candidate[i * self.dimensionality + j]

        return centers

    def _obtain_widths(self, candidate):
        widths = np.zeros(self.kappa)
        for i in range(self.kappa):
            widths[i] = candidate[self.kappa * self.dimensionality + i]

        return widths

    def _evaluate_population(self):
        max_cost = np.max(self.current_population_fitness)
        min_cost = np.min(self.current_population_fitness)
        max_idx = np.argmax(self.current_population_fitness)
        min_idx = np.argmin(self.current_population_fitness)
        avg_cost = np.average(self.current_population_fitness)

        return max_cost, min_cost, avg_cost, max_idx, min_idx

    def _generate_trail_vector(self, idx):
        new_candidate = copy.deepcopy(self.current_population[idx])
        r0 = np.random.randint(low=0, high=self.population_size)
        while r0 == idx:
            r0 = np.random.randint(low=0, high=self.population_size)

        r1 = np.random.randint(low=0, high=self.population_size)
        while r1 == idx or r1 == r0:
            r1 = np.random.randint(low=0, high=self.population_size)

        r2 = np.random.randint(low=0, high=self.population_size)
        while r2 == idx or r2 == r1 or r2 == r0:
            r2 = np.random.randint(low=0, high=self.population_size)

        randi = np.random.randint(low=0, high=self.vsize)

        for i in range(self.vsize):
            if np.random.rand() <= self.cr or i == randi:
                if self.strategy == "BEST":
                    diff = self.current_population[self.min_idx][i] + self.f * (
                            self.current_population[r1][i] - self.current_population[r2][i])
                else:
                    diff = self.current_population[r0][i] + self.f * (
                            self.current_population[r1][i] - self.current_population[r2][i])

                if diff < self.minbound:
                    diff = self.current_population[r0][i] + np.random.rand() * (
                            self.minbound - self.current_population[r0][i])
                if diff > self.maxbound:
                    diff = self.current_population[r0][i] + np.random.rand() * (
                            self.maxbound - self.current_population[r0][i])

                new_candidate[i] = diff
            else:
                new_candidate[i] = self.current_population[idx][i]

        return new_candidate

    def get_centers(self):
        return self._activate_centers(self.current_population[self.min_idx])

    def get_widths(self):
        return self._obtain_widths(self.current_population[self.min_idx])


class PSOFixed:
    def __init__(self, X_train, y_train, k, population_size=30, w=0.724, c1=1.468, c2=1.468, iterations=10000):
        self.alg_name = "PSOFixed"
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.population_size = population_size
        self.iterations = iterations
        self.dimensionality = len(X_train[0])
        self.no_classes = len(set(y_train))
        self.size = len(X_train)
        self.X_train = X_train
        self.y_train = y_train
        self.kappa = k
        self.vsize = self.kappa * (self.dimensionality + 1)
        self.maxbound = 1
        self.minbound = 0

    def run(self):
        performance = np.zeros(self.iterations)
        self.positions = np.zeros((self.population_size, self.vsize))
        self.velocities = np.zeros((self.population_size, self.vsize))
        self.pbests = np.zeros((self.population_size, self.vsize))

        self.current_fitness = np.zeros(self.population_size)
        self.pbest_fitness = np.zeros(self.population_size)

        self.init_positions()
        self.calculate_swarm_stats()

        for i in range(self.iterations):
            print("Iteration: " + str(i))
            for j in range(self.population_size):
                self.update_vel_positions(j)

            for j in range(self.population_size):
                self.current_fitness[j] = self.calculate_solution_cost(self.positions[j])
                if self.current_fitness[j] < self.pbest_fitness[j]:
                    self.pbest_fitness[j] = self.current_fitness[j]
                    self.pbests[j] = copy.deepcopy(self.positions[j])

            self.calculate_swarm_stats()

            performance[i] = self.min_cost

        return performance

    def init_positions(self):
        self.velocities = np.zeros((self.population_size, self.vsize))
        for i in range(self.population_size):
            for j in range(self.kappa):
                self.positions[i][self.kappa * self.dimensionality + j] = 1 * np.random.rand() * 1
            for j in range(self.kappa):
                r = np.random.randint(0, self.size)
                for k in range(self.dimensionality):
                    self.positions[i][j * self.dimensionality + k] = self.X_train[r][k]

            cj = np.zeros(self.dimensionality)
            ck = np.zeros(self.dimensionality)
            for j in range(self.kappa):
                dt = 0
                for z in range(self.dimensionality):
                    cj[z] = self.positions[i][j * self.dimensionality + z]
                for k in range(self.kappa):
                    if j != k:
                        for z in range(self.dimensionality):
                            ck[z] = self.positions[i][k * self.dimensionality + z]
                        dt += dst.euclidean(cj, ck)

                self.positions[i][self.kappa * self.dimensionality + j] = dt / (self.kappa - 1)

            self.current_fitness[i] = self.pbest_fitness[i] = self.calculate_solution_cost(self.positions[i])

        for i in range(self.population_size):
            self.pbests[i] = copy.deepcopy(self.positions[i])

    def calculate_solution_cost(self, candidate):
        centers = self._activate_centers(candidate)
        widths = self._obtain_widths(candidate)
        rbfn = RBFN(centers, widths)
        sse = rbfn.fit(self.X_train, self.y_train)
        return sse / len(self.X_train) / self.no_classes

    def calculate_swarm_stats(self):
        self.max_cost = np.max(self.pbest_fitness)
        self.max_idx = np.argmax(self.pbest_fitness)
        self.min_cost = np.min(self.pbest_fitness)
        self.min_idx = np.argmin(self.pbest_fitness)
        self.avg_cost = np.average(self.pbest_fitness)

    def _activate_centers(self, candidate):
        centers = np.zeros((self.kappa, self.dimensionality))
        for i in range(self.kappa):
            for j in range(self.dimensionality):
                centers[i][j] = candidate[i * self.dimensionality + j]

        return centers

    def _obtain_widths(self, candidate):
        widths = np.zeros(self.kappa)
        for i in range(self.kappa):
            widths[i] = candidate[self.kappa * self.dimensionality + i]

        return widths

    def update_vel_positions(self, idx):
        for i in range(self.vsize):
            v = self.w * self.velocities[idx][i] + self.c1 * np.random.rand() * (
                        self.pbests[idx][i] - self.positions[idx][i]) + self.c2 * np.random.rand() * (
                            self.pbests[self.min_idx][i] - self.positions[idx][i])
            if v > 1:
                v = 1
            if v < -1:
                v = -1
            self.velocities[idx][i] = v
            self.positions[idx][i] += v

            if self.positions[idx][i] > self.maxbound:
                self.positions[idx][i] -= 2 * (self.positions[idx][i] - self.maxbound)
                self.velocities[idx][i] *= -1

            if self.positions[idx][i] < self.minbound:
                self.positions[idx][i] += 2 * (self.minbound - self.positions[idx][i])
                self.velocities[idx][i] *= -1

    def get_centers(self):
        return self._activate_centers(self.pbests[self.min_idx])

    def get_widths(self):
        return self._obtain_widths(self.pbests[self.min_idx])
