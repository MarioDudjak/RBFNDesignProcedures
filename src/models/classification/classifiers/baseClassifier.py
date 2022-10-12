class BaseClassifier:

    def __init__(self, alg, alg_name):
        self.alg = alg
        self.name = alg_name

    def fit(self, X_train, y_train):
        return self.alg.fit()

    def predict(self, X_test):
        return self.alg.predict()

    def get_params(self):
        return self.get_params()

    def predict_proba(self, X_test):
        return self.alg.predict_proba()
