__all__ = ['FFM', 'read_libffm']

import logging

import numpy as np
from sklearn import metrics
from sklearn.base import BaseEstimator

from ._wrapper import srand, FFM_Model, FFM_Parameters, FFM_Problem


logger = logging.getLogger('ffm')


class FFM(BaseEstimator):

    def __init__(self, eta=0.2, lam=0.00002, k=4, normalization=True, nr_iters=10, auto_stop=5,
                 score='neg_log_loss', nr_threads=1, randomization=True):
        """
        :param eta: learning rate
        :param lam: regularization parameter
        :param k: number of latent factors
        :param normalization: enable/disable instance-wise normalization
        :param nr_iters: number of iterations
        :param auto_stop: early stopping rounds
        :param score: an sklearn.metrics Scorer, or one of string keys in sklearn.metrics.SCORERS
        """
        self.set_params(eta=eta, lam=lam, k=k, normalization=normalization, nr_iters=nr_iters,
                        auto_stop=auto_stop, score=score, nr_threads=nr_threads,
                        randomization=randomization)
        self._model = None

    def read_model(self, path):
        self._model = FFM_Model.from_file(path)
        return self

    def save_model(self, path):
        if self._model is None:
            raise ValueError('Model has not been trained')
        self._model.to_file(path)

    def predict_proba(self, X):
        problem = FFM_Problem(X) if not isinstance(X, FFM_Problem) else X
        return self._model.predict_batch(problem)

    decision_function = predict_proba

    def fit_from_file(self, training_path, validation_path=None):
        self._model = FFM_Model.train(training_path, validation_path, self._params, self.nr_threads)

    def fit(self, X, y, val_X_y=None):
        """
        :param X: training data as a sequence of (field, feature, value) triples
        :param y: target as a len(X) sequence of values
        :param val_X_y: (X, y) data for validation
        """
        problem = FFM_Problem(X, y)
        ffm_params = self._params
        if self.randomization:
            srand(1)
        self._model = FFM_Model(problem, ffm_params)
        if val_X_y:
            val_X_y = (FFM_Problem(val_X_y[0]), val_X_y[1])
            logger.info('%-8s%-16s%-16s', 'iter', 'tr_logloss', 'va_score')
        else:
            logger.info('%-8s%-16s', 'iter', 'tr_logloss')

        best_model = FFM_Model(problem, ffm_params)
        best_va_score = -np.inf
        best_iter = 0
        auto_stop = self.auto_stop
        for i in range(1, self.nr_iters + 1):
            tr_logloss = self._model.train_iteration(problem, ffm_params, self.nr_threads)
            if not val_X_y:
                logger.info('%-8d%-16.5f', i, tr_logloss)
            else:
                va_score = self.scorer(self, *val_X_y)
                logger.info('%-8d%-16.5f%-16.5f', i, tr_logloss, abs(va_score))
                if auto_stop:
                    if va_score <= best_va_score:
                        best_model.copy_to(self._model)
                        if i - best_iter >= auto_stop:
                            logger.info('Auto-stop. Use model at %dth iteration', best_iter)
                            break
                    else:
                        self._model.copy_to(best_model)
                        best_va_score = va_score
                        best_iter = i
        return self

    @property
    def scorer(self):
        score = self.score
        if isinstance(score, str):
            try:
                return metrics.SCORERS[score]
            except KeyError:
                raise ValueError('Unknown score: {}'.format(score))
        return score

    @property
    def _params(self):
        return FFM_Parameters(eta=self.eta, lam=self.lam, nr_iters=self.nr_iters, k=self.k,
                              normalization=self.normalization, randomization=self.randomization,
                              auto_stop=bool(self.auto_stop))


def read_libffm(path):
    X, y = [], []
    with open(path, 'rt') as f:
        for line in f:
            items = line.strip().split(' ')
            y.append(int(items.pop(0)))
            row = []
            for item in items:
                field, feature, value = item.split(':')
                row.append((int(field), int(feature), float(value)))
            X.append(row)
    return X, y
