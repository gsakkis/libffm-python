
__all__ = ['FFMEstimator', 'read_libffm']

import logging

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import get_scorer

from ._wrapper import srand, FFM_Model, FFM_Parameters, FFM_Problem


logger = logging.getLogger('ffm')


class FFMEstimator(BaseEstimator):

    def __init__(self, eta=0.2, lam=0.00002, k=4, normalization=True, nr_iters=10, auto_stop=5,
                 scorer='neg_log_loss', nr_threads=1, randomization=True):
        """
        :param eta: learning rate
        :param lam: regularization parameter
        :param k: number of latent factors
        :param normalization: enable/disable instance-wise normalization
        :param nr_iters: number of iterations
        :param auto_stop: early stopping rounds
        :param scorer: an sklearn.metrics Scorer, or one of string keys in sklearn.metrics.SCORERS
        """
        self.set_params(eta=eta, lam=lam, k=k, normalization=normalization, nr_iters=nr_iters,
                        auto_stop=auto_stop, scorer=scorer, nr_threads=nr_threads,
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
        problem = self._get_ffm_problem(X) if not isinstance(X, FFM_Problem) else X
        pred_true = self._model.predict_batch(problem)
        return np.c_[1 - pred_true, pred_true]

    decision_function = predict_proba

    def fit_from_file(self, training_path, validation_path=None):
        self._model = FFM_Model.train(training_path, validation_path,
                                      self._ffm_params, self.nr_threads)

    def fit(self, X, y, val_X_y=None):
        """
        :param X: training data as a sequence of (field, feature, value) triples
        :param y: target as a len(X) sequence of values
        :param val_X_y: (X, y) data for validation
        """
        if self.randomization:
            srand(1)

        if val_X_y is not None:
            val_X_y = (self._get_ffm_problem(val_X_y[0]), val_X_y[1])
            logger.info('%-8s%-16s%-16s', 'iter', 'tr_logloss', 'va_score')
        else:
            logger.info('%-8s%-16s', 'iter', 'tr_logloss')

        best_iter = 0
        best_va_score = -np.inf
        problem = self._get_ffm_problem(X, y)
        ffm_params = self._ffm_params
        best_model = FFM_Model(problem, ffm_params)
        self._model = current_model = FFM_Model(problem, ffm_params)
        for i in range(1, self.nr_iters + 1):
            tr_logloss = current_model.train_iteration(problem, ffm_params, self.nr_threads)
            if val_X_y is None:
                logger.info('%-8d%-16.5f', i, tr_logloss)
            else:
                va_score = self.score(*val_X_y)
                logger.info('%-8d%-16.5f%-16.5f', i, tr_logloss, abs(va_score))
                if self.auto_stop:
                    if va_score <= best_va_score:
                        best_model.copy_to(current_model)
                        if i - best_iter >= self.auto_stop:
                            logger.info('Auto-stop. Use model at %dth iteration', best_iter)
                            break
                    else:
                        current_model.copy_to(best_model)
                        best_va_score = va_score
                        best_iter = i
        return self

    def score(self, X, y):
        return get_scorer(self.scorer)(self, X, y)

    @property
    def _ffm_params(self):
        return FFM_Parameters(eta=self.eta, lam=self.lam, nr_iters=self.nr_iters, k=self.k,
                              normalization=self.normalization, randomization=self.randomization,
                              auto_stop=bool(self.auto_stop))

    @classmethod
    def _get_ffm_problem(cls, X, y=None):
        return FFM_Problem(X, y)


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
