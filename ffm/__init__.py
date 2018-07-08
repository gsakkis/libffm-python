__all__ = ['FFM', 'read_libffm']

import ctypes
import logging

import numpy as np
from sklearn import metrics
from sklearn.base import BaseEstimator, ClassifierMixin

from ._wrapper import lib, FFM_Problem, FFM_Parameters


logger = logging.getLogger('ffm')
srand = ctypes.CDLL('libc.so.6').srand


class FFM(BaseEstimator, ClassifierMixin):

    def __init__(self, eta=0.2, lam=0.00002, k=4, normalization=True, nr_iters=10, early_stopping=5,
                 scorer='neg_log_loss', num_threads=1, randomization=True):
        """
        :param eta: learning rate
        :param lam: regularization parameter
        :param k: number of latent factors
        :param normalization: enable/disable instance-wise normalization
        :param nr_iters: number of iterations
        :param early_stopping: early stopping rounds
        :param scorer: an sklearn.metrics Scorer, or one of string keys in sklearn.metrics.SCORERS
        """
        self.set_params(eta=eta, lam=lam, k=k, normalization=normalization, nr_iters=nr_iters,
                        early_stopping=early_stopping, scorer=scorer, num_threads=num_threads,
                        randomization=randomization)
        self._model = None

    def read_model(self, path):
        path_char = ctypes.c_char_p(path.encode())
        self._model = lib.ffm_load_model_c_string(path_char)
        return self

    def save_model(self, path):
        if self._model is None:
            raise ValueError('Model has not been trained')
        path_char = ctypes.c_char_p(path.encode())
        lib.ffm_save_model_c_string(self._model, path_char)

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(np.uint8)

    def predict_proba(self, X):
        problem = FFM_Problem(X) if not isinstance(X, FFM_Problem) else X
        pred_ptr = lib.ffm_predict_batch(problem, self._model)
        try:
            pred_ptr_address = ctypes.addressof(pred_ptr.contents)
            array_cast = (ctypes.c_float * problem.size).from_address(pred_ptr_address)
            return np.ctypeslib.as_array(array_cast).copy()
        finally:
            lib.ffm_cleanup_prediction(pred_ptr)

    decision_function = predict_proba

    def fit(self, X, y, val_X_y=None):
        """
        :param X: training data as a sequence of (field, feature, value) triples
        :param y: target as a len(X) sequence of values
        :param val_X_y: (X, y) data for validation
        """
        scorer = self.scorer
        if isinstance(scorer, str):
            try:
                scorer = metrics.SCORERS[scorer]
            except KeyError:
                raise ValueError('Unknown scorer: {}'.format(scorer))

        problem = FFM_Problem(X, y)
        ffm_params = self._params
        if self.randomization:
            srand(1)
        self._model = lib.ffm_init_model(problem, ffm_params)
        if val_X_y:
            val_X_y = (FFM_Problem(val_X_y[0]), val_X_y[1])
            logger.info('%-8s%-16s%-16s', 'iter', 'tr_logloss', 'va_score')
        else:
            logger.info('%-8s%-16s', 'iter', 'tr_logloss')

        best_model = lib.ffm_init_model(problem, ffm_params)
        best_va_score = -np.inf
        best_iter = 0
        early_stopping = self.early_stopping
        for i in range(1, self.nr_iters + 1):
            tr_logloss = lib.ffm_train_iteration(problem, self._model, ffm_params, self.num_threads)
            if not val_X_y:
                logger.info('%-8d%-16.5f', i, tr_logloss)
            else:
                va_score = scorer(self, *val_X_y)
                logger.info('%-8d%-16.5f%-16.5f', i, tr_logloss, abs(va_score))
                if early_stopping:
                    if va_score <= best_va_score:
                        lib.ffm_copy_model(best_model, self._model)
                        if i - best_iter >= early_stopping:
                            logger.info('Auto-stop. Use model at %dth iteration', best_iter)
                            break
                    else:
                        lib.ffm_copy_model(self._model, best_model)
                        best_va_score = va_score
                        best_iter = i
        return self

    @property
    def _params(self):
        return FFM_Parameters(eta=self.eta, lam=self.lam, nr_iters=self.nr_iters, k=self.k,
                              normalization=self.normalization, randomization=self.randomization,
                              auto_stop=bool(self.early_stopping))


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
