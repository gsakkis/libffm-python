__all__ = ['FFM', 'read_libffm']

import ctypes
import logging

import numpy as np
from sklearn import metrics
from sklearn.base import BaseEstimator, ClassifierMixin

from ._wrapper import lib, FFM_Problem, FFM_Parameter


logger = logging.getLogger('ffm')
srand = ctypes.CDLL('libc.so.6').srand


class FFM(BaseEstimator, ClassifierMixin):

    def __init__(self, eta=0.2, lam=0.00002, k=4, normalization=True, num_iter=10, early_stopping=5,
                 scorer='neg_log_loss', randomization=True):
        """
        :param eta: learning rate
        :param lam: regularization parameter
        :param k: number of latent factors
        :param normalization: enable/disable instance-wise normalization
        :param num_iter: number of iterations
        :param early_stopping: early stopping rounds
        :param scorer: an sklearn.metrics Scorer, or one of string keys in sklearn.metrics.SCORERS
        """
        self.set_params(eta=eta, lam=lam, k=k, normalization=normalization, num_iter=num_iter,
                        early_stopping=early_stopping, scorer=scorer, randomization=randomization)
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
        ffm_params = FFM_Parameter(eta=self.eta, lam=self.lam, k=self.k,
                                   normalization=self.normalization,
                                   randomization=self.randomization)
        if self.randomization:
            srand(1)
        self._model = lib.ffm_init_model(problem, ffm_params)
        if val_X_y:
            val_X_y = (FFM_Problem(val_X_y[0]), val_X_y[1])
            log_format = '%(i)-8d%(train_score)-16.4f%(score)-16.4f%(best_score_index)-8d'
            logger.info('%-8s%-16s%-16s%-16s%-8s',
                        'Iter', 'Train_Loss', 'Train_Score', 'Val_Score', 'Best_Iter')
        else:
            log_format = '%(i)-8d%(train_score)-16.4f%(best_score_index)-8d'
            logger.info('%-8s%-16s%-16s%-8s', 'Iter', 'Train_Loss', 'Train_Score', 'Best_Iter')

        best_model = lib.ffm_init_model(problem, ffm_params)
        best_score = -np.inf
        best_score_index = -1
        early_stopping = self.early_stopping
        for i in range(self.num_iter):
            lib.ffm_train_iteration(problem, self._model, ffm_params)
            train_score = scorer(self, problem, y)
            score = scorer(self, *val_X_y) if val_X_y else train_score
            if best_score < score:
                best_score = score
                best_score_index = i
                lib.ffm_copy_model(self._model, best_model)
            else:
                lib.ffm_copy_model(best_model, self._model)

            train_score *= scorer._sign
            score *= scorer._sign
            logger.info(log_format, locals())
            if (i - best_score_index) >= early_stopping:
                logger.info('Early stopping at %d rounds', i)
                break

        return self


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
