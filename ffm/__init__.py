import ctypes
import logging
import operator as op
from collections import namedtuple

import numpy as np
from sklearn import metrics
from sklearn.base import BaseEstimator, ClassifierMixin

from ._wrapper import lib, FFM_Problem, FFM_Parameter


logger = logging.getLogger('ffm')
Scorer = namedtuple('Scorer', ['metric', 'maximum', 'probabilities'])
_scorers = {
    'log_loss': Scorer(metrics.log_loss, maximum=False, probabilities=True),
    'roc_auc': Scorer(metrics.roc_auc_score, maximum=True, probabilities=True),
    'f1': Scorer(metrics.f1_score, maximum=True, probabilities=False),
    'accuracy': Scorer(metrics.accuracy_score, maximum=True, probabilities=False),
}


class FFM(BaseEstimator, ClassifierMixin):

    def __init__(self, eta=0.2, lam=0.00002, k=4, normalization=True, num_iter=10, early_stopping=5,
                 scorer='log_loss'):
        """
        :param eta: learning rate
        :param lam: regularization parameter
        :param k: number of latent factors
        :param normalization: enable/disable instance-wise normalization
        :param num_iter: number of iterations
        :param early_stopping: early stopping rounds
        :param scorer: either a `Scorer` instance or one of the predefined scorers:
            'log_loss', 'roc_auc', 'f1', 'accuracy'
        """
        self.set_params(eta=eta, lam=lam, k=k, normalization=normalization, num_iter=num_iter,
                        early_stopping=early_stopping, scorer=scorer)
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

    def fit(self, X, y, val_X_y=None):
        """
        :param X: training data as a sequence of (field, feature, value) triples
        :param y: target as a len(X) sequence of values
        :param val_X_y: (X, y) data for validation
        """
        scorer = self.scorer
        if isinstance(scorer, str):
            try:
                scorer = _scorers[scorer]
            except KeyError:
                raise ValueError('Unknown scorer: {}'.format(scorer))

        problem = FFM_Problem(X, y)
        if val_X_y:
            val_X_y = (FFM_Problem(val_X_y[0]), val_X_y[1])

        ffm_params = FFM_Parameter(eta=self.eta, lam=self.lam, k=self.k,
                                   normalization=self.normalization)
        self._model = lib.ffm_init_model(problem, ffm_params)

        best_model = None
        score_index = -1
        if scorer.maximum:
            cmp = op.gt
            score = -np.inf
        else:
            cmp = op.lt
            score = np.inf

        # Training Process
        if val_X_y:
            logger.info('%-8s%-16s%-16s%-16s%-8s',
                        'Iter', 'Train_Loss', 'Train_Score', 'Val_Score', 'Best_Iter')
        else:
            logger.info('%-8s%-16s%-16s%-8s', 'Iter', 'Train_Loss', 'Train_Score', 'Best_Iter')

        log_loss = _scorers['log_loss']
        early_stopping = self.early_stopping
        for i in range(self.num_iter):
            lib.ffm_train_iteration(problem, self._model, ffm_params)
            train_loss = self._score(problem, y, log_loss)
            train_score = self._score(problem, y, scorer)
            if val_X_y:
                val_score = self._score(*val_X_y, scorer)
            else:
                val_score = train_score

            if cmp(val_score, score):
                score = val_score
                score_index = i
                best_model = self._model

            if val_X_y:
                logger.info('%-8d%-16.4f%-16.4f%-16.4f%-8d', i, train_loss, train_score, val_score, score_index)
            else:
                logger.info('%-8d%-16.4f%-16.4f%-8d', i, train_loss, train_score, score_index)

            if (i - score_index) >= early_stopping:
                logger.info('Early stopping at %d rounds', i)
                break

            self._model = best_model

        return self

    def _score(self, X, y, scorer):
        predict = self.predict_proba if scorer.probabilities else self.predict
        return scorer.metric(y, predict(X))
