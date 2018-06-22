# coding: utf-8

import os
import ctypes
import itertools as it
import operator as op
from collections import namedtuple

import numpy as np
from sklearn import metrics
from sklearn.base import BaseEstimator, ClassifierMixin


Float_ptr = ctypes.POINTER(ctypes.c_float)


class FFM_Parameter(ctypes.Structure):
    _fields_ = [
        ('eta', ctypes.c_float),
        ('lam', ctypes.c_float),
        ('nr_iters', ctypes.c_int),
        ('k', ctypes.c_int),
        ('normalization', ctypes.c_bool),
        ('auto_stop', ctypes.c_bool),
    ]


class FFM_Model(ctypes.Structure):
    _fields_ = [
        ('n', ctypes.c_int),
        ('m', ctypes.c_int),
        ('k', ctypes.c_int),
        ('W', Float_ptr),
        ('normalization', ctypes.c_bool)
    ]
FFM_Model_ptr = ctypes.POINTER(FFM_Model)


class FFM_Node(ctypes.Structure):
    _fields_ = [
        ('f', ctypes.c_int),
        ('j', ctypes.c_int),
        ('v', ctypes.c_float),
    ]
FFM_Node_ptr = ctypes.POINTER(FFM_Node)


class FFM_Line(ctypes.Structure):
    _fields_ = [
        ('data', ctypes.POINTER(FFM_Node)),
        ('label', ctypes.c_float),
        ('size', ctypes.c_int),
    ]
FFM_Line_ptr = ctypes.POINTER(FFM_Line)


class FFM_Problem(ctypes.Structure):
    _fields_ = [
        ('size', ctypes.c_int),
        ('num_nodes', ctypes.c_long),
        ('data', ctypes.POINTER(FFM_Node)),
        ('pos', ctypes.POINTER(ctypes.c_long)),
        ('labels', Float_ptr),
        ('scales', Float_ptr),
        ('n', ctypes.c_int),
        ('m', ctypes.c_int),
    ]
FFM_Problem_ptr = ctypes.POINTER(FFM_Problem)


path = os.path.dirname(os.path.abspath(__file__))
lib_path = path + '/' + next(i for i in os.listdir(path) if i.endswith('.so'))
_lib = ctypes.cdll.LoadLibrary(lib_path)

_lib.ffm_convert_data.restype = FFM_Problem
_lib.ffm_convert_data.argtypes = [FFM_Line_ptr, ctypes.c_int]

_lib.ffm_init_model.restype = FFM_Model
_lib.ffm_init_model.argtypes = [FFM_Problem_ptr, FFM_Parameter]

_lib.ffm_train_iteration.restype = ctypes.c_float
_lib.ffm_train_iteration.argtypes = [FFM_Problem_ptr, FFM_Model_ptr, FFM_Parameter]

_lib.ffm_predict_array.restype = ctypes.c_float
_lib.ffm_predict_array.argtypes = [FFM_Node_ptr, ctypes.c_int, FFM_Model_ptr]

_lib.ffm_predict_batch.restype = Float_ptr
_lib.ffm_predict_batch.argtypes = [FFM_Problem_ptr, FFM_Model_ptr]

_lib.ffm_load_model_c_string.restype = FFM_Model
_lib.ffm_load_model_c_string.argtypes = [ctypes.c_char_p]

_lib.ffm_save_model_c_string.argtypes = [FFM_Model_ptr, ctypes.c_char_p]

_lib.ffm_cleanup_problem.argtypes = [FFM_Problem_ptr]

_lib.ffm_cleanup_prediction.argtypes = [Float_ptr]


def to_ffm_problem(X, y=it.repeat(0)):
    lines = (FFM_Line * len(X))()
    for line, row, label in zip(lines, X, y):
        line.label = label
        line.size = len(row)
        line.data = (FFM_Node * line.size)()
        for node, (f, j, v) in zip(line.data, row):
            node.f = f
            node.j = j
            node.v = v
    return _lib.ffm_convert_data(lines, len(lines))


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
        self._model = _lib.ffm_load_model_c_string(path_char)
        return self

    def save_model(self, path):
        if self._model is None:
            raise ValueError('Model has not been trained')
        path_char = ctypes.c_char_p(path.encode())
        _lib.ffm_save_model_c_string(self._model, path_char)

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(np.uint8)

    def predict_proba(self, X):
        problem = to_ffm_problem(X) if not isinstance(X, FFM_Problem) else X
        pred_ptr = _lib.ffm_predict_batch(problem, self._model)
        try:
            pred_ptr_address = ctypes.addressof(pred_ptr.contents)
            array_cast = (ctypes.c_float * problem.size).from_address(pred_ptr_address)
            return np.ctypeslib.as_array(array_cast).copy()
        finally:
            _lib.ffm_cleanup_prediction(pred_ptr)
            if problem is not X:
                _lib.ffm_cleanup_problem(problem)

    def fit(self, X, y, val_X_y=None):
        """
        :param X: feature data
        :param y: target
        :param val_X_y: (X, y) data for validation
        """
        scorer = self.scorer
        if isinstance(scorer, str):
            try:
                scorer = _scorers[scorer]
            except KeyError:
                raise ValueError('Unknown scorer: {}'.format(scorer))

        val_problem = None
        problem = to_ffm_problem(X, y)
        try:
            if val_X_y:
                val_problem = to_ffm_problem(val_X_y[0])
                val_X_y = (val_problem, val_X_y[1])
            self._fit(problem, y, val_X_y, scorer)
        finally:
            _lib.ffm_cleanup_problem(problem)
            if val_problem is not None:
                _lib.ffm_cleanup_problem(val_problem)

        return self

    def _fit(self, problem, y, val_X_y, scorer):
        ffm_params = FFM_Parameter(eta=self.eta, lam=self.lam, k=self.k,
                                   normalization=self.normalization)
        self._model = _lib.ffm_init_model(problem, ffm_params)

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
            print('%-8s%-16s%-16s%-16s%-8s' % ('Iter', 'Train_Loss', 'Train_Score', 'Val_Score', 'Best_Iter'))
        else:
            print('%-8s%-16s%-16s%-8s' % ('Iter', 'Train_Loss', 'Train_Score', 'Best_Iter'))

        log_loss = _scorers['log_loss']
        early_stopping = self.early_stopping
        for i in range(self.num_iter):
            _lib.ffm_train_iteration(problem, self._model, ffm_params)
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
                print('%-8d%-16.4f%-16.4f%-16.4f%-8d' % (i, train_loss, train_score, val_score, score_index))
            else:
                print('%-8d%-16.4f%-16.4f%-8d' % (i, train_loss, train_score, score_index))

            if (i - score_index) >= early_stopping:
                print('Early stopping at %d rounds' % i)
                break

            self._model = best_model

    def _score(self, X, y, scorer):
        predict = self.predict_proba if scorer.probabilities else self.predict
        return scorer.metric(y, predict(X))
