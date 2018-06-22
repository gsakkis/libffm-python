# coding: utf-8

import os
import ctypes
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


def wrap_tuples(row):
    size = len(row)
    nodes_array = (FFM_Node * size)()

    for i, (f, j, v) in enumerate(row):
        node = nodes_array[i]
        node.f = f
        node.j = j
        node.v = v

    return nodes_array


def wrap_dataset_init(X, target):
    l = len(target)
    data = (FFM_Line * l)()

    for i, (x, y) in enumerate(zip(X, target)):
        d = data[i]
        nodes = wrap_tuples(x)
        d.data = nodes
        d.label = y
        d.size = nodes._length_

    return data


def wrap_dataset(X, y):
    line_array = wrap_dataset_init(X, y)
    return _lib.ffm_convert_data(line_array, line_array._length_)


class FFMData:

    def __init__(self, X, y):
        self.labels = y
        self._data = wrap_dataset(X, y)

    def __del__(self):
        _lib.ffm_cleanup_problem(self._data)


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
        if not isinstance(X, FFMData):
            ffm_data = FFMData(X, np.zeros(len(X)))
        else:
            ffm_data = X
        pred_ptr = _lib.ffm_predict_batch(ffm_data._data, self._model)
        try:
            pred_ptr_address = ctypes.addressof(pred_ptr.contents)
            array_cast = (ctypes.c_float * ffm_data._data.size).from_address(pred_ptr_address)
            return np.ctypeslib.as_array(array_cast).copy()
        finally:
            _lib.ffm_cleanup_prediction(pred_ptr)

    def fit(self, X, y, val_X_y=None):
        """
        :param X: feature data or FFMData format
        :param y: target
        :param val_X_y: (X, y) data for validation
        """
        params = self.get_params()
        ffm_params = FFM_Parameter(eta=params['eta'], lam=params['lam'], k=params['k'],
                                   normalization=params['normalization'])

        train_data = FFMData(X, y)
        self._model = _lib.ffm_init_model(train_data._data, ffm_params)
        scorer = params['scorer']
        if isinstance(scorer, str):
            try:
                scorer = _scorers[scorer]
            except KeyError:
                raise ValueError('Unknown scorer: {}'.format(scorer))

        # Translate Validation Data
        if val_X_y:
            val_data = FFMData(*val_X_y)
            print('%-8s%-16s%-16s%-16s%-8s' % ('Iter', 'Train_Loss', 'Train_Score', 'Val_Score', 'Best_Iter'))
        else:
            val_data = None
            print('%-8s%-16s%-16s%-8s' % ('Iter', 'Train_Loss', 'Train_Score', 'Best_Iter'))

        # Score Recorder: > or <
        best_model = None
        score_index = -1
        if scorer.maximum:
            cmp = op.gt
            score = -np.inf
        else:
            cmp = op.lt
            score = np.inf

        # Training Process
        log_loss = _scorers['log_loss']
        early_stopping = params['early_stopping']
        for i in range(params['num_iter']):
            _lib.ffm_train_iteration(train_data._data, self._model, ffm_params)
            train_loss = self._score(train_data, log_loss)
            train_score = self._score(train_data, scorer)
            if val_data:
                val_score = self._score(val_data, scorer)
            else:
                val_score = train_score

            if cmp(val_score, score):
                score = val_score
                score_index = i
                best_model = self._model

            if val_data:
                print('%-8d%-16.4f%-16.4f%-16.4f%-8d' % (i, train_loss, train_score, val_score, score_index))
            else:
                print('%-8d%-16.4f%-16.4f%-8d' % (i, train_loss, train_score, score_index))

            if (i - score_index) >= early_stopping:
                print('Early stopping at %d rounds' % i)
                break

            self._model = best_model
        return self

    def _score(self, ffm_data, scorer):
        predict = self.predict_proba if scorer.probabilities else self.predict
        return scorer.metric(ffm_data.labels, predict(ffm_data))
