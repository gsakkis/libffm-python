# coding: utf-8

import os
import ctypes
import itertools as it

import numpy as np


Float_ptr = ctypes.POINTER(ctypes.c_float)


def py_to_c_str(string):
    if isinstance(string, str):
        string = string.encode()
    return ctypes.c_char_p(string)


class Structure(ctypes.Structure):

    @classmethod
    def pointer(cls):
        return ctypes.POINTER(cls)


class FFM_Parameters(Structure):
    _fields_ = [
        ('eta', ctypes.c_float),
        ('lam', ctypes.c_float),
        ('nr_iters', ctypes.c_int),
        ('k', ctypes.c_int),
        ('normalization', ctypes.c_bool),
        ('randomization', ctypes.c_bool),
        ('auto_stop', ctypes.c_bool),
    ]


class FFM_Model(Structure):
    _fields_ = [
        ('n', ctypes.c_int),
        ('m', ctypes.c_int),
        ('k', ctypes.c_int),
        ('W', Float_ptr),
        ('normalization', ctypes.c_bool)
    ]

    def predict_batch(self, problem):
        pred_ptr = lib.ffm_predict_batch(problem, self)
        try:
            pred_ptr_address = ctypes.addressof(pred_ptr.contents)
            array_cast = (ctypes.c_float * problem.size).from_address(pred_ptr_address)
            return np.ctypeslib.as_array(array_cast).copy()
        finally:
            lib.ffm_cleanup_prediction(pred_ptr)

    @staticmethod
    def from_file(path):
        return lib.ffm_load_model_c_string(py_to_c_str(path))

    @staticmethod
    def train(params, training_path, validation_path=None, num_threads=1):
        return lib.ffm_train_model(
            py_to_c_str(training_path),
            py_to_c_str(training_path + '.bin'),
            py_to_c_str(validation_path) if validation_path else None,
            py_to_c_str(validation_path + '.bin') if validation_path else None,
            params,
            num_threads,
        )

    def to_file(self, path):
        lib.ffm_save_model_c_string(self, py_to_c_str(path))


class FFM_Node(Structure):
    _fields_ = [
        ('f', ctypes.c_int),
        ('j', ctypes.c_int),
        ('v', ctypes.c_float),
    ]


class FFM_Line(Structure):
    _fields_ = [
        ('data', FFM_Node.pointer()),
        ('label', ctypes.c_float),
        ('size', ctypes.c_int),
    ]


class FFM_Problem(Structure):
    _fields_ = [
        ('size', ctypes.c_int),
        ('num_nodes', ctypes.c_long),
        ('data', FFM_Node.pointer()),
        ('pos', ctypes.POINTER(ctypes.c_long)),
        ('labels', Float_ptr),
        ('scales', Float_ptr),
        ('n', ctypes.c_int),
        ('m', ctypes.c_int),
    ]

    def __init__(self, X, y=it.repeat(0)):
        lines = (FFM_Line * len(X))()
        for line, row, label in zip(lines, X, y):
            line.label = label
            line.size = len(row)
            line.data = (FFM_Node * line.size)()
            for node, (f, j, v) in zip(line.data, row):
                node.f = f
                node.j = j
                node.v = v
        lib.ffm_init_problem(self, lines, len(lines))

    def __del__(self):
        lib.ffm_cleanup_problem(self)


def setup_lib(lib_path=None):
    if lib_path is None:
        path = os.path.dirname(os.path.abspath(__file__))
        lib_path = path + '/' + next(i for i in os.listdir(path) if i.endswith('.so'))

    lib = ctypes.CDLL(lib_path)

    lib.ffm_init_problem.argtypes = [FFM_Problem.pointer(), FFM_Line.pointer(), ctypes.c_int]

    lib.ffm_init_model.restype = FFM_Model
    lib.ffm_init_model.argtypes = [FFM_Problem.pointer(), FFM_Parameters]

    lib.ffm_copy_model.argtypes = [FFM_Model.pointer(), FFM_Model.pointer()]

    lib.ffm_train_model.restype = FFM_Model
    lib.ffm_train_model.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                                    ctypes.c_char_p, ctypes.c_char_p,
                                    FFM_Parameters, ctypes.c_int]

    lib.ffm_train_iteration.restype = ctypes.c_float
    lib.ffm_train_iteration.argtypes = [FFM_Problem.pointer(), FFM_Model.pointer(),
                                        FFM_Parameters, ctypes.c_int]

    lib.ffm_predict_batch.restype = Float_ptr
    lib.ffm_predict_batch.argtypes = [FFM_Problem.pointer(), FFM_Model.pointer()]

    lib.ffm_load_model_c_string.restype = FFM_Model
    lib.ffm_load_model_c_string.argtypes = [ctypes.c_char_p]

    lib.ffm_save_model_c_string.argtypes = [FFM_Model.pointer(), ctypes.c_char_p]

    lib.ffm_cleanup_problem.argtypes = [FFM_Problem.pointer()]

    lib.ffm_cleanup_prediction.argtypes = [Float_ptr]

    return lib


lib = setup_lib()
srand = ctypes.CDLL('libc.so.6').srand