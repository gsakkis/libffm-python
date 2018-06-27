# coding: utf-8

import os
import ctypes
import itertools as it


Float_ptr = ctypes.POINTER(ctypes.c_float)


class Structure(ctypes.Structure):

    @classmethod
    def pointer(cls):
        return ctypes.POINTER(cls)


class FFM_Parameter(Structure):
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
    lib.ffm_init_model.argtypes = [FFM_Problem.pointer(), FFM_Parameter]

    lib.ffm_copy_model.argtypes = [FFM_Model.pointer(), FFM_Model.pointer()]

    lib.ffm_train_iteration.restype = ctypes.c_float
    lib.ffm_train_iteration.argtypes = [FFM_Problem.pointer(), FFM_Model.pointer(), FFM_Parameter,
                                        ctypes.c_int]

    lib.ffm_predict_array.restype = ctypes.c_float
    lib.ffm_predict_array.argtypes = [FFM_Node.pointer(), ctypes.c_int, FFM_Model.pointer()]

    lib.ffm_predict_batch.restype = Float_ptr
    lib.ffm_predict_batch.argtypes = [FFM_Problem.pointer(), FFM_Model.pointer()]

    lib.ffm_load_model_c_string.restype = FFM_Model
    lib.ffm_load_model_c_string.argtypes = [ctypes.c_char_p]

    lib.ffm_save_model_c_string.argtypes = [FFM_Model.pointer(), ctypes.c_char_p]

    lib.ffm_cleanup_problem.argtypes = [FFM_Problem.pointer()]

    lib.ffm_cleanup_prediction.argtypes = [Float_ptr]

    return lib


lib = setup_lib()
