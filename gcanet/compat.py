"""Compatibility for licensed SMPL pickles that depend on legacy Chumpy.

The original body-model arrays are unchanged. These aliases only restore names
removed by recent NumPy/Python versions before an old pickle imports Chumpy.
"""
from collections import namedtuple
import inspect
import numpy as np


def prepare_legacy_smpl():
    aliases = {'bool': np.bool_, 'int': int, 'float': float, 'complex': complex,
               'object': object, 'unicode': np.str_, 'str': str}
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)
    if not hasattr(inspect, 'getargspec'):
        ArgSpec = namedtuple('ArgSpec', 'args varargs keywords defaults')
        def getargspec(function):
            spec = inspect.getfullargspec(function)
            return ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)
        inspect.getargspec = getargspec
