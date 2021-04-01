from pycallgraph2 import PyCallGraph
from pycallgraph2 import Config
from pycallgraph2.output import GraphvizOutput
from pycallgraph2 import GlobbingFilter


from breakpoints import *
import numpy as np

def filter_fun(name):
    return name[0] == "_"

# def filtercalls(clas, func, full):
#     mod_ignore = ['shutil','scipy.optimize','re','os','sys','json', 'numpy', 'np', 'logging']
#     func_ignore = filter(filter_fun, func) ++ ['logging']
#     clas_ignore = ['pdb', 'logging']
#     return modul not in mod_ignore and func not in func_ignore and clas not in clas_ignore

config = Config(max_depth=7)
config.trace_filter = GlobbingFilter(exclude=[
    'logging.*',
    '__main__',
    "all",
    "any",
    "abc.__instancecheck__",
    "abc.__subclasscheck__",
    "utils.Breakpoints.__init__",
    "copy",
    "repeat",
    "vstack",
    "allclose",
    "concatenate",
    "isclose",
    "argmin",
    "copyto",
    "utils",
    "atleast_2d",
    "sum",
    "nanargmin",
    "column_stack",
    "result_type",
    "threading.RLock",
    "cumsum",
    "ndim",
    "nan_to_num",
    "outer",
    "typing.new_type",
    "apply_along_axis",
    "around",
])
graphviz = GraphvizOutput(output_file='filter_max_depth.png')


with PyCallGraph(output=graphviz, config=config):
    n = 50
    ones = np.ones(n).reshape((n, 1)).astype("float64")
    y = np.arange(1, n+1).astype("float64")
    X = np.copy(y).reshape((n, 1))
    # X = np.column_stack((ones, X))
    # X = ones
    # X[5] = np.nan
    y[14:] = y[14:] * 0.03
    y[5] = np.nan
    y[34:] = y[34:] + 10


    bp = Breakpoints(X, y, use_mp=False, verbosity=0).breakpoints
    # print("Breakpoints:", bp)
    # print()

print("done")




