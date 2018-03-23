floatX = "float32"
intX = "int32"

from .core import get_logger
from .core import scan
from .core import dot
from .core import get_params_dict
from .core import run_loop
from .nodes import make_numpy_weights
from .nodes import make_numpy_biases
from .plot import viridis_cm
