"""
OpenMC MGXS Library and Coupled Depletion Framework

Provides tools for managing burnup-parametrized MGXS libraries and
running coupled transport-depletion simulations with proper normalization.
"""

from .mgxs_library_manager import MGXSLibraryManager
from .coupled_depletion_driver import CoupledDepletionDriver
from .sub_stepped_depletion_driver import SubSteppedDepletionDriver
from .predictor_corrector_driver import PredictorCorrectorDepletionDriver

__all__ = [
    'MGXSLibraryManager',
    'CoupledDepletionDriver',
    'SubSteppedDepletionDriver',
    'PredictorCorrectorDepletionDriver',
]
__version__ = '1.0.0'
