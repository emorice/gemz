"""
Shrinkage models based on Ledoit--Wolf spectral deconvolution
"""

import numpy as np

from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri

numpy2ri.activate()

def spectrum(data):
    """
    Spectrum obtained by numerically inverting the discretized spectral
    dispersion equations.

    Args:
        data: N1 x N2
    Returns:
        N1-long estimated spectrum
    """

    nlshrink = importr("nlshrink")

    # Working dimension last
    return np.array(nlshrink.tau_estimate(data.T))[::-1]
