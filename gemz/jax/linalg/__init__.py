"""
Jax versions of virtual arrays
"""

import gemz.linalg
from gemz.jax.linalg.array_api import MetaJaxAPI

class ScaledIdentity(gemz.linalg.ScaledIdentity):
    aa = MetaJaxAPI
