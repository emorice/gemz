"""
Dataset with a nonlinearity
"""

import numpy as np

import plotly.graph_objects as go

from gemz.cases import case
from gemz.reporting import open_report, write_fig

def gen_hypertwist(len1, len2_train, len2_test, small_dims, small_var=0.2, large_var=1., seed=0):
    """
    A high-dimensional roof shape

    Made of two pieces with ... dims

    len1 is the "big" dim, split along the roof.
    """

    len2 = len2_train + len2_test

    spectrum = np.hstack((
        np.ones(len2 - small_dims) * large_var,
        np.ones(small_dims) * small_var
        ))
    rng = np.random.default_rng(seed)

    # Generate a len2xlen2 matrix
    basis = rng.normal(0., 1., size=(len2, len2))
    # Orthonormalize it, you get a Haar distribution I think
    basis, _ = np.linalg.qr(basis, mode='complete')
    assert np.allclose(basis @ basis.T, np.eye(len2))
    # Build a matrix with the desired spectrum
    sq_cov = (basis * np.sqrt(spectrum)) @ basis.T

    # Generate sperical random values
    data = rng.normal(0., 1., size=(len1, len2))

    # Transform it
    data = data @ sq_cov

    # Define nonlinearity
    support = rng.normal(0., 1., size=len2)
    source = np.hstack((
        rng.normal(0., 1., size=len2 - small_dims),
        np.zeros(small_dims)
        ))
    source = basis @ source
    source /= np.sum(source**2)**.5
    dest = np.hstack((
        np.zeros(len2 - small_dims),
        rng.normal(0., 1., size=small_dims)
        ))
    dest = basis @ dest
    dest /= np.sum(dest**2)**.5

    # Apply nonlinearity
    subset = data @ support > 0.
    dsub = data[subset]
    data[subset] = (
        dsub
        + (dsub @ source)[:, None] * (dest - source)
        - (dsub @ dest)[:, None] * (dest + source)
        )

    # Split train/test
    train = data[:, :len2_train]
    test = data[:, len2_train:]

    return train, test, (subset, source, dest)


def plot_nonlinearity(ddata, nonlinearity):
    """
    Draw a projection making the curvature change visible
    """
    dlen = ddata.shape[-1]

    subset, source, dest = nonlinearity

    proj_source = ddata @ source[:dlen]
    proj_dest = ddata @ dest[:dlen]

    return go.Figure(
        data=[
            go.Scatter(
                x=proj_source[sub],
                y=proj_dest[sub],
                mode='markers',
                )
            for sub in (subset, ~subset)
            ],
        layout={
            'height': 900
            }
        )

@case
def nonlinear(_, case_name, report_path):
    """
    High-dimensional dataset with two distinct sets of linear invariants
    """

    train, test, nonlinearity = gen_hypertwist(
        len1=200,
        len2_train=100,
        len2_test=1,
        small_dims=10
        )

    fig_nl = plot_nonlinearity(train, nonlinearity)

    with open_report(report_path, case_name) as stream:
        write_fig(stream, fig_nl)
