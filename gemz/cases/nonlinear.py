"""
Dataset with a nonlinearity
"""

import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from gemz import models
from gemz.cases import case
from gemz.reporting import open_report, write_fig

def gen_hypertwist(len1, len2_train, len2_test, small_dims, small_var=0.01, large_var=1., seed=0):
    """
    A high-dimensional twisted shape

    Made of two pieces with ... dims

    len1 is the "big" dim, split along the twist.
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
    #support = rng.normal(0., 1., size=len2)
    support = np.hstack((
        rng.normal(0., 1., size=len2 - small_dims),
        np.zeros(small_dims)
        ))
    support = basis @ support
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
    #data[subset] *= 0.5

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

def plot_pcs(data, subset):
    pcs, _, _ = np.linalg.svd(data, full_matrices=False)

    return go.Figure(
        data=[
            go.Scatter(
                x=pcs[:, 0][sub],
                y=pcs[:, 1][sub],
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
        len1=1000,
        len2_train=100,
        len2_test=1,
        small_dims=90
        )

    train_c1 = train[nonlinearity[0]]
    train_c2 = train[~nonlinearity[0]]

    model_defs = {
        'kmeans': ('kmeans', {'n_clusters': 2}),
        'gmm_free': ('gmm', {
            'n_groups': 2,
            'n_init': 10,
            #'bayesian': True,
            'init_params': 'random_from_data'
            }),
        'gmm_forced': ('gmm', {
            'n_groups': 2,
            'weights_init': np.ones(2)/2.,
            'means_init': np.stack((np.mean(train_c1, 0), np.mean(train_c2, 0))),
            'precisions_init': np.linalg.inv(np.stack((
                np.cov(train_c1.T),
                np.cov(train_c2.T))
                )),
            #'precisions_init': np.stack([np.eye(train.shape[-1])]*2),
            'n_init': 1
            }),
        'igmm': ('igmm', {
            'n_groups': 2
            })
    }

    model_fits = {
        name: models.get(algo).fit(train, **options)
        for name, (algo, options) in model_defs.items()
    }

    fig_nl = (
        plot_nonlinearity(train, nonlinearity)
        .update_layout(
            title="Ground truth twist"
            )
        )

    figs_nl_model = []
    for model in model_defs:
        nl_model = (
            model_fits[model]['groups'] == 0,
            *nonlinearity[1:]
            )
        figs_nl_model.append(
            plot_nonlinearity(train, nl_model)
            .update_layout(
                title=f'{model.capitalize()} twist recovery'
                )
            )

        figs_nl_model.append(
            plot_pcs(train, nl_model[0])
            .update_layout(
                title=f'{model.capitalize()} PCs'
                )
            )

    with open_report(report_path, case_name) as stream:
        write_fig(stream,
            plot_pcs(train, nonlinearity[0]),
            px.line(y=np.linalg.svd(train, compute_uv=False)),
            fig_nl,
            *figs_nl_model
            )
