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
    data[subset] *= 0.5

    # Split train/test
    train = data[:, :len2_train]
    test = data[:, len2_train:]

    return train, test, (subset, source, dest)

def gen_lowdim(len1, seed=0, **_):
    """
    Easy 2-D case
    """
    corrs = [
        np.array([[1., +0.7], [+0.7, 1.]]),
        np.array([[1., -0.5], [-0.5, 1.]])
        ]
    means = [
        np.array([0, -2]),
        np.array([1, 1])
    ]

    rng = np.random.default_rng(seed)
    classes = rng.choice(2, size=len1)

    data = rng.normal(size=(len1, 2))
    for cls in [0, 1]:
        sel = classes == cls
        data[sel] = (data[sel][:, None, :] @ corrs[cls])[:, 0, :] + means[cls]

    return data, data, (classes == 1, np.array([1, 0]), np.array([0, 1]))

def plot_nonlinearity(ddata, nonlinearity, resps=None):
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
                hovertext=None if resps is None
                    else list(map(str, resps[:, sub].T)),
                mode='markers',
                )
            for sub in (subset, ~subset)
            ],
        layout={
            'height': 900
            }
        )

def plot_pcs(data, subset, resps=None):
    pcs, _, _ = np.linalg.svd(data, full_matrices=False)

    return go.Figure(
        data=[
            go.Scatter(
                x=pcs[:, 0][sub],
                y=pcs[:, 1][sub],
                hovertext=None if resps is None
                    else list(map(str, resps[:, sub].T)),
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

    # Dummy test case
    #train, test, nonlinearity = gen_lowdim(len1=1000)

    train_c1 = train[nonlinearity[0]]
    train_c2 = train[~nonlinearity[0]]

    eps = 0.45
    model_defs = {
        'kmeans': ('kmeans', {'n_groups': 2}),
        'gmm_free': ('gmm', {
            'n_groups': 2,
            'n_init': 10,
            #'bayesian': True,
            #'init_params': 'random_from_data'
            }),
        'gmm_bayes': ('gmm', {
            'n_groups': 2,
            'n_init': 10,
            'bayesian': True,
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
            'n_groups': 2,
            'seed': 0,
            'barrier_strength': 0.1,
            #'init_resps': np.stack(( 0.5 + eps - 2 * eps * nonlinearity[0], 0.5 - eps + 2 * eps * nonlinearity[0],))
            })
    }

    # Uncomment to try only a subset of models
    # model_defs = { k: d for k, d in model_defs.items() if k == 'igmm' }

    model_fits = {
        name: models.get(algo).fit(train.T, **options)
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
        resps = model_fits[model].get('responsibilities')
        figs_nl_model.append(
            plot_nonlinearity(train, nl_model, resps)
            .update_layout(
                title=f'{model.capitalize()} twist recovery'
                )
            )

        figs_nl_model.append(
            plot_pcs(train, nl_model[0], resps)
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
