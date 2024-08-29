"""
Cross-validation utils
"""

import numpy as np

from . import methods, ops

methods.add_module('cv', __name__)
OPS_AWARE = True

LOSSES = {}

def loss(name):
    """
    Record a named loss function
    """
    def _wrap(function):
        LOSSES[name] = function
        return function
    return _wrap

@loss('RSS')
def rss_loss(method, model, test):
    """
    Evaluate a fitted mode with a classical RSS loss
    """

    predictions = method.predict_loo(model, test)
    return np.sum((test - predictions)**2)

@loss('iRSS')
def irss_loss(method, model, test):
    """
    Classical RSS loss except not aggregated over the working dimensions

    (so only summed over the replicates in the fold)
    """

    predictions = method.predict_loo(model, test)
    return np.sum((test - predictions)**2, 0)

@loss('NAIC')
def naic_loss(method, model, test):
    """
    Negative Akaike Information Criterion loss

    Untested.
    """
    log_pdfs = method.log_pdf(model, test)
    return - np.sum(log_pdfs)

@loss('GEOM')
def geom_loss(method, model, test):
    """
    Geometric aggregate of RSS over dimensions.

    More robust than RSS with heteroskedastic data.
    """
    predictions = method.predict_loo(model, test)
    dim_squares = np.sum((test - predictions)**2, -1)

    return np.sum(np.log(dim_squares))

def fit(data, inner, fold_count=10, seed=0, loss_name="RSS", grid_size=20,
        grid=None, grid_max=None, _ops=ops):
    """
    Fit and eval the given method on folds of data

    Args:
        data: N1 x N2. Models are fixed-dimension N2 x N2, and cross-validation
            is performed along N1.
        grid: if given, use this explicitely defined grid instead of generating
            one.
    """

    specs = _ops.build_eval_grid(
                inner, data, fold_count, loss_name,
                grid_size, grid_max, grid,
                seed
            )

    reduced_grid = _ops.extract_cv(specs)

    best_model = _ops.select_best(reduced_grid)

    return {
        'inner': inner,
        'loss_name': loss_name,
        'selected': best_model,
        'fit': (best_model, _ops.fit(best_model, data)),
        'grid': specs,
        }

EVAL_REQUIRES = 'fit' # Only the "fit" key above to be passed to predict_loo

def predict_loo(model_fit, new_data):
    """
    Linear shrinkage loo prediction for the best model found during cv.
    """
    model_spec, inner_fit = model_fit
    inner_model = methods.get(model_spec['model'])
    return inner_model.predict_loo(inner_fit, new_data)

def get_name(spec):
    """
    Readable description
    """
    return f"{spec['model']}/{spec['inner']['model']}"

class OneDimCV:
    """
    Args:
        spec_name: name of the corresponding key in the spec dictionnaries for
            this model.
        display_name: name to use in plot.
        log: whether to plot on a log-scale
    """
    def __init__(self, spec_name, display_name, log=True):
        self.spec_name = spec_name
        self.display_name = display_name
        self.log = log

    def make_grid_specs(self, partial_spec, grid):
        """
        Generate model specs from grid values
        """
        return [{
            **partial_spec,
            self.spec_name: size
            }
            for size in grid
            ]

    def get_grid_axes(self, specs):
        """
        Compact summary of the variable parameter of a list of models
        """
        return [{
            'name': self.display_name,
            'log': self.log,
            'values': [ s[self.spec_name] for s in specs ]
            }]

class Int1dCV(OneDimCV):
    """
    One-dim cv with integer log scale bounded by data dimensions
    """
    def make_grid(self, data, grid_size, grid_max=None):
        """
        Simple logarithmic scale of not more than grid_size entries.

        Grid can be smaller than requested.
        """

        if grid_max is None:
            grid_max = min(data.shape)

        return [ int(size) for size in np.unique(
                np.int32(np.floor(
                    np.exp(
                        np.linspace(0., np.log(grid_max), grid_size)
                        )
                    ))
                )]

class Real1dCV(OneDimCV):
    """
    One-dim cv with positive real log scale
    """
    def make_grid(self, data, grid_size, grid_max=None):
        """
        A standard grid of prior vars to screen

        Logarithmic from 0.01 to 100, which should be reasonable if the dataset is
        standardized.
        """
        # We could extract a scale from the data here
        _ = data

        if grid_max is None:
            grid_max = 100

        return 10**np.linspace(-2, np.log10(grid_max), grid_size)

class CartesianCV:
    """
    Cartesian-product CV along several dimensions.

    Args: any number of 1D cv objects
    """
    def __init__(self, *cvs):
        self.cvs = cvs

    def make_grid(self, data, grid_size, grid_max=None):
        """
        Independent even-sized grids along each dimension
        """
        if grid_max is not None:
            raise NotImplementedError('CartesianCV does not support grid_max')

        grid_size_dim = int(np.exp(np.log(grid_size) / len(self.cvs)))

        return [ cv.make_grid(data, grid_size_dim) for cv in self.cvs ]

    def make_grid_specs(self, partial_spec, grid):
        """
        Cartesian grid building
        """
        specs = [partial_spec]

        for cv_dim, grid_dim in zip(self.cvs, grid):
            next_specs = []
            for spec in specs:
                next_specs.extend(cv_dim.make_grid_specs(spec, grid_dim))
            specs = next_specs

        return specs

    def get_grid_axes(self, specs):
        """
        All axes in order
        """
        return sum((cv.get_grid_axes(specs) for cv in self.cvs), start=[])
