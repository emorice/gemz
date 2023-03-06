---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.4
  kernelspec:
    display_name: venv
    language: python
    name: venv
---

# Linear model with background

```python tags=[]
%load_ext autoreload
%autoreload 2
```

```python tags=[]
import numpy as np
import jax.numpy as jnp
import jax.scipy.special as jsc
import scipy.special as sc
import jax
from jax import lax
import optax
from tqdm.auto import tqdm
import plotly.express as px
import plotly.graph_objects as go
import plotly_template
from plotly_template import write, plotly_init
plotly_init()
import colorcet

import pandas as pd
import scipy.stats
```

```python tags=[]
import block_mt as bmt
import data as dg
```

```python tags=[]
rng = np.random.default_rng(0)
N = 200
bg_frac = 0.5

dist = dg.mixture((
    dg.rectangle([-1, 0.], [3., 5.]),
    scipy.stats.multivariate_normal(
        (1., 3.),
        ((0.25, 0.25),
         (0.25, 0.3))
    )),
    (bg_frac, 1. - bg_frac)
)

labels, data = dist.rvs(size=N, random_state=1)
df = dg.mixture_to_df(labels, data, ('background', 'signal'), ('x', 'y'))

data = data.T
X, Y = data
is_bg = labels == 0
```

```python tags=[]
data_fig = (
    px.scatter(df, x='x', y='y')
    .update_traces(marker={'color': 'black', 'size': 4})
    .update_yaxes(scaleanchor='x', scaleratio=1)
)
data_fig
```

```python tags=[]
write(px.scatter(df, x='x', y='y', color='label'), 'linbg_data')
```

```python tags=[]
dfs = 1.
scale = 1.

dist = bmt.NonCentralMatrixT.from_params(
    dfs, left=scale*np.eye(2), right=scale*np.eye(N),
    gram_mean_left=0.5
)

predictive_dist = (
    dist
    .extend(data, scale*np.eye(1), axis=0)
)

def logp(new):
    return predictive_dist.log_pdf(new[:, None])

Lx = np.linspace(-2., 4., 100)
Ly = np.linspace(-1., 6., 100)

G = np.stack(np.meshgrid(Lx, Ly), -1)

logP = jax.vmap(lambda x: jax.vmap(lambda y: logp(jnp.array([x, y])))(Ly))(Lx)
               
write(
    go.Figure(data_fig).add_contour(
        x=Lx, y=Ly, z=np.exp(logP), zmin=0,
        contours={'coloring': 'heatmap'}, ncontours=10, colorscale=colorcet.CET_L18,
        transpose=True
    ),
    'linbg_null'
)
```

```python tags=[]
_m, _v, uni_logps = dist.uni_cond(data)
```

```python tags=[]
go.Figure(data_fig).update_traces(marker={'color': uni_logps[0]/np.log(10), 'colorscale': colorcet.CET_L18, 'showscale': True})
```

```python tags=[]
def dist_weighted(log_precs, data=data, dfs=dfs):
    return (
        bmt.NonCentralMatrixT
        .from_params(
            dfs, left=jnp.eye(2), right=jnp.diag(jnp.exp(-log_precs)),
            gram_mean_left=0.5
        )
    )
```

```python tags=[]
@jax.jit
def ll(log_precs, dfs=dfs):
    print('Tracing...')
    dist = dist_weighted(log_precs)
    return dist.log_pdf(data)
```

```python tags=[]
@jax.jit
def qll(log_precs, dfs=dfs):
    print('Tracing...')
    dist = dist_weighted(log_precs)
    _m, _v, logps = dist.uni_cond(data)
    return jnp.sum(logps)
```

```python tags=[]
@jax.jit
def nqelbo(log_precs, prior_log_alpha):
    # Bijectors
    prior_alpha = jnp.exp(prior_log_alpha)
    precs = jnp.exp(log_precs)

    # Analytical optimizations
    var_alpha = prior_alpha + .5 * 2
    prior_scale = jnp.mean(precs) / prior_alpha

    qll_mean = qll(log_precs)
    log_prior_mean = jax.scipy.stats.gamma.logpdf(
        precs,
        prior_alpha,
        scale=prior_scale,
    )
    log_var_mean = jax.scipy.stats.gamma.logpdf(
        precs,
        var_alpha,
        scale=precs/var_alpha
    )
    return (
        - qll_mean
        - jnp.sum(log_prior_mean)
        + jnp.sum(log_var_mean)
    )
```

```python tags=[]
opt = optax.adam(0.2)
params = dict(
    log_precs = jnp.zeros(N),
    prior_log_alpha = jnp.array(0., jnp.float32),
)
state = opt.init(params)
trace = []
for i in tqdm(range(100)):
    val, grad = jax.value_and_grad(lambda kw: nqelbo(**kw))(params)
    trace.append({
        'iteration': i,
        'nqelbo': val,
        'prior_log_alpha': params['prior_log_alpha'],
        'prior_scale': np.mean(jnp.exp(params['log_precs'])) * jnp.exp(-params['prior_log_alpha'])
    })
    dp, state = opt.update(grad, state)
    for k in params:
        params[k]  += dp[k]

trace = pd.DataFrame(trace)
px.line(trace, x='iteration', y='nqelbo')
```

```python tags=[]
trace['prior_alpha'] = np.exp(trace['prior_log_alpha'].astype(np.float32))
trace['prior_scale'] = trace['prior_scale'].astype(np.float32)
trace['prior_mean'] = trace['prior_alpha'] * trace['prior_scale']
trace['prior_std'] = np.sqrt(trace['prior_alpha']) * trace['prior_scale']
```

```python tags=[]
px.line(
    trace.melt(id_vars='iteration', value_vars=['prior_alpha', 'prior_scale']),
    x='iteration', y='value', facet_col='variable', width=1200
).update_layout({'yaxis2.matches': None, 'yaxis2.showticklabels': True})
```

```python
px.line(trace, x='iteration', y=['prior_mean', 'prior_std'])
```

```python tags=[]
log_alpha = jnp.log(jnp.exp(params['prior_log_alpha']) + .5 * 2)
```

```python tags=[]
log_precs = params['log_precs']
```

```python tags=[]
precs_mean = np.exp(log_precs)
precs_ci_low, precs_ci_high = [
    scipy.stats.gamma.isf(q, np.exp(log_alpha), scale=np.exp(log_precs - log_alpha))
    for q in (0.95, 0.05)
]
```

```python tags=[]
write(
    px.scatter(
        df.assign(
            precision=precs_mean,
            precision_error_minus=precs_mean - precs_ci_low,
            precision_error_plus=precs_ci_high - precs_mean,
        ).sort_values('precision').reset_index(drop=True),
        y='precision',
        error_y='precision_error_plus',
        error_y_minus='precision_error_minus',
        color='label',
        width=1200,
        log_y=True
    ).update_traces(error_y={'thickness': 1, 'width': 1}),
    'linbg_params'
)
```

```python tags=[]
px.histogram(
    df.assign(**{'precision (log 10)': np.log10(precs_mean)}),
    x='precision (log 10)',
    color='label',
    histnorm='probability density',
    barmode='overlay'
)
```

```python tags=[]
go.Figure(data_fig).update_traces(
    text=np.exp(log_precs),
    hovertemplate='x: %{x}<br>y: %{y}<br>λ: %{text:g}',
    marker={'color': log_precs/np.log(10), 'colorscale': colorcet.CET_D2, 'showscale': True, 'colorbar.title': 'Precision (log 10)'}
).update_layout(width=700)
```

```python tags=[]
_post = dist_weighted(log_precs).post(data)

@jax.jit
def _logk_prec(new, log_prec):
    print('Tracing...')
    return (
        _post
        .extend(jnp.exp(-log_prec) * np.eye(1))
        .log_pdf(new[:, None])
    )

logPs = jnp.stack([
    jax.vmap(lambda x: jax.vmap(lambda y: _logk_prec(jnp.array([x, y]), lp))(Ly))(Lx)
     for lp in tqdm(log_precs)])

```

```python tags=[]
avg_logP = jsc.logsumexp(logPs, 0) - jnp.log(len(log_precs))
```

```python tags=[]
zm = float(jnp.max(jnp.exp(avg_logP)))

write(
    go.Figure(data=[
        data_fig.data[0],
        go.Contour(x=Lx, y=Ly, z=jnp.exp(avg_logP), zmin=0., zmax=zm, contours={'coloring': 'heatmap'}, ncontours=10, colorscale=colorcet.CET_L18, transpose=True),
    ], layout={
        'xaxis1': {'scaleanchor': 'y', 'scaleratio': 1, 'title': 'Weighted model'},
        'showlegend': False,
    }),
    'linbg_final'
)
go.Figure(data=[
    data_fig.data[0],
    go.Contour(x=Lx, y=Ly, z=jnp.exp(avg_logP), zmin=0., zmax=zm, contours={'coloring': 'heatmap'}, ncontours=10, colorscale=colorcet.CET_L18, transpose=True, colorbar={'x': 0.47}),
    go.Scatter(data_fig.data[0]).update(xaxis='x2'),
    go.Contour(x=Lx, y=Ly, z=jnp.exp(logP),
               zmin=0., zmax=zm, contours={'coloring': 'heatmap'}, ncontours=10, colorscale=colorcet.CET_L18, transpose=True,
               showscale=False, xaxis='x2'),
], layout={
    'xaxis1': {'domain': [0., 0.45], 'scaleanchor': 'y', 'scaleratio': 1, 'title': 'Weighted model'},
    'xaxis2': {'domain': [0.55, 1.0],  'matches': 'x1', 'title': 'Reference linear model'},
    'showlegend': False,
    'width': 1200,
})
```

```python tags=[]
def multit_logpdf(x, mean, sqrt_prec, log_dfs):
    dim = 2
    tx = (x - mean) @ sqrt_prec
    misfits = jnp.sum(tx**2, -1)
    dfs = jnp.exp(log_dfs)
    Nx = jnp.atleast_2d(x).shape[-2]
    return (
        - 0.5 * (dfs + dim) * jnp.sum(jnp.log(1 + misfits))
        + Nx * jnp.linalg.slogdet(sqrt_prec)[1]
        + Nx * (jsc.gammaln(0.5 * (dfs + dim)) - jsc.gammaln(0.5*dfs) - 0.5 * dim * jnp.log(jnp.pi))
    )
```

```python tags=[]
@jax.jit
def nll_student(mean, sqrt_prec, log_dfs):
    return - multit_logpdf(data.T, mean, sqrt_prec, log_dfs)
```

```python tags=[]
t_opt = optax.adam(0.05)
t_params = dict(
    mean = jnp.zeros(2),
    sqrt_prec = jnp.eye(2),
    log_dfs = 0.
)
t_state = t_opt.init(t_params)
t_trace = []
for i in tqdm(range(100)):
    val, grad = jax.value_and_grad(lambda kw: nll_student(**kw))(t_params)
    t_trace.append({
        'iteration': i,
        'nll': val,
        'log_dfs': t_params['log_dfs']
    })
    dp, t_state = t_opt.update(grad, t_state)
    for k in t_params:
        t_params[k]  += dp[k]

t_trace = pd.DataFrame(t_trace)
px.line(t_trace, x='iteration', y='nll')
```

```python tags=[]
px.line(t_trace.assign(dfs=lambda df: np.exp(df.log_dfs.astype(np.float32))), x='iteration', y='dfs')
```

```python tags=[]
logp_multit = jax.vmap(lambda x : jax.vmap(lambda y: multit_logpdf(jnp.array([x, y]), **t_params))(Ly))(Lx)
```

```python tags=[]
zm = float(jnp.max(jnp.exp(logp_multit)))

go.Figure(data=[
    data_fig.data[0],
    go.Contour(x=Lx, y=Ly, z=jnp.exp(logp_multit), zmax=zm, contours={'coloring': 'heatmap'}, ncontours=10, colorscale=colorcet.CET_L18, transpose=True, colorbar={'x': 0.47}),
    go.Scatter(data_fig.data[0]).update(xaxis='x2'),
    go.Contour(x=Lx, y=Ly, z=jnp.exp(logP),
               zmin=0., zmax=zm, contours={'coloring': 'heatmap'}, ncontours=10, colorscale=colorcet.CET_L18, transpose=True,
               showscale=False, xaxis='x2'),
], layout={
    'xaxis1': {'domain': [0., 0.45], 'scaleanchor': 'y', 'scaleratio': 1, 'title': 'Weighted model'},
    'xaxis2': {'domain': [0.55, 1.0],  'matches': 'x1', 'title': 'Reference linear model'},
    'showlegend': False,
    'width': 1200,
})
```
