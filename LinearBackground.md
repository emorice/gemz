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
import plotly
import colorcet
plotly.io.templates.default = go.layout.Template(layout={
    'width': 600, 'height': 600, 'autosize': False, **{
        f'{a}axis': {'showline': True, 'ticks': 'outside', 'exponentformat': 'power'}
        for a in 'xy'},
    },
    data={'contour': [{'colorbar': {'exponentformat': 'power'}, 'opacity': 0.97}]}
)
import pandas as pd
import scipy.stats
```

```python tags=[]
import block_mt as bmt
```

```python tags=[]
rng = np.random.default_rng(0)
N = 200
bg_frac = 0.5
_x_bg = rng.uniform(-1., 3., size=N)
_y_bg = rng.uniform(0., 5., size=N)

_x_sig = rng.normal(1, 0.5, size=N)
_y_sig = _x_sig + 2. + rng.normal(0., 0.2, size=N)

is_bg = rng.uniform(size=N) < bg_frac
X = np.where(is_bg, _x_bg, _x_sig)
Y = np.where(is_bg, _y_bg, _y_sig)
data = np.vstack((X, Y))
```

```python tags=[]
data_trace = go.Scatter(x=X, y=Y, mode='markers', marker={'color': 'black', 'size': 4})
go.Figure([data_trace])
```

```python tags=[]
go.Figure([go.Scatter(x=X[filt], y=Y[filt], mode='markers') for filt in (is_bg, ~is_bg)])
```

```python tags=[]
dfs = 1.
scale = 1.

observed = bmt.NonCentralMatrixT.from_params(
    dfs, left=scale*np.eye(2), right=scale*np.eye(N),
    gram_mean_left=0.5
).observe(data)

predictive_dist = (
    observed
    .post_left()
    .extend_right(scale*np.eye(1))
)

def logp(new):
    return predictive_dist.observe(new[:, None]).log_pdf()

Lx = np.linspace(-2., 4., 100)
Ly = np.linspace(-1., 6., 100)

G = np.stack(np.meshgrid(Lx, Ly), -1)

logP = jax.vmap(lambda x: jax.vmap(lambda y: logp(jnp.array([x, y])))(Ly))(Lx)
               
go.Figure(data=[
    data_trace,
    go.Contour(
        x=Lx, y=Ly, z=np.exp(logP), zmin=0,
        contours={'coloring': 'heatmap'}, ncontours=10, colorscale=colorcet.CET_L18,
        transpose=True),
], layout={
    'yaxis': {'scaleanchor': 'x', 'scaleratio': 1},
})
```

```python tags=[]
_m, _v, uni_logps = observed.uni_cond()
```

```python tags=[]
go.Figure(go.Scatter(data_trace).update({'marker': {'color': uni_logps[0], 'colorscale': colorcet.CET_L18}}))
```

```python tags=[]
def observe_weighted(log_precs, data=data, dfs=dfs):
    return (
        bmt.NonCentralMatrixT
        .from_params(
            dfs, left=jnp.eye(2), right=jnp.diag(jnp.exp(-log_precs)),
            gram_mean_left=0.5
        )
        .observe(data)
    )
```

```python tags=[]
@jax.jit
def qll(log_precs, dfs=dfs):
    print('Tracing...')
    dist = observe_weighted(log_precs)
    _m, _v, logps = dist.uni_cond()
    return jnp.sum(logps)
```

```python tags=[]
@jax.jit
def nqelbo(log_precs, prior_log_alpha, prior_log_beta):
    var_alpha = jnp.exp(prior_log_alpha) + .5 * 2
    log_var_alpha = jnp.log(var_alpha)

    qll_mean = qll(log_precs)
    log_prior_mean = jax.scipy.stats.gamma.logpdf(
        jnp.exp(log_precs),
        jnp.exp(prior_log_alpha),
        scale=jnp.exp(-prior_log_beta)
    )
    log_var_mean = jax.scipy.stats.gamma.logpdf(
        jnp.exp(log_precs),
        var_alpha,
        scale=jnp.exp(log_precs)/var_alpha
    )
    return (
        - qll_mean
        - jnp.sum(log_prior_mean)
        + jnp.sum(log_var_mean)
    )
```

```python tags=[]
opt = optax.adam(0.5)
params = dict(
    log_precs = jnp.zeros(N),
    prior_log_alpha = jnp.array(0., jnp.float32),
    prior_log_beta = jnp.array(0., jnp.float32),
)
state = opt.init(params)
trace = []
for i in tqdm(range(100)):
    val, grad = jax.value_and_grad(lambda kw: nqelbo(**kw))(params)
    trace.append({
        'iteration': i,
        'nqelbo': val,
        'prior_log_alpha': params['prior_log_alpha'],
        'prior_log_beta': params['prior_log_beta']
    })
    dp, state = opt.update(grad, state)
    for k in params:
        params[k]  += dp[k]

trace = pd.DataFrame(trace)
px.line(trace, x='iteration', y='nqelbo')
```

```python tags=[]
trace['prior_alpha'] = np.exp(trace['prior_log_alpha'].astype(np.float32))
trace['prior_beta'] = np.exp(trace['prior_log_beta'].astype(np.float32))
trace['prior_mean'] = trace['prior_alpha'] / trace['prior_beta']
trace['prior_std'] = np.sqrt(trace['prior_alpha'] / trace['prior_beta']**2)
```

```python tags=[]
px.line(trace, x='iteration', y=['prior_alpha', 'prior_beta'])
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

```python
go.Figure(
    [
        go.Scatter(
            x=np.arange(1, N+1)[filt],
            y=precs_mean[filt],
            error_y={
                #'array': None,#np.exp(log_precs_std[filt]),
                'arrayminus': (precs_mean - precs_ci_low)[filt],
                'array': (precs_ci_high - precs_mean)[filt],
                'thickness': 1,
                'color': 'lightgrey',
                'width': 1
            },
            mode='markers', name=name)
        for (filt, name) in ((is_bg, 'bg'), (~is_bg, 'signal'))
    ],
    layout={
        'yaxis': {'type': 'log'},
        'width': 1200,
    }
)
```

```python tags=[]
go.Figure(
    [
        go.Histogram(
            x=np.log10(precs_mean[filt]),
            opacity=0.5,
            histnorm='probability density',
            name=name)
        for (filt, name) in ((is_bg, 'bg'), (~is_bg, 'signal'))
    ],
    {
        'barmode': 'overlay'
    }
)
```

```python tags=[]
go.Figure(go.Scatter(data_trace).update({
    'text': np.exp(log_precs),
    'marker': {'color': log_precs, 'colorscale': colorcet.CET_D2}
}))
```

```python tags=[]
_post = observe_weighted(log_precs).post_left()

@jax.jit
def _logk_prec(new, log_prec):
    print('Tracing...')
    return (
        _post
        .extend_right(jnp.exp(-log_prec) * np.eye(1))
        .observe(new[:, None])
        .log_pdf()
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

go.Figure(data=[
    data_trace,
    go.Contour(x=Lx, y=Ly, z=jnp.exp(avg_logP), zmin=0., zmax=zm, contours={'coloring': 'heatmap'}, ncontours=10, colorscale=colorcet.CET_L18, transpose=True, colorbar={'x': 0.47}),
    go.Scatter(data_trace).update(xaxis='x2'),
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
t_opt = optax.adam(0.1)
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
        'log_dfs': params['prior_log_alpha'],
        'prior_log_beta': params['prior_log_beta']
    })
    dp, t_state = t_opt.update(grad, t_state)
    for k in t_params:
        t_params[k]  += dp[k]

t_trace = pd.DataFrame(t_trace)
px.line(t_trace, x='iteration', y='nll')
```

```python tags=[]
t_params['log_dfs']
```

```python tags=[]
logp_multit = jax.vmap(lambda x : jax.vmap(lambda y: multit_logpdf(jnp.array([x, y]), **t_params))(Ly))(Lx)
```

```python tags=[]
zm = float(jnp.max(jnp.exp(logp_multit)))

go.Figure(data=[
    data_trace,
    go.Contour(x=Lx, y=Ly, z=jnp.exp(logp_multit), zmax=zm, contours={'coloring': 'heatmap'}, ncontours=10, colorscale=colorcet.CET_L18, transpose=True, colorbar={'x': 0.47}),
    go.Scatter(data_trace).update(xaxis='x2'),
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
