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
    data={'contour': [{'colorbar': {'exponentformat': 'power'}}]}
)
```

```python tags=[]
import matrix_t as mt
```

```python tags=[]
rng = np.random.default_rng(0)
N = 100
bg_frac = 0.5
X = rng.uniform(-2., 2., size=N)
_y_bg = rng.uniform(-2., 2., size=N)
_y_sig = X + rng.normal(0., 0.1, size=N)
is_bg = rng.uniform(size=N) < bg_frac
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

observed = mt.MatrixTObservation(dfs, scale*np.eye(2), scale*np.eye(N), data)
predictive_dist = mt.from_left(
    mt.post_left(observed),
    scale*np.eye(1)
)

def logp(new):
    return mt.ref_log_pdf(
        mt.observe(predictive_dist, new[:, None])
    )

L = np.linspace(-3., 3., 100)
G = np.stack(np.meshgrid(L, L), -1)

logP = jax.vmap(lambda x: jax.vmap(lambda y: logp(jnp.array([x, y])))(L))(L)
               
go.Figure(data=[
    data_trace,
    go.Contour(x=L, y=L, z=np.exp(logP), zmin=0, contours={'coloring': 'heatmap'}, ncontours=10, colorscale=colorcet.CET_L18, transpose=True),
], layout={
    'yaxis': {'scaleanchor': 'x', 'scaleratio': 1},
})
```

```python tags=[]
_m, _v, uni_logps = mt.ref_uni_cond(observed)
```

```python tags=[]
go.Figure(go.Scatter(data_trace).update({'marker': {'color': uni_logps[0], 'colorscale': colorcet.CET_L18}}))
```

```python tags=[]
def observe_weighted(log_precs, data=data, dfs=dfs):
    return mt.MatrixTObservation(dfs, jnp.eye(2), jnp.diag(jnp.exp(-log_precs)), data)

@jax.jit
def nql(log_precs, dfs=dfs):
    print('Tracing...')
    dist = observe_weighted(log_precs)
    _m, _v, logps = mt.ref_uni_cond(dist)
    return - jnp.sum(logps) + jnp.sum(jnp.exp(log_precs))
```

```python tags=[]
opt = optax.adam(0.1)
log_precs = jnp.zeros(N)
state = opt.init(log_precs)
trace = []
for i in tqdm(range(100)):
    val, grad = jax.value_and_grad(nql)(log_precs)
    trace.append(val)
    dlp, scale = opt.update(grad, state)
    log_precs += dlp

go.Figure(
    go.Scatter(x=np.arange(len(trace)), y=trace)
)
```

```python
go.Figure(
    [
        go.Scatter(y=np.exp(log_precs[is_bg]), mode='markers', name='bg'),
        go.Scatter(y=np.exp(log_precs[~is_bg]), mode='markers', name='signal'),
    ],
    layout={
        'yaxis': {'type': 'linear'}
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
_post = mt.post_left(observe_weighted(log_precs))

@jax.jit
def _logk_prec(new, log_prec):
    print('Tracing...')
    return mt.ref_log_pdf(mt.observe(mt.from_left(_post, jnp.exp(-log_prec) * np.eye(1)), new[:, None]))

logPs = jnp.stack([
    jax.vmap(lambda x: jax.vmap(lambda y: _logk_prec(jnp.array([x, y]), lp))(L))(L)
     for lp in tqdm(log_precs)])

```

```python tags=[]
avg_logP = jsc.logsumexp(logPs, 0) - jnp.log(len(log_precs))
```

```python tags=[]
zm = float(jnp.max(jnp.exp(avg_logP)))

go.Figure(data=[
    data_trace,
    go.Contour(x=L, y=L, z=jnp.exp(avg_logP), zmin=0., zmax=zm, contours={'coloring': 'heatmap'}, ncontours=10, colorscale=colorcet.CET_L18, transpose=True, colorbar={'x': 0.46}),
    go.Scatter(data_trace).update(xaxis='x2'),
    go.Contour(x=L, y=L, z=jnp.exp(logP),
               zmin=0., zmax=zm, contours={'coloring': 'heatmap'}, ncontours=10, colorscale=colorcet.CET_L18, transpose=True,
               xaxis='x2'),
], layout={
    'xaxis1': {'domain': [0., 0.45], 'scaleanchor': 'y', 'scaleratio': 1, 'title': 'Weighted model'},
    'xaxis2': {'domain': [0.55, 1.0],  'matches': 'x1', 'title': 'Reference linear model'},
    'showlegend': False,
    'width': 1200,
})
```
