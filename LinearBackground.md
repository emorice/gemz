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
        for a in 'xy'}
})
```

```python tags=[]
import matrix_t
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
def make_std_mtd_left(new, data, dfs, scale):
    data = jnp.hstack((data, new[:, None]))
    len1, len2 = data.shape
    dist = matrix_t.MatrixT(data, dfs, scale*np.eye(len1), scale*np.eye(len2))
    return dist

def logk_left(new, data, *params):
    return matrix_t.ref_log_kernel(
        make_std_mtd_left(new, data, *params)
    )

dfs = 1.0
scale = 1.0

L = np.linspace(-3., 3., 100)
G = np.stack(np.meshgrid(L, L), -1)

lnK = jax.vmap(lambda x: jax.vmap(lambda y: logk_left(jnp.array([x, y]), data, dfs, scale))(L))(L)
               
go.Figure(data=[
    data_trace,
    go.Contour(x=L, y=L, z=lnK, zmin=0, contours={'coloring': 'heatmap'}, ncontours=10, colorscale=colorcet.CET_L18, transpose=True),
], layout={
    'yaxis': {'scaleanchor': 'x', 'scaleratio': 1},
})
```

```python tags=[]
dist = matrix_t.MatrixT(data, dfs, scale*np.eye(2), scale*np.eye(N))
_m, _v, logps = matrix_t.ref_uni_cond(dist)
```

```python tags=[]
go.Figure(go.Scatter(data_trace).update({'marker': {'color': logps[0], 'colorscale': colorcet.CET_L18}}))
```

```python tags=[]
def nql(log_weights, dfs=dfs):
    dist = matrix_t.MatrixT(data, dfs, jnp.eye(2), jnp.diag(jnp.exp(log_weights)))
    _m, _v, logps = matrix_t.ref_uni_cond(dist)
    return - jnp.sum(logps) + 0.01*jnp.var(jnp.exp(-log_weights))
```

```python tags=[]
opt = optax.adam(0.4)
log_weights = jnp.zeros(N)
state = opt.init(log_weights)
trace = []
for i in tqdm(range(100)):
    val, grad = jax.value_and_grad(nql)(log_weights)
    trace.append(val)
    dlw, scale = opt.update(grad, state)
    log_weights += dlw

go.Figure(
    go.Scatter(x=np.arange(len(trace)), y=trace)
)
```

```python
go.Figure(
    [
        go.Scatter(y=np.exp(-log_weights[is_bg]), mode='markers', name='bg'),
        go.Scatter(y=np.exp(-log_weights[~is_bg]), mode='markers', name='signal'),
    ],
    layout={
        'yaxis': {'type': 'linear'}
    }
)
```

```python tags=[]
go.Figure(go.Scatter(data_trace).update({
    'text': np.exp(-log_weights),
    'marker': {'color': -log_weights, 'colorscale': colorcet.CET_D8}
}))
```

```python

```
