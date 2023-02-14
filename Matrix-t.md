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

# Matrix-t interpolation

```python
%load_ext autoreload
%autoreload 2
```

```python
import numpy as np
import jax.numpy as jnp
import jax.scipy.special as jsc
import jax
from jax import lax
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

```python
import matrix_t
```

```python
X = np.array([[-1., 1., 2.],
              [-1., 1.5, 2.]])
```

```python
px.scatter(x=X[0], y=X[1])
```

```python tags=[]
def make_std_ncmtd_right(x, X, *params):
    dfs, *other_params = params
    data = jnp.hstack((X, x[:, None]))
    len1, len2 = data.shape
    dist = matrix_t.NonCentralMatrixT(data, dfs, np.eye(len1), np.eye(len2), *other_params)
    return dist
```

```python
def logk_right(x, X, *params):
    return matrix_t.ref_log_kernel_noncentral(
        make_std_ncmtd_right(x, X, *params)
    )
```

```python tags=[]
def cond_right(x, X, *params):
    return matrix_t.ref_uni_cond_noncentral(
        make_std_ncmtd_right(x, X, *params)
    )
```

```python
dfs = 2.0
scale = 0.5
vmr = 0.01
vml = 1000.
params = (dfs, scale, vmr, vml)
axis = 1

L = np.linspace(-2., 5., 1000)
G = np.stack(np.meshgrid(L, L), -1)
lnK = jax.vmap(lambda x: jax.vmap(lambda y: logk_right(jnp.array([x, y]), X[:, [1,2]], *params))(L))(L)
go.Figure(data=[
    go.Scatter(x=X[0], y=X[1], mode='markers', marker={'color': 'darkorange'}),
    go.Contour(x=L, y=L, z=-jnp.abs(.5 - jnp.cumsum(jnp.exp(lnK - jsc.logsumexp(lnK, axis, keepdims=True)), axis)), zmin=0, contours={'coloring': 'heatmap'}, ncontours=10, colorscale='blues', transpose=True),
], layout={
    'yaxis': {'scaleanchor': 'x', 'scaleratio': 1}
}
    )
```

```python tags=[]
dfs = 2.0
scale = 0.5
vmr = 0.01
vml = 1000.
params = (dfs, scale, vmr, vml)

L = np.linspace(-2., 5., 100)
means, variances = jax.vmap(lambda x: cond_right(jnp.array([x, 0.]), X[:, [1,2]], *params))(L)
means = means[:, 1, -1]
variances = variances[:, 1, -1]

go.Figure(data=[
    go.Scatter(x=X[0], y=X[1], mode='markers', marker={'color': 'darkorange'}),
    go.Scatter(x=L, y=means, mode='lines', name='Mean', line={'color': 'darkblue'}),
    go.Scatter(x=L, y=means+np.sqrt(variances), mode='lines', name='Mean + 1 std', line={'color': 'grey'}),
    go.Scatter(x=L, y=means-np.sqrt(variances), mode='lines', name='Mean - 1 std', line={'color': 'grey'}),
], layout={
    'yaxis': {'scaleanchor': 'x', 'scaleratio': 1}
}
    )
```

```python

```
