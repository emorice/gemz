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

```python
def lnk(X, dfs, scale, vml, vmr):
    len1, len2 = X.shape
    return matrix_t.ref_log_kernel_centered(X, dfs, np.eye(len1), np.eye(len2), scale, vml, vmr)
```

```python
def cond_right(x, X, *params):
    return lnk(jnp.hstack((X, x[:, None])), *params)
```

```python
dfs = 2.0
scale = 1.0
vmr = 0.01
vml = 1000.
params = (dfs, scale, vmr, vml)

L = np.linspace(-2., 5., 1000)
G = np.stack(np.meshgrid(L, L), -1)
lnK = jax.vmap(lambda x: jax.vmap(lambda y: cond_right(jnp.array([x, y]), X[:, [1,2]], *params))(L))(L)
go.Figure(data=[
    go.Scatter(x=X[0], y=X[1], mode='markers', marker={'color': 'darkorange'}),
    go.Contour(x=L, y=L, z=-jnp.abs(.5 - jnp.cumsum(jnp.exp(lnK - jsc.logsumexp(lnK, -1, keepdims=True)), -1)), zmin=0, contours={'coloring': 'heatmap'}, ncontours=10, colorscale='blues', transpose=True),
], layout={
    'yaxis': {'scaleanchor': 'x', 'scaleratio': 1}
}
    )
```

```python

```
