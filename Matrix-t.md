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

```python tags=[]
import block_mt as bmt
```

```python
X = np.array([[-1., 1., 2.],
              [-1., 1.5, 2.]])
```

```python
data_trace = go.Scatter(x=X[0], y=X[1], mode='markers', marker={'color': 'darkblue'})
go.Figure(data_trace)
```

```python
Xsub = X[:, [1, 2]]
scale = .5

data = bmt.NonCentralMatrixT.from_params(
    dfs=2.0,
    left=jnp.eye(Xsub.shape[0])*scale,
    right=jnp.eye(Xsub.shape[1])*scale,
    gram_mean_left=1e-6 / scale,
    gram_mean_right=None
).observe(Xsub)

predictive = (
    data
    .post(axis=0)
    .extend(np.eye(1)*scale, axis=0)
)

axis = 1

L = np.linspace(-2., 5., 100)
G = np.stack(np.meshgrid(L, L), -1)
```

```python tags=[]
def pred_logp(x, y):
    return predictive.observe(jnp.array([x, y])[:, None]).log_pdf()
```

```python tags=[]
logP = jax.vmap(lambda x: jax.vmap(lambda y: pred_logp(x, y))(L))(L)
```

```python tags=[]
# Well normalized
dL = L[1] - L[0]
_Z = jnp.exp(jsc.logsumexp(logP))*dL*dL
print(_Z)
```

```python tags=[]
go.Figure(data=[
    data_trace,
    go.Contour(x=L, y=L, z=jnp.exp(logP), zmin=0, contours={'coloring': 'heatmap'}, ncontours=10, colorscale=colorcet.CET_L18, transpose=True),
], layout={
    'yaxis': {'scaleanchor': 'x', 'scaleratio': 1},
    'title': 'Predictive distribution'
})
```

```python tags=[]
disc_clnp = logP - jsc.logsumexp(logP, axis, keepdims=True) - jnp.log(dL)
go.Figure(data=[
    data_trace,
    go.Contour(x=L, y=L, z=jnp.exp(disc_clnp), zmin=0, contours={'coloring': 'heatmap'}, ncontours=10, colorscale=colorcet.CET_L18, transpose=True),
], layout={
    'yaxis': {'scaleanchor': 'x', 'scaleratio': 1},
    'title': 'Conditional by discrete integration'
})
```

```python tags=[]
def pred_uni(x, y):
    return predictive.observe(jnp.array([[x, y]]).T).uni_cond()
```

```python tags=[]
_means, _vars, all_clnps = jax.vmap(lambda x: jax.vmap(lambda y: pred_uni(x, y))(L))(L)
clnp = all_clnps[:, :, axis, -1]
go.Figure(data=[
    go.Scatter(x=X[0], y=X[1], mode='markers', marker={'color': 'darkblue'}),
    go.Contour(x=L, y=L, z=jnp.exp(clnp), contours={'coloring': 'heatmap'}, ncontours=10, colorscale=colorcet.CET_L18, transpose=True),
], layout={
    'yaxis': {'scaleanchor': 'x', 'scaleratio': 1},
    'title': 'Conditional by direct calculation'
})
```

```python tags=[]
go.Figure(go.Scatter(x=disc_clnp.flatten(), y=clnp.flatten(), mode='markers', marker={'size': 1.0}))
```

```python tags=[]
L = np.linspace(-2., 5., 100)
means, variances, _logps = jax.vmap(lambda x: pred_uni(x, 0.))(L)
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
